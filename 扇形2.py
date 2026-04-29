import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import copy
from time import time
from 采样 import configure_sampling, get_samples, get_validation_samples, print_sampling_info

# =========================================================
# 0. 硬件与全局设置
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 使用 float32，训练速度和显存占用更适合大批量采样
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# 关闭 TF32，保持标准 float32 计算路径
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# =========================================================
# 1. 材料参数（严格按题意）
# =========================================================
E_val = 2.0e11
v_val = 0.3

E = torch.tensor(E_val, dtype=DTYPE, device=device)
v = torch.tensor(v_val, dtype=DTYPE, device=device)
lmda = E * v / ((1 + v) * (1 - 2 * v))
mu = E / (2 * (1 + v))

# =========================================================
# 2. 几何参数（严格按题意）
# 坐标系：
#   x: 水平向右
#   y: 竖直向上（高度方向）
#   z: 垂直于 x-y 平面
# 柱坐标关系：
#   r = sqrt(x^2 + z^2), x = r cos(theta), z = r sin(theta)
# =========================================================
R_inner = 0.187
R_outer = 0.392
H_total = 0.07

theta_half_deg = 19.8
theta_half = np.deg2rad(theta_half_deg)
theta_min = -theta_half
theta_max = theta_half

# 凹槽参数
x_groove = 0.2895
R_groove = 0.05
y_groove_bottom = 0.05
y_groove_top = 0.07

configure_sampling(
    device_in=device,
    dtype_in=DTYPE,
    R_inner_in=R_inner,
    R_outer_in=R_outer,
    H_total_in=H_total,
    theta_min_in=theta_min,
    theta_max_in=theta_max,
    x_groove_in=x_groove,
    R_groove_in=R_groove,
    y_groove_bottom_in=y_groove_bottom,
    y_groove_top_in=y_groove_top,
)

# =========================================================
# 3. 载荷参数
# 底面 y=0 施加沿 +Y 的总力 F_total，换算为均布面力 q_load
# =========================================================
F_total = 76400.0

# 底面面积 = 扇环面积
bottom_area = 0.5 * (theta_max - theta_min) * (R_outer**2 - R_inner**2)
q_load = F_total / bottom_area  # Pa

print(f"[INFO] bottom_area = {bottom_area:.6e} m^2")
print(f"[INFO] q_load      = {q_load:.6e} Pa")

# =========================================================
# 4. 训练时的参考尺度（只用于损失归一化，不改变物理模型）
# =========================================================
L_ref = R_outer - R_inner
SIGMA_REF = q_load
U_REF = q_load * L_ref / E_val
PDE_REF = SIGMA_REF / L_ref               # div(sigma) 的量纲尺度

print(f"[INFO] L_ref       = {L_ref:.6e} m (radial thickness)")
print(f"[INFO] U_REF       = {U_REF:.6e} m")
print(f"[INFO] SIGMA_REF   = {SIGMA_REF:.6e} Pa")
print(f"[INFO] PDE_REF     = {PDE_REF:.6e} Pa/m")

# =========================================================
# 4.1 训练控制参数
# =========================================================
resample_every = 50
adam_epochs = 2000
plateau_patience = 120
plateau_factor = 0.5
plateau_min_lr = 1.0e-6

# PDE = equilibrium + constitutive consistency. The raw Eq term is the main
# bottleneck in the current logs, so keep it explicit. The constitutive term
# should not be too large from epoch 0, otherwise it can flatten the
# displacement gradients before the thickness response is learned.
w_eq_pde = 10.0
w_const_pde_start = 2.0
w_const_pde_end = 4.0
const_ramp_start = 600
const_ramp_end = 1600

# The bottom load is a resultant-force condition in practice. A pointwise
# traction MSE can look modest even when the mean traction is still 15-20%
# below the target, so add an explicit mean-traction penalty.
load_mean_weight = 5.0

def get_const_weight(epoch):
    if epoch <= const_ramp_start:
        return w_const_pde_start
    if epoch >= const_ramp_end:
        return w_const_pde_end
    t = (epoch - const_ramp_start) / (const_ramp_end - const_ramp_start)
    return w_const_pde_start + t * (w_const_pde_end - w_const_pde_start)

# =========================================================
# 5. 输入归一化边界（保持模板风格）
# 注意这里用整个包围盒做 [-1,1] 归一化
# =========================================================
lb = torch.tensor([-R_outer, 0.0, -R_outer], dtype=DTYPE, device=device)
ub = torch.tensor([ R_outer, H_total, R_outer], dtype=DTYPE, device=device)

# =========================================================
# 6. 模型定义（整体架构保持不变）
# 仍然输出：
# u, v, w, sxx, syy, szz, sxy, syz, sxz
# =========================================================
class PINN3D(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.SiLU()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for i in range(len(self.linears) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)

        nn.init.xavier_normal_(self.linears[-1].weight.data)
        nn.init.zeros_(self.linears[-1].bias.data)

    def forward(self, x_in):
        a = 2.0 * (x_in - lb) / (ub - lb) - 1.0
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        out = self.linears[-1](a)

        # 位移与应力采用不同参考尺度输出，避免网络在初始化阶段陷入“全零应力”解。
        u = U_REF * out[:, 0:1]
        v = U_REF * out[:, 1:2]
        w = U_REF * out[:, 2:3]
        sxx = SIGMA_REF * out[:, 3:4]
        syy = SIGMA_REF * out[:, 4:5]
        szz = SIGMA_REF * out[:, 5:6]
        sxy = SIGMA_REF * out[:, 6:7]
        syz = SIGMA_REF * out[:, 7:8]
        sxz = SIGMA_REF * out[:, 8:9]
        return u, v, w, sxx, syy, szz, sxy, syz, sxz

# =========================================================
# 9. Loss 计算
# 六项损失结构不改，只替换成扇形模型对应含义
# =========================================================
def get_gradients(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def traction_from_stress(sxx, syy, szz, sxy, syz, sxz, nx, ny, nz):
    tx = sxx * nx + sxy * ny + sxz * nz
    ty = sxy * nx + syy * ny + syz * nz
    tz = sxz * nx + syz * ny + szz * nz
    return tx, ty, tz

def compute_losses(
    model,
    X_col,
    X_bottom_load,
    X_groove_bottom_fix,
    X_groove_side_free,
    X_side_free,
    X_top_free,
    X_radial_free,
    const_weight=None
):
    if const_weight is None:
        const_weight = w_const_pde_end

    # -----------------------------------------------------
    # 1. PDE Loss：平衡方程 + 本构方程
    # -----------------------------------------------------
    u, v, w, sxx, syy, szz, sxy, syz, sxz = model(X_col)

    gu = get_gradients(u, X_col)
    gv = get_gradients(v, X_col)
    gw = get_gradients(w, X_col)

    ux, uy, uz = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    vx, vy, vz = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    wx, wy, wz = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]

    # 平衡方程 div(sigma)=0
    gsxx = get_gradients(sxx, X_col)
    gsyy = get_gradients(syy, X_col)
    gszz = get_gradients(szz, X_col)
    gsxy = get_gradients(sxy, X_col)
    gsyz = get_gradients(syz, X_col)
    gsxz = get_gradients(sxz, X_col)

    res_x = gsxx[:, 0:1] + gsxy[:, 1:2] + gsxz[:, 2:3]
    res_y = gsxy[:, 0:1] + gsyy[:, 1:2] + gsyz[:, 2:3]
    res_z = gsxz[:, 0:1] + gsyz[:, 1:2] + gszz[:, 2:3]

    # 本构方程
    div_u = ux + vy + wz

    sxx_c = lmda * div_u + 2.0 * mu * ux
    syy_c = lmda * div_u + 2.0 * mu * vy
    szz_c = lmda * div_u + 2.0 * mu * wz
    sxy_c = mu * (uy + vx)
    syz_c = mu * (vz + wy)
    sxz_c = mu * (uz + wx)

    loss_const = torch.mean(((sxx_c - sxx) / SIGMA_REF) ** 2) + \
                 torch.mean(((syy_c - syy) / SIGMA_REF) ** 2) + \
                 torch.mean(((szz_c - szz) / SIGMA_REF) ** 2) + \
                 torch.mean(((sxy_c - sxy) / SIGMA_REF) ** 2) + \
                 torch.mean(((syz_c - syz) / SIGMA_REF) ** 2) + \
                 torch.mean(((sxz_c - sxz) / SIGMA_REF) ** 2)

    loss_eq = torch.mean((res_x / PDE_REF) ** 2) + \
              torch.mean((res_y / PDE_REF) ** 2) + \
              torch.mean((res_z / PDE_REF) ** 2)

    loss_pde = w_eq_pde * loss_eq + const_weight * loss_const

    # -----------------------------------------------------
    # 2. 底面受力边界（代替模板里的 inner pressure）
    # y = 0, outward n = (0,-1,0)
    # prescribed traction = (0, q_load, 0)
    # -----------------------------------------------------
    _, _, _, sxx_b, syy_b, szz_b, sxy_b, syz_b, sxz_b = model(X_bottom_load)

    nx_b = torch.zeros((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)
    ny_b = -torch.ones((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)
    nz_b = torch.zeros((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)

    tx_b, ty_b, tz_b = traction_from_stress(sxx_b, syy_b, szz_b, sxy_b, syz_b, sxz_b, nx_b, ny_b, nz_b)

    target_tx_b = torch.zeros_like(tx_b)
    target_ty_b = torch.full_like(ty_b, q_load)
    target_tz_b = torch.zeros_like(tz_b)

    loss_bottom_load = torch.mean(((tx_b - target_tx_b) / SIGMA_REF) ** 2 +
                                  ((ty_b - target_ty_b) / SIGMA_REF) ** 2 +
                                  ((tz_b - target_tz_b) / SIGMA_REF) ** 2)
    mean_ty_ratio = torch.mean(ty_b) / q_load
    loss_bottom_mean = (mean_ty_ratio - 1.0) ** 2
    loss_bottom_load = loss_bottom_load + load_mean_weight * loss_bottom_mean

    # -----------------------------------------------------
    # 3. 凹槽底面固定边界
    # -----------------------------------------------------
    u_fix, v_fix, w_fix, _, _, _, _, _, _ = model(X_groove_bottom_fix)
    loss_groove_bottom_fixed = torch.mean((u_fix / U_REF) ** 2 +
                                          (v_fix / U_REF) ** 2 +
                                          (w_fix / U_REF) ** 2)

    # -----------------------------------------------------
    # 4. 凹槽侧壁自由边界
    # rho = R_groove, outward n = ((x-x_groove)/rho, 0, z/rho)
    # -----------------------------------------------------
    _, _, _, sxx_g, syy_g, szz_g, sxy_g, syz_g, sxz_g = model(X_groove_side_free)

    x_g = X_groove_side_free[:, 0:1]
    z_g = X_groove_side_free[:, 2:3]
    rho_g = torch.sqrt((x_g - x_groove) ** 2 + z_g ** 2 + 1e-30)

    nx_g = (x_g - x_groove) / rho_g
    ny_g = torch.zeros_like(nx_g)
    nz_g = z_g / rho_g

    tx_g, ty_g, tz_g = traction_from_stress(sxx_g, syy_g, szz_g, sxy_g, syz_g, sxz_g, nx_g, ny_g, nz_g)

    loss_groove_side_free = torch.mean((tx_g / SIGMA_REF) ** 2 +
                                       (ty_g / SIGMA_REF) ** 2 +
                                       (tz_g / SIGMA_REF) ** 2)

    # -----------------------------------------------------
    # 5. 内外圆柱侧面自由边界
    # -----------------------------------------------------
    _, _, _, sxx_s, syy_s, szz_s, sxy_s, syz_s, sxz_s = model(X_side_free)

    x_s = X_side_free[:, 0:1]
    z_s = X_side_free[:, 2:3]
    r_s = torch.sqrt(x_s**2 + z_s**2 + 1e-30)

    # 对内外圆柱面，法向分别可取 ±(x/r, 0, z/r)
    # 因为目标是 traction=0，符号不会影响平方损失
    nx_s = x_s / r_s
    ny_s = torch.zeros_like(nx_s)
    nz_s = z_s / r_s

    tx_s, ty_s, tz_s = traction_from_stress(sxx_s, syy_s, szz_s, sxy_s, syz_s, sxz_s, nx_s, ny_s, nz_s)

    loss_side_free = torch.mean((tx_s / SIGMA_REF) ** 2 +
                                (ty_s / SIGMA_REF) ** 2 +
                                (tz_s / SIGMA_REF) ** 2)

    # -----------------------------------------------------
    # 6. 上表面自由边界
    # y = H_total, outward n=(0,1,0)
    # traction = [sxy, syy, syz]
    # -----------------------------------------------------
    _, _, _, sxx_t, syy_t, szz_t, sxy_t, syz_t, sxz_t = model(X_top_free)

    nx_t = torch.zeros((X_top_free.shape[0], 1), dtype=DTYPE, device=device)
    ny_t = torch.ones((X_top_free.shape[0], 1), dtype=DTYPE, device=device)
    nz_t = torch.zeros((X_top_free.shape[0], 1), dtype=DTYPE, device=device)

    tx_t, ty_t, tz_t = traction_from_stress(sxx_t, syy_t, szz_t, sxy_t, syz_t, sxz_t, nx_t, ny_t, nz_t)

    loss_top_free = torch.mean((tx_t / SIGMA_REF) ** 2 +
                               (ty_t / SIGMA_REF) ** 2 +
                               (tz_t / SIGMA_REF) ** 2)

    # -----------------------------------------------------
    # 7. 两个径向侧面自由边界
    # theta = ±theta_half
    # 法向可取 e_theta = (-sin theta, 0, cos theta)
    # traction = 0
    # -----------------------------------------------------
    _, _, _, sxx_r, syy_r, szz_r, sxy_r, syz_r, sxz_r = model(X_radial_free)

    x_r = X_radial_free[:, 0:1]
    z_r = X_radial_free[:, 2:3]
    theta_r = torch.atan2(z_r, x_r)

    nx_r = -torch.sin(theta_r)
    ny_r = torch.zeros_like(nx_r)
    nz_r = torch.cos(theta_r)

    tx_r, ty_r, tz_r = traction_from_stress(sxx_r, syy_r, szz_r, sxy_r, syz_r, sxz_r, nx_r, ny_r, nz_r)

    loss_radial_free = torch.mean((tx_r / SIGMA_REF) ** 2 +
                                  (ty_r / SIGMA_REF) ** 2 +
                                  (tz_r / SIGMA_REF) ** 2)

    return (
        loss_pde,
        loss_eq,
        loss_const,
        loss_bottom_load,
        loss_groove_bottom_fixed,
        loss_groove_side_free,
        loss_side_free,
        loss_top_free,
        loss_radial_free,
        mean_ty_ratio.detach(),
    )

def total_weighted_loss_from_losses(losses, base_w):
    l_pde, _, _, l_load, l_fix, l_gside, l_side, l_top, l_rad, _ = losses
    return (
        base_w[0] * l_pde +
        base_w[1] * l_load +
        base_w[2] * l_fix +
        base_w[3] * l_gside +
        base_w[4] * l_side +
        base_w[5] * l_top +
        base_w[6] * l_rad
    )

def _sample_points_for_plot(X, max_points):
    pts = X.detach().cpu().numpy()
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]

def show_sampling_points(
    X_col,
    X_bottom_load,
    X_groove_bottom_fix,
    X_groove_side_free,
    X_side_free,
    X_top_free,
    X_radial_free,
    max_points_each=2500,
    save_path="sampling_points_preview.png",
):
    groups = [
        ("Interior", X_col, "#4C78A8", 2),
        ("Bottom load", X_bottom_load, "#F58518", 8),
        ("Groove fixed", X_groove_bottom_fix, "#E45756", 10),
        ("Groove side free", X_groove_side_free, "#72B7B2", 7),
        ("Inner/outer side free", X_side_free, "#54A24B", 7),
        ("Top free", X_top_free, "#B279A2", 7),
        ("Radial free", X_radial_free, "#FF9DA6", 7),
    ]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, X, color, size in groups:
        pts = _sample_points_for_plot(X, max_points_each)
        ax.scatter(
            pts[:, 0],
            pts[:, 2],
            pts[:, 1],
            s=size,
            c=color,
            alpha=0.65,
            label=f"{name} ({X.shape[0]})",
            depthshade=False,
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("y (m)")
    ax.set_title("Sampling Points Before Training")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((2.0 * R_outer, 2.0 * R_outer, H_total))
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    print(f"[INFO] sampling point preview saved to: {save_path}")
    plt.show()

# =========================================================
# 10. 主程序（整体架构保持不变）
# =========================================================
if __name__ == '__main__':
    model = PINN3D([3, 192, 192, 192, 192, 192, 9]).to(device=device, dtype=DTYPE)

    # 生成数据
    (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free
    ) = get_samples()
    show_sampling_points(
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
    )

    # 基础权重（保留模板风格）
    # Order: PDE, bottom load, groove fix, groove side free, inner/outer side
    # free, top free, radial side free. The latest run showed Top/Rad free
    # traction losses staying high, so the free-surface weights are raised
    # while keeping the PDE term explicit through w_eq_pde/w_const_pde.
    base_w = torch.tensor([1.0, 240.0, 120.0, 20.0, 35.0, 50.0, 40.0], dtype=DTYPE, device=device)

    # Adam 阶段
    print("Starting Adam training...")
    print(f"[INFO] Adam epochs = {adam_epochs}")
    print(
        f"[INFO] LR scheduler = ReduceLROnPlateau("
        f"factor={plateau_factor:g}, patience={plateau_patience}, min_lr={plateau_min_lr:g})"
    )
    print(
        f"[INFO] PDE loss = {w_eq_pde:g} * Eq + "
        f"Const weight ramp {w_const_pde_start:g}->{w_const_pde_end:g} "
        f"(epoch {const_ramp_start}->{const_ramp_end})"
    )
    print(f"[INFO] bottom load mean penalty weight = {load_mean_weight:g}")
    print("[INFO] loss weights = fixed base_w")
    print_sampling_info()

    optimizer_adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        foreach=False,
        fused=False,
        capturable=(device.type == "cuda")
    )

    scheduler = ReduceLROnPlateau(
        optimizer_adam,
        mode="min",
        factor=plateau_factor,
        patience=plateau_patience,
        threshold=1.0e-4,
        threshold_mode="rel",
        min_lr=plateau_min_lr,
    )

    t0 = time()
    for epoch in range(adam_epochs + 1):
        if epoch > 0 and epoch % resample_every == 0:
            (
                X_col,
                X_bottom_load,
                X_groove_bottom_fix,
                X_groove_side_free,
                X_side_free,
                X_top_free,
                X_radial_free
            ) = get_samples()

        optimizer_adam.zero_grad()

        const_weight = get_const_weight(epoch)
        l_pde, l_eq, l_const, l_load, l_fix, l_gside, l_side, l_top, l_rad, mean_ty_ratio = compute_losses(
            model,
            X_col,
            X_bottom_load,
            X_groove_bottom_fix,
            X_groove_side_free,
            X_side_free,
            X_top_free,
            X_radial_free,
            const_weight=const_weight
        )

        loss = (
            base_w[0] * l_pde +
            base_w[1] * l_load +
            base_w[2] * l_fix +
            base_w[3] * l_gside +
            base_w[4] * l_side +
            base_w[5] * l_top +
            base_w[6] * l_rad
        )

        if epoch % 50 == 0:
            elapsed = time() - t0
            weighted_pde = base_w[0] * l_pde
            weighted_bc = (
                base_w[1] * l_load +
                base_w[2] * l_fix +
                base_w[3] * l_gside +
                base_w[4] * l_side +
                base_w[5] * l_top +
                base_w[6] * l_rad
            )
            print(
                f"Ep {epoch:5d} | Total:{loss.item():.4e} | PDE:{l_pde.item():.4e} | "
                f"Eq:{l_eq.item():.4e} | Const:{l_const.item():.4e} | "
                f"WConst:{const_weight:.3g} | "
                f"Load:{l_load.item():.4e} | Fix:{l_fix.item():.4e} | "
                f"GrooveSide:{l_gside.item():.4e} | Side:{l_side.item():.4e} | "
                f"Top:{l_top.item():.4e} | Rad:{l_rad.item():.4e} | "
                f"W_PDE:{weighted_pde.item():.4e} | W_BC:{weighted_bc.item():.4e} | "
                f"mean_ty/q:{mean_ty_ratio.item():.4e} | "
                f"LR:{optimizer_adam.param_groups[0]['lr']:.3e} | "
                f"time:{elapsed:.1f}s"
            )

        loss.backward()
        optimizer_adam.step()
        scheduler.step(loss.detach())

    # L-BFGS 阶段
    adam_state = copy.deepcopy(model.state_dict())
    validation_samples = get_validation_samples()
    lbfgs_const_weight = w_const_pde_end
    val_losses_before = compute_losses(
        model,
        *validation_samples,
        const_weight=lbfgs_const_weight
    )
    val_loss_before = total_weighted_loss_from_losses(val_losses_before, base_w).detach()

    print("\nStarting L-BFGS...")
    print(f"[L-BFGS] validation loss before = {val_loss_before.item():.4e}")

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=2000,
        line_search_fn="strong_wolfe"
    )

    iter_count = [0]

    def closure():
        optimizer_lbfgs.zero_grad()

        l_pde, _, _, l_load, l_fix, l_gside, l_side, l_top, l_rad, _ = compute_losses(
            model,
            X_col,
            X_bottom_load,
            X_groove_bottom_fix,
            X_groove_side_free,
            X_side_free,
            X_top_free,
            X_radial_free,
            const_weight=lbfgs_const_weight
        )

        loss = (
            base_w[0] * l_pde +
            base_w[1] * l_load +
            base_w[2] * l_fix +
            base_w[3] * l_gside +
            base_w[4] * l_side +
            base_w[5] * l_top +
            base_w[6] * l_rad
        )

        loss.backward()
        iter_count[0] += 1

        if iter_count[0] % 20 == 0:
            print(f"L-BFGS Iter: {iter_count[0]} | Loss: {loss.item():.4e}")

        return loss

    optimizer_lbfgs.step(closure)

    val_losses_after = compute_losses(
        model,
        *validation_samples,
        const_weight=lbfgs_const_weight
    )
    val_loss_after = total_weighted_loss_from_losses(val_losses_after, base_w).detach()
    print(f"[L-BFGS] validation loss after  = {val_loss_after.item():.4e}")

    if val_loss_after > 1.05 * val_loss_before:
        model.load_state_dict(adam_state)
        print("[L-BFGS] validation got worse; restored Adam-final weights.")
    else:
        print("[L-BFGS] validation improved or stayed stable; kept L-BFGS weights.")

    print("Training Finished.")

    save_path = "pinn_sector_groove_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已成功保存至: {save_path}")
