import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyDOE2 import lhs
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import os
import copy
from time import time

# =========================================================
# 0. 硬件与全局设置
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 建议使用 float64，适合固体力学 PINN
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

# 关闭 TF32，避免双精度/高阶导数时出现额外数值扰动
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
adam_milestones = [800, 1400, 1800]

# The boundary terms converge quickly in this case. Keeping the adaptive
# GradNorm-style weights on tends to keep amplifying already-small boundary
# losses, so the default run uses fixed, PDE-focused weights.
use_adaptive_weights = False

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
# 6. 几何辅助函数
# =========================================================
def lhs_tensor(dim, n):
    n = int(n)
    if n <= 0:
        return torch.empty((0, dim), dtype=DTYPE, device=device)
    return torch.tensor(lhs(dim, n), dtype=DTYPE, device=device)

def to_cartesian(r, theta, y):
    """
    扇形模型使用：
        x = r cos(theta)
        y = y
        z = r sin(theta)
    """
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    return torch.cat([x, y, z], dim=1)

def groove_rho_torch(x, z):
    return torch.sqrt((x - x_groove) ** 2 + z ** 2 + 1e-30)

def is_in_groove_void_torch(X):
    """
    判断点是否在凹槽空腔内部
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]
    rho = groove_rho_torch(x, z)
    cond_r = rho <= R_groove
    cond_y = (y >= y_groove_bottom) & (y <= y_groove_top)
    return cond_r & cond_y

def is_in_main_sector_torch(X):
    """
    判断点是否在主体扇环柱内部（未扣除凹槽）
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]

    r = torch.sqrt(x**2 + z**2 + 1e-30)
    theta = torch.atan2(z, x)

    cond_r = (r >= R_inner) & (r <= R_outer)
    cond_t = (theta >= theta_min) & (theta <= theta_max)
    cond_y = (y >= 0.0) & (y <= H_total)
    return cond_r & cond_t & cond_y

def is_in_solid_torch(X):
    return is_in_main_sector_torch(X) & (~is_in_groove_void_torch(X))

# =========================================================
# 7. 数据采样（保持模板 get_samples 架构）
# 返回 7 组点，保持模板整体结构不变
#
# X_col          : 内部配点
# X_bottom_load  : 底面受力边界
# X_groove_bottom_fix : 凹槽底面固定边界
# X_groove_side_free  : 凹槽侧壁自由边界
# X_side_free    : 内外圆柱侧面自由边界
# X_top_free     : 上表面自由边界（去掉凹槽开口）
# X_radial_free  : 两个扇形径向侧面自由边界
# =========================================================
def sample_interior_points(N_col):
    pts_list = []
    total = 0
    while total < N_col:
        m = max(2 * (N_col - total), 1024)
        raw = lhs_tensor(3, m)

        r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        theta = theta_min + raw[:, 1:2] * (theta_max - theta_min)
        y = raw[:, 2:3] * H_total

        X = to_cartesian(r, theta, y)
        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X_col = torch.cat(pts_list, dim=0)[:N_col]
    return X_col.requires_grad_(True)

def sample_groove_edge_interior_points(N_edge):
    """
    Interior collocation points concentrated near the re-entrant groove bottom
    edge: rho ~= R_groove, y ~= y_groove_bottom. This is where the stress field
    changes fastest, so uniform interior sampling under-resolves it.
    """
    pts_list = []
    total = 0
    radial_span = 0.008
    vertical_span = 0.006

    while total < N_edge:
        m = max(4 * (N_edge - total), 1024)
        raw = lhs_tensor(3, m)

        alpha = 2.0 * np.pi * raw[:, 0:1]
        rho = R_groove + (2.0 * raw[:, 1:2] - 1.0) * radial_span
        y = y_groove_bottom + (2.0 * raw[:, 2:3] - 1.0) * vertical_span

        x = x_groove + rho * torch.cos(alpha)
        z = rho * torch.sin(alpha)
        X = torch.cat([x, y, z], dim=1)

        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X_edge = torch.cat(pts_list, dim=0)[:N_edge]
    return X_edge.requires_grad_(True)

def sample_groove_influence_interior_points(N_inf):
    """
    Interior PDE points in a wider annular influence zone around the groove.
    The narrow edge sampler captures the peak, while this sampler helps the
    network learn the stress diffusion path from the groove into the body.
    """
    pts_list = []
    total = 0
    rho_min = R_groove
    rho_max = min(0.135, R_outer - x_groove + 0.035)
    y_min = max(0.0, y_groove_bottom - 0.035)
    y_max = H_total

    while total < N_inf:
        m = max(4 * (N_inf - total), 1024)
        raw = lhs_tensor(3, m)

        alpha = 2.0 * np.pi * raw[:, 0:1]
        rho = rho_min + raw[:, 1:2] * (rho_max - rho_min)
        y = y_min + raw[:, 2:3] * (y_max - y_min)

        x = x_groove + rho * torch.cos(alpha)
        z = rho * torch.sin(alpha)
        X = torch.cat([x, y, z], dim=1)

        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X_inf = torch.cat(pts_list, dim=0)[:N_inf]
    return X_inf.requires_grad_(True)

def sample_bottom_loaded(N_b):
    raw = lhs_tensor(2, N_b)
    r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
    theta = theta_min + raw[:, 1:2] * (theta_max - theta_min)
    y = torch.zeros((N_b, 1), dtype=DTYPE, device=device)
    X = to_cartesian(r, theta, y)
    return X.requires_grad_(True)

def sample_groove_bottom(Nb):
    pts_list = []
    total = 0
    while total < Nb:
        m = max(2 * (Nb - total), 256)
        raw = lhs_tensor(2, m)
        rho = torch.sqrt(raw[:, 0:1]) * R_groove
        alpha = 2.0 * np.pi * raw[:, 1:2]

        x = x_groove + rho * torch.cos(alpha)
        z = rho * torch.sin(alpha)
        y = torch.full((m, 1), y_groove_bottom, dtype=DTYPE, device=device)
        X = torch.cat([x, y, z], dim=1)

        mask = is_in_main_sector_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X = torch.cat(pts_list, dim=0)[:Nb]
    return X

def sample_groove_side(Ns):
    pts_list = []
    total = 0
    while total < Ns:
        m = max(2 * (Ns - total), 256)
        raw = lhs_tensor(2, m)
        alpha = 2.0 * np.pi * raw[:, 0:1]
        y = y_groove_bottom + raw[:, 1:2] * (y_groove_top - y_groove_bottom)

        x = x_groove + R_groove * torch.cos(alpha)
        z = R_groove * torch.sin(alpha)
        X = torch.cat([x, y, z], dim=1)

        mask = is_in_main_sector_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X = torch.cat(pts_list, dim=0)[:Ns]
    return X

def sample_groove_bottom_fixed(N_fix):
    """
    固定边界仅为凹槽底面 y = y_groove_bottom。
    """
    X = sample_groove_bottom(N_fix)
    return X.requires_grad_(True)

def sample_groove_side_free(Ns):
    """
    凹槽侧壁自由边界：
    sqrt((x-x_groove)^2 + z^2) = R_groove, y in [y_groove_bottom, y_groove_top]
    """
    X = sample_groove_side(Ns)
    return X.requires_grad_(True)

def sample_side_free(N_side):
    """
    内外圆柱面自由边界
    """
    N_each = N_side // 2

    raw_out = lhs_tensor(2, N_each)
    theta_out = theta_min + raw_out[:, 0:1] * (theta_max - theta_min)
    y_out = raw_out[:, 1:2] * H_total
    r_out = torch.full((N_each, 1), R_outer, dtype=DTYPE, device=device)
    X_outer = to_cartesian(r_out, theta_out, y_out)

    raw_in = lhs_tensor(2, N_side - N_each)
    theta_in = theta_min + raw_in[:, 0:1] * (theta_max - theta_min)
    y_in = raw_in[:, 1:2] * H_total
    r_in = torch.full((N_side - N_each, 1), R_inner, dtype=DTYPE, device=device)
    X_inner = to_cartesian(r_in, theta_in, y_in)

    X = torch.cat([X_outer, X_inner], dim=0)
    return X.requires_grad_(True)

def sample_top_free(N_top):
    """
    上表面 y=H_total，自由边界，但需剔除凹槽开口
    """
    pts_list = []
    total = 0
    while total < N_top:
        m = max(2 * (N_top - total), 512)
        raw = lhs_tensor(2, m)
        r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        theta = theta_min + raw[:, 1:2] * (theta_max - theta_min)
        y = torch.full((m, 1), H_total, dtype=DTYPE, device=device)

        X = to_cartesian(r, theta, y)

        # 顶面固体区域要去掉凹槽开口
        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X = torch.cat(pts_list, dim=0)[:N_top]
    return X.requires_grad_(True)

def sample_radial_free(N_rad):
    """
    两个扇形径向侧面：theta = ±theta_half
    这里不需要递归补点，也不需要用 is_in_solid_torch 过滤。
    原因：凹槽不会与这两个径向侧面相交。
    """
    N_rad = int(N_rad)
    if N_rad <= 0:
        return torch.empty((0, 3), dtype=DTYPE, device=device, requires_grad=True)

    N_each = N_rad // 2
    N_rem = N_rad - N_each

    # theta = theta_min 面
    raw1 = lhs_tensor(2, N_each)
    if N_each > 0:
        r1 = torch.sqrt(raw1[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        y1 = raw1[:, 1:2] * H_total
        th1 = torch.full((N_each, 1), theta_min, dtype=DTYPE, device=device)
        X1 = to_cartesian(r1, th1, y1)
    else:
        X1 = torch.empty((0, 3), dtype=DTYPE, device=device)

    # theta = theta_max 面
    raw2 = lhs_tensor(2, N_rem)
    if N_rem > 0:
        r2 = torch.sqrt(raw2[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        y2 = raw2[:, 1:2] * H_total
        th2 = torch.full((N_rem, 1), theta_max, dtype=DTYPE, device=device)
        X2 = to_cartesian(r2, th2, y2)
    else:
        X2 = torch.empty((0, 3), dtype=DTYPE, device=device)

    X = torch.cat([X1, X2], dim=0)
    return X.requires_grad_(True)

def is_in_main_sector_torch(X):
    """
    判断点是否在主体扇环柱内部（未扣除凹槽）
    """
    EPS = 1e-10

    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]

    r = torch.sqrt(x**2 + z**2 + 1e-30)
    theta = torch.atan2(z, x)

    cond_r = (r >= R_inner - EPS) & (r <= R_outer + EPS)
    cond_t = (theta >= theta_min - EPS) & (theta <= theta_max + EPS)
    cond_y = (y >= 0.0 - EPS) & (y <= H_total + EPS)
    return cond_r & cond_t & cond_y

def get_samples():
    N_col_uniform = 22000
    N_col_influence = 8000
    N_col_edge = 6000
    N_b = 4000
    N_fix = 4000
    N_groove_side = 4000
    N_rad = 4000

    X_col_uniform = sample_interior_points(N_col_uniform)
    X_col_influence = sample_groove_influence_interior_points(N_col_influence)
    X_col_edge = sample_groove_edge_interior_points(N_col_edge)
    X_col = torch.cat([X_col_uniform, X_col_influence, X_col_edge], dim=0).requires_grad_(True)
    X_bottom_load = sample_bottom_loaded(N_b)
    X_groove_bottom_fix = sample_groove_bottom_fixed(N_fix)
    X_groove_side_free = sample_groove_side_free(N_groove_side)
    X_side_free = sample_side_free(N_b)
    X_top_free = sample_top_free(N_b)
    X_radial_free = sample_radial_free(N_rad)

    return (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
    )

def print_sampling_info():
    print("[INFO] train collocation samples:")
    print("       uniform=22000, groove_influence=8000, groove_edge=6000")
    print("       total interior=36000")
    print("[INFO] validation collocation samples:")
    print("       uniform=6500, groove_influence=2500, groove_edge=2000")
    print("       total interior=11000")

def get_validation_samples():
    N_col_uniform = 6500
    N_col_influence = 2500
    N_col_edge = 2000
    N_b = 1200
    N_fix = 1200
    N_groove_side = 1200
    N_rad = 1200

    X_col_uniform = sample_interior_points(N_col_uniform)
    X_col_influence = sample_groove_influence_interior_points(N_col_influence)
    X_col_edge = sample_groove_edge_interior_points(N_col_edge)
    X_col = torch.cat([X_col_uniform, X_col_influence, X_col_edge], dim=0).requires_grad_(True)
    X_bottom_load = sample_bottom_loaded(N_b)
    X_groove_bottom_fix = sample_groove_bottom_fixed(N_fix)
    X_groove_side_free = sample_groove_side_free(N_groove_side)
    X_side_free = sample_side_free(N_b)
    X_top_free = sample_top_free(N_b)
    X_radial_free = sample_radial_free(N_rad)

    return (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
    )

# =========================================================
# 8. 模型定义（整体架构保持不变）
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
# 9. 自适应权重（整体架构保持不变）
# =========================================================
class AdaptiveWeighter:
    def __init__(self, num_losses, alpha=0.9):
        self.weights = torch.ones(num_losses, dtype=DTYPE, device=device)
        self.alpha = alpha

    def update(self, model, losses):
        last_layer = model.linears[-1].weight
        grads_norm = []
        for loss in losses:
            grad = torch.autograd.grad(loss, last_layer, retain_graph=True)[0]
            grad_norm = torch.mean(torch.abs(grad))
            grads_norm.append(grad_norm)

        grads_norm = torch.stack(grads_norm)

        target_weights = grads_norm[0] / (grads_norm + 1e-10)
        target_weights = torch.clamp(target_weights, min=0.01, max=100.0)
        target_weights[0] = max(target_weights[0].item(), 1.0)

        self.weights = self.alpha * self.weights + (1 - self.alpha) * target_weights.detach()
        return self.weights

# =========================================================
# 10. Loss 计算
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

def total_weighted_loss_from_losses(losses, base_w, adapt_w):
    l_pde, _, _, l_load, l_fix, l_gside, l_side, l_top, l_rad, _ = losses
    return (
        base_w[0] * adapt_w[0] * l_pde +
        base_w[1] * adapt_w[1] * l_load +
        base_w[2] * adapt_w[2] * l_fix +
        base_w[3] * adapt_w[3] * l_gside +
        base_w[4] * adapt_w[4] * l_side +
        base_w[5] * adapt_w[5] * l_top +
        base_w[6] * adapt_w[6] * l_rad
    )

# =========================================================
# 11. 主程序（整体架构保持不变）
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

    # 基础权重（保留模板风格）
    # Order: PDE, bottom load, groove fix, groove side free, inner/outer side
    # free, top free, radial side free. The latest run showed Top/Rad free
    # traction losses staying high, so the free-surface weights are raised
    # while keeping the PDE term explicit through w_eq_pde/w_const_pde.
    base_w = torch.tensor([1.0, 240.0, 120.0, 20.0, 35.0, 50.0, 40.0], dtype=DTYPE, device=device)

    # Adam 阶段
    print("Starting Adam training...")
    print(f"[INFO] Adam epochs = {adam_epochs}, milestones = {adam_milestones}")
    print(
        f"[INFO] PDE loss = {w_eq_pde:g} * Eq + "
        f"Const weight ramp {w_const_pde_start:g}->{w_const_pde_end:g} "
        f"(epoch {const_ramp_start}->{const_ramp_end})"
    )
    print(f"[INFO] bottom load mean penalty weight = {load_mean_weight:g}")
    print(f"[INFO] adaptive weights enabled = {use_adaptive_weights}")
    print_sampling_info()

    optimizer_adam = optim.Adam(
        model.parameters(),
        lr=0.001,
        foreach=False,
        fused=False,
        capturable=(device.type == "cuda")
    )

    scheduler = MultiStepLR(optimizer_adam, milestones=adam_milestones, gamma=0.5)
    weighter = AdaptiveWeighter(num_losses=7, alpha=0.9)

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
        loss_list = [l_pde, l_load, l_fix, l_gside, l_side, l_top, l_rad]

        if use_adaptive_weights and epoch % 100 == 0 and epoch > 0:
            weighter.update(model, loss_list)
            if epoch % 500 == 0:
                w = weighter.weights.detach().cpu().numpy()
                print(
                    f"Ep {epoch} | "
                    f"PDE:{w[0]:.2f} | Load:{w[1]:.2f} | Fix:{w[2]:.2f} | "
                    f"GrooveSide:{w[3]:.2f} | Side:{w[4]:.2f} | Top:{w[5]:.2f} | Rad:{w[6]:.2f}"
                )

        adapt_w = weighter.weights if use_adaptive_weights else torch.ones_like(weighter.weights)
        loss = (
            base_w[0] * adapt_w[0] * l_pde +
            base_w[1] * adapt_w[1] * l_load +
            base_w[2] * adapt_w[2] * l_fix +
            base_w[3] * adapt_w[3] * l_gside +
            base_w[4] * adapt_w[4] * l_side +
            base_w[5] * adapt_w[5] * l_top +
            base_w[6] * adapt_w[6] * l_rad
        )

        if epoch % 50 == 0:
            elapsed = time() - t0
            weighted_pde = base_w[0] * adapt_w[0] * l_pde
            weighted_bc = (
                base_w[1] * adapt_w[1] * l_load +
                base_w[2] * adapt_w[2] * l_fix +
                base_w[3] * adapt_w[3] * l_gside +
                base_w[4] * adapt_w[4] * l_side +
                base_w[5] * adapt_w[5] * l_top +
                base_w[6] * adapt_w[6] * l_rad
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
                f"time:{elapsed:.1f}s"
            )

        loss.backward()
        optimizer_adam.step()
        scheduler.step()

    # L-BFGS 阶段
    adam_state = copy.deepcopy(model.state_dict())
    validation_samples = get_validation_samples()
    lbfgs_const_weight = w_const_pde_end
    val_losses_before = compute_losses(
        model,
        *validation_samples,
        const_weight=lbfgs_const_weight
    )
    val_loss_before = total_weighted_loss_from_losses(val_losses_before, base_w, adapt_w).detach()

    print("\nStarting L-BFGS...")
    print(f"[L-BFGS] validation loss before = {val_loss_before.item():.4e}")
    final_w = weighter.weights.detach() if use_adaptive_weights else torch.ones_like(weighter.weights)

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
            base_w[0] * final_w[0] * l_pde +
            base_w[1] * final_w[1] * l_load +
            base_w[2] * final_w[2] * l_fix +
            base_w[3] * final_w[3] * l_gside +
            base_w[4] * final_w[4] * l_side +
            base_w[5] * final_w[5] * l_top +
            base_w[6] * final_w[6] * l_rad
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
    val_loss_after = total_weighted_loss_from_losses(val_losses_after, base_w, final_w).detach()
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
