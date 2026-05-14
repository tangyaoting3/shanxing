import copy
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from 采样_仅位移 import (
    configure_sampling,
    get_samples,
    get_validation_samples,
    print_sampling_info,
    set_sampling_profile,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


E_val = 2.0e11
v_val = 0.3

E = torch.tensor(E_val, dtype=DTYPE, device=device)
v = torch.tensor(v_val, dtype=DTYPE, device=device)
lmda = E * v / ((1.0 + v) * (1.0 - 2.0 * v))
mu = E / (2.0 * (1.0 + v))


R_inner = 0.187
R_outer = 0.392
H_total = 0.07

theta_half_deg = 19.8
theta_half = np.deg2rad(theta_half_deg)
theta_min = -theta_half
theta_max = theta_half

# 保留这组变量名，方便和现有采样/显示脚本兼容
x_groove = 0.2895
R_groove = 0.05
y_groove_bottom = H_total
y_groove_top = H_total

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


F_total = 76400.0
bottom_area = 0.5 * (theta_max - theta_min) * (R_outer**2 - R_inner**2)
q_load = F_total / bottom_area

print(f"[INFO] bottom_area = {bottom_area:.6e} m^2")
print(f"[INFO] q_load      = {q_load:.6e} Pa")


L_ref = R_outer - R_inner
SIGMA_REF = q_load
U_REF = q_load * L_ref / E_val
PDE_REF = SIGMA_REF / L_ref

print(f"[INFO] L_ref       = {L_ref:.6e} m")
print(f"[INFO] U_REF       = {U_REF:.6e} m")
print(f"[INFO] SIGMA_REF   = {SIGMA_REF:.6e} Pa")
print(f"[INFO] PDE_REF     = {PDE_REF:.6e} Pa/m")


resample_every = 50
stage1_adam_epochs = 2000
stage2_adam_epochs = 2000
plateau_patience_stage1 = 120
plateau_patience_stage2 = 80
plateau_factor = 0.5
plateau_min_lr = 1.0e-6
stage1_lr = 1.0e-3
stage2_lr = 5.0e-4
stage1_lbfgs_max_iter = 2000
stage2_lbfgs_max_iter = 1200

w_eq_pde = 10.0
load_mean_weight = 5.0
adf_power = 1.0
stage2_anchor_weight = 0.0

use_pretrained_stage1 = True
pretrained_stage1_path = "pinn_sector_disp_only_model_stage1_soft.pth"

stage1_base_w = [1.0, 240.0, 1000.0, 20.0, 35.0, 50.0, 40.0]
stage2_base_w = [1.0, 240.0, 0.0, 20.0, 35.0, 50.0, 40.0]


lb = torch.tensor([-R_outer, 0.0, -R_outer], dtype=DTYPE, device=device)
ub = torch.tensor([R_outer, H_total, R_outer], dtype=DTYPE, device=device)


def top_surface_phi(x_in):
    return torch.clamp((H_total - x_in[:, 1:2]) / H_total, min=0.0, max=1.0) ** adf_power


def project_to_top_surface(x_in):
    return torch.cat(
        (
            x_in[:, 0:1],
            torch.full_like(x_in[:, 1:2], H_total),
            x_in[:, 2:3],
        ),
        dim=1,
    )


class PINN3D(nn.Module):
    """
    仅输出位移 u, v, w。
    应力不再由独立 head 预测，而是完全由位移梯度通过线弹性本构反推。
    """

    def __init__(self, layers, bc_mode="hard_adf"):
        super().__init__()
        self.bc_mode = bc_mode
        self.activation = nn.SiLU()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for layer in self.linears[:-1]:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

        nn.init.xavier_normal_(self.linears[-1].weight.data)
        nn.init.zeros_(self.linears[-1].bias.data)

    def set_bc_mode(self, bc_mode):
        if bc_mode not in ("soft", "hard_adf"):
            raise ValueError(f"Unsupported bc_mode: {bc_mode}")
        self.bc_mode = bc_mode

    def forward(self, x_in):
        a = 2.0 * (x_in - lb) / (ub - lb) - 1.0
        for layer in self.linears[:-1]:
            a = self.activation(layer(a))
        out = self.linears[-1](a)
        u_raw = U_REF * out[:, 0:1]
        v_raw = U_REF * out[:, 1:2]
        w_raw = U_REF * out[:, 2:3]
        if self.bc_mode == "soft":
            return u_raw, v_raw, w_raw
        if self.bc_mode == "hard_adf":
            # phi = 0 on the top Dirichlet boundary and phi > 0 in the interior.
            phi_top = top_surface_phi(x_in)
            u = phi_top * u_raw
            v = phi_top * v_raw
            w = phi_top * w_raw
            return u, v, w
        raise RuntimeError(f"Unsupported bc_mode during forward: {self.bc_mode}")


class ZeroInitCorrectionNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.SiLU()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for layer in self.linears[:-1]:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

        # Start stage 2 from an exact copy of the stage-1 field by making the
        # correction network output identically zero at initialization.
        nn.init.zeros_(self.linears[-1].weight.data)
        nn.init.zeros_(self.linears[-1].bias.data)

    def forward(self, x_in):
        a = 2.0 * (x_in - lb) / (ub - lb) - 1.0
        for layer in self.linears[:-1]:
            a = self.activation(layer(a))
        out = self.linears[-1](a)
        u_corr = U_REF * out[:, 0:1]
        v_corr = U_REF * out[:, 1:2]
        w_corr = U_REF * out[:, 2:3]
        return u_corr, v_corr, w_corr


class HardCorrectionModel(nn.Module):
    def __init__(self, base_model, correction_net, reference_model=None):
        super().__init__()
        self.base_model = base_model
        self.correction_net = correction_net
        self.reference_model = reference_model
        self.bc_mode = "hard_adf"

        self.base_model.set_bc_mode("soft")
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad_(False)
        if self.reference_model is not None:
            self.reference_model.set_bc_mode("soft")
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

    def forward(self, x_in):
        # The stage-1 field is frozen in parameter space, but we still need its
        # spatial derivatives for equilibrium and traction losses in stage 2.
        u_base, v_base, w_base = self.base_model(x_in)
        x_top = project_to_top_surface(x_in)
        u_top, v_top, w_top = self.base_model(x_top)
        u_corr_raw, v_corr_raw, w_corr_raw = self.correction_net(x_in)
        phi_top = top_surface_phi(x_in)

        # Remove only the boundary trace of the soft solution, then add a
        # correction that is also masked by phi. This gives exact u=0 on the top
        # surface without shrinking the whole stage-1 field away from that surface.
        u = u_base - (1.0 - phi_top) * u_top + phi_top * u_corr_raw
        v = v_base - (1.0 - phi_top) * v_top + phi_top * v_corr_raw
        w = w_base - (1.0 - phi_top) * w_top + phi_top * w_corr_raw
        return u, v, w


def get_gradients(u, x, create_graph=True):
    return torch.autograd.grad(
        u,
        x,
        torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def stress_from_displacement(u, v, w, x, create_graph=True):
    gu = get_gradients(u, x, create_graph=create_graph)
    gv = get_gradients(v, x, create_graph=create_graph)
    gw = get_gradients(w, x, create_graph=create_graph)

    ux, uy, uz = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    vx, vy, vz = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    wx, wy, wz = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]

    div_u = ux + vy + wz

    sxx = lmda * div_u + 2.0 * mu * ux
    syy = lmda * div_u + 2.0 * mu * vy
    szz = lmda * div_u + 2.0 * mu * wz
    sxy = mu * (uy + vx)
    syz = mu * (vz + wy)
    sxz = mu * (uz + wx)

    return {
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "wx": wx,
        "wy": wy,
        "wz": wz,
        "sxx": sxx,
        "syy": syy,
        "szz": szz,
        "sxy": sxy,
        "syz": syz,
        "sxz": sxz,
    }


def evaluate_displacement_and_stress(model, x_in, create_graph=True):
    u, v, w = model(x_in)
    stress = stress_from_displacement(u, v, w, x_in, create_graph=create_graph)
    return (
        u,
        v,
        w,
        stress["sxx"],
        stress["syy"],
        stress["szz"],
        stress["sxy"],
        stress["syz"],
        stress["sxz"],
    )


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
    const_weight=None,
):
    del const_weight

    u, v, w, sxx, syy, szz, sxy, syz, sxz = evaluate_displacement_and_stress(
        model,
        X_col,
        create_graph=True,
    )

    gsxx = get_gradients(sxx, X_col, create_graph=True)
    gsyy = get_gradients(syy, X_col, create_graph=True)
    gszz = get_gradients(szz, X_col, create_graph=True)
    gsxy = get_gradients(sxy, X_col, create_graph=True)
    gsyz = get_gradients(syz, X_col, create_graph=True)
    gsxz = get_gradients(sxz, X_col, create_graph=True)

    res_x = gsxx[:, 0:1] + gsxy[:, 1:2] + gsxz[:, 2:3]
    res_y = gsxy[:, 0:1] + gsyy[:, 1:2] + gsyz[:, 2:3]
    res_z = gsxz[:, 0:1] + gsyz[:, 1:2] + gszz[:, 2:3]

    loss_eq = (
        torch.mean((res_x / PDE_REF) ** 2)
        + torch.mean((res_y / PDE_REF) ** 2)
        + torch.mean((res_z / PDE_REF) ** 2)
    )
    loss_const = torch.zeros((), dtype=DTYPE, device=device)
    loss_pde = w_eq_pde * loss_eq

    _, _, _, sxx_b, syy_b, szz_b, sxy_b, syz_b, sxz_b = evaluate_displacement_and_stress(
        model,
        X_bottom_load,
        create_graph=True,
    )
    nx_b = torch.zeros((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)
    ny_b = -torch.ones((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)
    nz_b = torch.zeros((X_bottom_load.shape[0], 1), dtype=DTYPE, device=device)

    tx_b, ty_b, tz_b = traction_from_stress(sxx_b, syy_b, szz_b, sxy_b, syz_b, sxz_b, nx_b, ny_b, nz_b)
    target_tx_b = torch.zeros_like(tx_b)
    target_ty_b = torch.full_like(ty_b, q_load)
    target_tz_b = torch.zeros_like(tz_b)

    loss_bottom_load = torch.mean(
        ((tx_b - target_tx_b) / SIGMA_REF) ** 2
        + ((ty_b - target_ty_b) / SIGMA_REF) ** 2
        + ((tz_b - target_tz_b) / SIGMA_REF) ** 2
    )
    mean_ty_ratio = torch.mean(ty_b) / q_load
    loss_bottom_load = loss_bottom_load + load_mean_weight * (mean_ty_ratio - 1.0) ** 2

    u_fix, v_fix, w_fix = model(X_groove_bottom_fix)
    loss_groove_bottom_fixed = torch.mean(
        (u_fix / U_REF) ** 2 + (v_fix / U_REF) ** 2 + (w_fix / U_REF) ** 2
    )

    if X_groove_side_free.shape[0] == 0:
        loss_groove_side_free = torch.zeros((), dtype=DTYPE, device=device)
    else:
        _, _, _, sxx_g, syy_g, szz_g, sxy_g, syz_g, sxz_g = evaluate_displacement_and_stress(
            model,
            X_groove_side_free,
            create_graph=True,
        )
        x_g = X_groove_side_free[:, 0:1]
        z_g = X_groove_side_free[:, 2:3]
        rho_g = torch.sqrt((x_g - x_groove) ** 2 + z_g ** 2 + 1e-30)
        nx_g = (x_g - x_groove) / rho_g
        ny_g = torch.zeros_like(nx_g)
        nz_g = z_g / rho_g
        tx_g, ty_g, tz_g = traction_from_stress(sxx_g, syy_g, szz_g, sxy_g, syz_g, sxz_g, nx_g, ny_g, nz_g)
        loss_groove_side_free = torch.mean(
            (tx_g / SIGMA_REF) ** 2 + (ty_g / SIGMA_REF) ** 2 + (tz_g / SIGMA_REF) ** 2
        )

    _, _, _, sxx_s, syy_s, szz_s, sxy_s, syz_s, sxz_s = evaluate_displacement_and_stress(
        model,
        X_side_free,
        create_graph=True,
    )
    x_s = X_side_free[:, 0:1]
    z_s = X_side_free[:, 2:3]
    r_s = torch.sqrt(x_s**2 + z_s**2 + 1e-30)
    nx_s = x_s / r_s
    ny_s = torch.zeros_like(nx_s)
    nz_s = z_s / r_s
    tx_s, ty_s, tz_s = traction_from_stress(sxx_s, syy_s, szz_s, sxy_s, syz_s, sxz_s, nx_s, ny_s, nz_s)
    loss_side_free = torch.mean(
        (tx_s / SIGMA_REF) ** 2 + (ty_s / SIGMA_REF) ** 2 + (tz_s / SIGMA_REF) ** 2
    )

    if X_top_free.shape[0] == 0:
        loss_top_free = torch.zeros((), dtype=DTYPE, device=device)
    else:
        _, _, _, sxx_t, syy_t, szz_t, sxy_t, syz_t, sxz_t = evaluate_displacement_and_stress(
            model,
            X_top_free,
            create_graph=True,
        )
        nx_t = torch.zeros((X_top_free.shape[0], 1), dtype=DTYPE, device=device)
        ny_t = torch.ones((X_top_free.shape[0], 1), dtype=DTYPE, device=device)
        nz_t = torch.zeros((X_top_free.shape[0], 1), dtype=DTYPE, device=device)
        tx_t, ty_t, tz_t = traction_from_stress(sxx_t, syy_t, szz_t, sxy_t, syz_t, sxz_t, nx_t, ny_t, nz_t)
        loss_top_free = torch.mean(
            (tx_t / SIGMA_REF) ** 2 + (ty_t / SIGMA_REF) ** 2 + (tz_t / SIGMA_REF) ** 2
        )

    _, _, _, sxx_r, syy_r, szz_r, sxy_r, syz_r, sxz_r = evaluate_displacement_and_stress(
        model,
        X_radial_free,
        create_graph=True,
    )
    x_r = X_radial_free[:, 0:1]
    z_r = X_radial_free[:, 2:3]
    theta_r = torch.atan2(z_r, x_r)
    nx_r = -torch.sin(theta_r)
    ny_r = torch.zeros_like(nx_r)
    nz_r = torch.cos(theta_r)
    tx_r, ty_r, tz_r = traction_from_stress(sxx_r, syy_r, szz_r, sxy_r, syz_r, sxz_r, nx_r, ny_r, nz_r)
    loss_radial_free = torch.mean(
        (tx_r / SIGMA_REF) ** 2 + (ty_r / SIGMA_REF) ** 2 + (tz_r / SIGMA_REF) ** 2
    )

    if isinstance(model, HardCorrectionModel) and model.reference_model is not None:
        with torch.no_grad():
            u_soft, v_soft, w_soft = model.reference_model(X_col)
        phi_top_col = top_surface_phi(X_col)
        anchor_mask = phi_top_col**2
        loss_anchor = torch.mean(
            anchor_mask * ((u - u_soft) / U_REF) ** 2
            + anchor_mask * ((v - v_soft) / U_REF) ** 2
            + anchor_mask * ((w - w_soft) / U_REF) ** 2
        )
    else:
        loss_anchor = torch.zeros((), dtype=DTYPE, device=device)

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
        loss_anchor,
        mean_ty_ratio.detach(),
    )


def total_weighted_loss_from_losses(losses, base_w):
    l_pde, _, _, l_load, l_fix, l_gside, l_side, l_top, l_rad, l_anchor, _ = losses
    return (
        base_w[0] * l_pde
        + base_w[1] * l_load
        + base_w[2] * l_fix
        + base_w[3] * l_gside
        + base_w[4] * l_side
        + base_w[5] * l_top
        + base_w[6] * l_rad
        + stage2_anchor_weight * l_anchor
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
    save_path="sampling_points_preview_disp_only.png",
):
    groups = [
        ("Interior", X_col, "#4C78A8", 2),
        ("Bottom load", X_bottom_load, "#F58518", 8),
        ("Top fixed", X_groove_bottom_fix, "#E45756", 10),
        ("Groove side free", X_groove_side_free, "#72B7B2", 7),
        ("Inner/outer side free", X_side_free, "#54A24B", 7),
        ("Top free", X_top_free, "#B279A2", 7),
        ("Radial free", X_radial_free, "#FF9DA6", 7),
    ]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for name, X, color, size in groups:
        if X.shape[0] == 0:
            continue
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
    ax.set_title("Sampling Points Before Training (Disp Only)")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((2.0 * R_outer, 2.0 * R_outer, H_total))
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] sampling point preview saved to: {save_path}")


def run_adam_stage(
    model,
    *,
    stage_name,
    base_w,
    adam_epochs,
    learning_rate,
    plateau_patience,
):
    (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
    ) = get_samples()

    optimizer_adam = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        foreach=False,
        fused=False,
        capturable=(device.type == "cuda"),
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

    print(f"\nStarting {stage_name} Adam training...")
    if hasattr(model, "bc_mode"):
        print(f"[INFO] {stage_name} bc_mode = {model.bc_mode}")
    else:
        print(f"[INFO] {stage_name} model = {model.__class__.__name__}")
    print(f"[INFO] {stage_name} Adam epochs = {adam_epochs}")
    print(
        f"[INFO] {stage_name} LR scheduler = ReduceLROnPlateau("
        f"factor={plateau_factor:g}, patience={plateau_patience}, min_lr={plateau_min_lr:g})"
    )
    print(f"[INFO] {stage_name} Adam lr = {learning_rate:.3e}")
    print(f"[INFO] PDE loss = {w_eq_pde:g} * Eq")
    print(f"[INFO] bottom load mean penalty weight = {load_mean_weight:g}")
    if getattr(model, "bc_mode", None) == "hard_adf" or isinstance(model, HardCorrectionModel):
        print(f"[INFO] top fixed Dirichlet BC = hard constraint via ADF^p, p={adf_power:g}")
    else:
        print("[INFO] top fixed Dirichlet BC = soft penalty")
    print(f"[INFO] {stage_name} loss weights = {base_w.detach().cpu().tolist()}")
    print_sampling_info()

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
                X_radial_free,
            ) = get_samples()

        optimizer_adam.zero_grad()

        l_pde, l_eq, l_const, l_load, l_fix, l_gside, l_side, l_top, l_rad, l_anchor, mean_ty_ratio = compute_losses(
            model,
            X_col,
            X_bottom_load,
            X_groove_bottom_fix,
            X_groove_side_free,
            X_side_free,
            X_top_free,
            X_radial_free,
        )

        loss = (
            base_w[0] * l_pde
            + base_w[1] * l_load
            + base_w[2] * l_fix
            + base_w[3] * l_gside
            + base_w[4] * l_side
            + base_w[5] * l_top
            + base_w[6] * l_rad
            + stage2_anchor_weight * l_anchor
        )

        if epoch % 50 == 0:
            elapsed = time() - t0
            weighted_pde = base_w[0] * l_pde
            weighted_bc = (
                base_w[1] * l_load
                + base_w[2] * l_fix
                + base_w[3] * l_gside
                + base_w[4] * l_side
                + base_w[5] * l_top
                + base_w[6] * l_rad
            )
            weighted_anchor = stage2_anchor_weight * l_anchor
            print(
                f"{stage_name} Ep {epoch:5d} | Total:{loss.item():.4e} | PDE:{l_pde.item():.4e} | "
                f"Eq:{l_eq.item():.4e} | Const:{l_const.item():.4e} | "
                f"Load:{l_load.item():.4e} | Fix:{l_fix.item():.4e} | "
                f"GrooveSide:{l_gside.item():.4e} | Side:{l_side.item():.4e} | "
                f"Top:{l_top.item():.4e} | Rad:{l_rad.item():.4e} | Anchor:{l_anchor.item():.4e} | "
                f"W_PDE:{weighted_pde.item():.4e} | W_BC:{weighted_bc.item():.4e} | "
                f"W_Anchor:{weighted_anchor.item():.4e} | "
                f"mean_ty/q:{mean_ty_ratio.item():.4e} | "
                f"LR:{optimizer_adam.param_groups[0]['lr']:.3e} | "
                f"time:{elapsed:.1f}s"
            )

        loss.backward()
        optimizer_adam.step()
        scheduler.step(loss.detach().item())


def run_lbfgs_stage(model, *, stage_name, base_w, max_iter):
    adam_state = copy.deepcopy(model.state_dict())
    validation_samples = get_validation_samples()
    val_losses_before = compute_losses(model, *validation_samples)
    val_loss_before = total_weighted_loss_from_losses(val_losses_before, base_w).detach()

    print(f"\nStarting {stage_name} L-BFGS...")
    print(f"[{stage_name} L-BFGS] validation loss before = {val_loss_before.item():.4e}")

    (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
    ) = get_samples()

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    iter_count = [0]

    def closure():
        optimizer_lbfgs.zero_grad()

        l_pde, _, _, l_load, l_fix, l_gside, l_side, l_top, l_rad, l_anchor, _ = compute_losses(
            model,
            X_col,
            X_bottom_load,
            X_groove_bottom_fix,
            X_groove_side_free,
            X_side_free,
            X_top_free,
            X_radial_free,
        )

        loss_local = (
            base_w[0] * l_pde
            + base_w[1] * l_load
            + base_w[2] * l_fix
            + base_w[3] * l_gside
            + base_w[4] * l_side
            + base_w[5] * l_top
            + base_w[6] * l_rad
            + stage2_anchor_weight * l_anchor
        )

        loss_local.backward()
        iter_count[0] += 1

        if iter_count[0] % 20 == 0:
            print(f"{stage_name} L-BFGS Iter: {iter_count[0]} | Loss: {loss_local.item():.4e}")

        return loss_local

    optimizer_lbfgs.step(closure)

    val_losses_after = compute_losses(model, *validation_samples)
    val_loss_after = total_weighted_loss_from_losses(val_losses_after, base_w).detach()
    print(f"[{stage_name} L-BFGS] validation loss after  = {val_loss_after.item():.4e}")

    if val_loss_after > 1.05 * val_loss_before:
        model.load_state_dict(adam_state)
        print(f"[{stage_name} L-BFGS] validation got worse; restored Adam-final weights.")
    else:
        print(f"[{stage_name} L-BFGS] validation improved or stayed stable; kept L-BFGS weights.")


if __name__ == "__main__":
    model = PINN3D([3, 192, 192, 192, 192, 192, 3], bc_mode="soft").to(device=device, dtype=DTYPE)

    (
        X_col,
        X_bottom_load,
        X_groove_bottom_fix,
        X_groove_side_free,
        X_side_free,
        X_top_free,
        X_radial_free,
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

    print("[INFO] stress strategy = displacement only, all stresses are reconstructed from displacement gradients")

    base_w_stage1 = torch.tensor(stage1_base_w, dtype=DTYPE, device=device)
    base_w_stage2 = torch.tensor(stage2_base_w, dtype=DTYPE, device=device)

    stage1_save_path = "pinn_sector_disp_only_model_stage1_soft.pth"
    if use_pretrained_stage1:
        if os.path.exists(pretrained_stage1_path):
            print(f"[INFO] loading existing stage-1 soft weights from: {pretrained_stage1_path}")
            stage1_checkpoint = torch.load(pretrained_stage1_path, map_location=device)
            model.load_state_dict(stage1_checkpoint)
            model.eval()
            print("[INFO] stage 1 training skipped; using pretrained soft-BC solution as the base field.")
        else:
            print(
                f"[WARN] pretrained stage-1 weights not found at: {pretrained_stage1_path}\n"
                "[WARN] falling back to training stage 1 from scratch."
            )
            run_adam_stage(
                model,
                stage_name="Stage1-Soft",
                base_w=base_w_stage1,
                adam_epochs=stage1_adam_epochs,
                learning_rate=stage1_lr,
                plateau_patience=plateau_patience_stage1,
            )
            run_lbfgs_stage(
                model,
                stage_name="Stage1-Soft",
                base_w=base_w_stage1,
                max_iter=stage1_lbfgs_max_iter,
            )
            torch.save(model.state_dict(), stage1_save_path)
            print(f"[INFO] stage 1 weights saved to: {stage1_save_path}")
    else:
        run_adam_stage(
            model,
            stage_name="Stage1-Soft",
            base_w=base_w_stage1,
            adam_epochs=stage1_adam_epochs,
            learning_rate=stage1_lr,
            plateau_patience=plateau_patience_stage1,
        )
        run_lbfgs_stage(
            model,
            stage_name="Stage1-Soft",
            base_w=base_w_stage1,
            max_iter=stage1_lbfgs_max_iter,
        )
        torch.save(model.state_dict(), stage1_save_path)
        print(f"[INFO] stage 1 weights saved to: {stage1_save_path}")

    base_ref = PINN3D([3, 192, 192, 192, 192, 192, 3], bc_mode="soft").to(device=device, dtype=DTYPE)
    base_ref.load_state_dict(copy.deepcopy(model.state_dict()))

    stage1_reference_model = PINN3D([3, 192, 192, 192, 192, 192, 3], bc_mode="soft").to(device=device, dtype=DTYPE)
    stage1_reference_model.load_state_dict(copy.deepcopy(model.state_dict()))

    correction_net = ZeroInitCorrectionNet([3, 192, 192, 192, 192, 192, 3]).to(device=device, dtype=DTYPE)
    stage2_model = HardCorrectionModel(base_ref, correction_net, reference_model=stage1_reference_model).to(
        device=device,
        dtype=DTYPE,
    )
    set_sampling_profile("stage2")
    print("\n[INFO] Switching to stage 2 hard/exact Dirichlet enforcement with stage-1 soft solution as the base field.")

    run_adam_stage(
        stage2_model,
        stage_name="Stage2-HardADF",
        base_w=base_w_stage2,
        adam_epochs=stage2_adam_epochs,
        learning_rate=stage2_lr,
        plateau_patience=plateau_patience_stage2,
    )
    run_lbfgs_stage(
        stage2_model,
        stage_name="Stage2-HardADF",
        base_w=base_w_stage2,
        max_iter=stage2_lbfgs_max_iter,
    )

    print("Training Finished.")

    final_save_path = "pinn_sector_disp_only_model.pth"
    torch.save(
        {
            "format": "stage2_hard_correction_v1",
            "base_model_state_dict": model.state_dict(),
            "correction_net_state_dict": correction_net.state_dict(),
        },
        final_save_path,
    )
    print(f"模型权重已成功保存至: {final_save_path}")
