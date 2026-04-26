import time
import math
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# 1. 参数定义模块
# ============================================================

# ---------- 运行设备与数值精度 ----------
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# ---------- 随机种子 ----------
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- 几何参数（严格按题述） ----------
R_outer = 0.392                 # m
R_inner = 0.187                 # m
theta_half_deg = 19.8           # deg
theta_min = -np.deg2rad(theta_half_deg)
theta_max =  np.deg2rad(theta_half_deg)
H_total = 0.07                  # m

# ---------- 凹槽参数（严格按题述） ----------
x_groove = 0.2895               # m
y_groove_top = 0.07             # m
y_groove_bottom = 0.05          # m
R_groove = 0.05                 # m

# ---------- 材料参数（严格按题述） ----------
E = 2.0e9                       # Pa
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# ---------- 载荷参数（严格按题述） ----------
F_total = 76400.0               # N

bottom_area = 0.5 * (theta_max - theta_min) * (R_outer**2 - R_inner**2)
q_load = F_total / bottom_area   # 均布面力大小，沿 +Y 方向
print(f"[信息] 底面受力面积 = {bottom_area:.6e} m^2")
print(f"[信息] 均布面力 q = {q_load:.6e} Pa (沿 +Y 方向)")

# ---------- 数值尺度（仅用于损失缩放与网络参数化，不改变物理单位） ----------
# 说明：所有输入输出仍然使用 SI 单位，以下参考尺度只用于 PINN 训练稳定化。
L_ref = H_total
T_ref = abs(q_load)                           # traction reference
U_ref = T_ref * L_ref / E                     # displacement reference ~ qL/E
groove_depth = y_groove_top - y_groove_bottom
main_volume = bottom_area * H_total
groove_volume = np.pi * R_groove**2 * groove_depth
solid_volume = main_volume - groove_volume
Energy_ref = bottom_area * T_ref * U_ref     # 能量参考尺度 ~ q * A * u

# ---------- 训练参数 ----------
N_int = 4000                  # 内部点数
N_edge = 2400                 # 凹槽底缘邻域内部点数
N_energy = 4000               # 能量积分用的均匀内部点数
N_dir = 2400                  # 凹槽底面固支边界点数
N_dir_edge = 1200             # 凹槽底面近边缘固支加密点数
N_neu = 2400                  # 底面受力边界点数
N_free = 4000                 # 其余自由表面零牵引边界点数（含凹槽侧壁）
N_free_edge = 1200            # 凹槽侧壁近底缘自由面加密点数

# ---------- 凹槽底缘局部加密采样参数 ----------
edge_refine_ratio_int = 0.30
edge_refine_ratio_dir = 0.25
edge_refine_ratio_free = 0.25

edge_refine_radius = 0.008    # m，凹槽底缘邻域径向加密范围
edge_refine_height = 0.006    # m，凹槽底缘邻域竖向加密范围
edge_surface_band = 0.003     # m，边界近底缘带宽

epochs = 12000
resample_every = 20
print_every = 100

lr = 1.0e-4
eta_min = 1.0e-6

w_pde = 1.0
w_pde_edge = 3.0
w_energy = 1.0
w_dir = 800.0
w_dir_edge = 2000.0
w_neu = 20.0
w_free = 20.0
w_free_edge = 80.0
w_const = 10.0

grad_clip = 1.0

# ---------- L-BFGS 精修参数 ----------
use_lbfgs_finetune = True
lbfgs_lr = 0.5
lbfgs_max_iter = 300
lbfgs_history_size = 50
lbfgs_print_every = 20

# ---------- SIREN 网络参数 ----------
in_features = 3
out_features = 9
hidden_features = 128
hidden_layers = 5
first_omega_0 = 30.0
hidden_omega_0 = 30.0

# ---------- 模型保存参数 ----------
model_save_name = "pinn_sector_siren_mixed_model.pth"

# ============================================================
# 2. 几何与采样模块
# ============================================================

def angle_np(x, z):
    """计算极角 theta = atan2(z, x)，单位为弧度。"""
    return np.arctan2(z, x)

def in_main_body_np(x, y, z):
    """
    判断点是否落在主体扇环柱体内部（未扣除凹槽）。
    """
    r = np.sqrt(x**2 + z**2)
    theta = angle_np(x, z)
    cond_r = (r >= R_inner) & (r <= R_outer)
    cond_t = (theta >= theta_min) & (theta <= theta_max)
    cond_y = (y >= 0.0) & (y <= H_total)
    return cond_r & cond_t & cond_y

def in_groove_void_np(x, y, z):
    """
    判断点是否落在凹槽空腔内部（被挖去的体积）。
    凹槽定义：sqrt((x-x_groove)^2 + z^2) <= R_groove 且 y in [0.05, 0.07]
    """
    rho_g = np.sqrt((x - x_groove)**2 + z**2)
    cond_rho = rho_g <= R_groove
    cond_y = (y >= y_groove_bottom) & (y <= y_groove_top)
    return cond_rho & cond_y

def in_solid_np(x, y, z):
    """
    判断点是否落在最终计算域（实体域）内。
    """
    return in_main_body_np(x, y, z) & (~in_groove_void_np(x, y, z))

def sample_sector_annulus(n, y_value):
    """
    在扇环面上均匀采样点（面积均匀），y 固定。
    适用于底面 y=0 的 Neumann 边界采样。
    """
    theta = np.random.uniform(theta_min, theta_max, size=(n,))
    r = np.sqrt(np.random.uniform(R_inner**2, R_outer**2, size=(n,)))
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    y = np.full_like(x, fill_value=y_value)
    pts = np.stack([x, y, z], axis=1)
    return pts

def sample_interior(n):
    """
    在实体内部均匀采样，完全避开凹槽区域。
    采样方式：先在主体扇环柱体中随机采样，再剔除凹槽空腔点。
    """
    pts_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 512)
        theta = np.random.uniform(theta_min, theta_max, size=(m,))
        r = np.sqrt(np.random.uniform(R_inner**2, R_outer**2, size=(m,)))
        y = np.random.uniform(0.0, H_total, size=(m,))
        x = r * np.cos(theta)
        z = r * np.sin(theta)

        mask = in_solid_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_groove_bottom(n):
    """
    采样凹槽底面（Dirichlet 边界的一部分）：
    y = 0.05，且以 (x_groove, 0, 0) 为圆心的圆盘。
    """
    pts_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 256)
        alpha = np.random.uniform(0.0, 2.0 * np.pi, size=(m,))
        rho = np.sqrt(np.random.uniform(0.0, R_groove**2, size=(m,)))
        x = x_groove + rho * np.cos(alpha)
        z = rho * np.sin(alpha)
        y = np.full_like(x, y_groove_bottom)

        # 凹槽底面必须仍位于主体扇环投影内
        mask = in_main_body_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_groove_side(n):
    """
    采样凹槽侧壁：
    sqrt((x-x_groove)^2 + z^2) = R_groove, y in [0.05, 0.07]
    """
    pts_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 256)
        alpha = np.random.uniform(0.0, 2.0 * np.pi, size=(m,))
        y = np.random.uniform(y_groove_bottom, y_groove_top, size=(m,))
        x = x_groove + R_groove * np.cos(alpha)
        z = R_groove * np.sin(alpha)

        mask = in_main_body_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_dirichlet_groove_bottom_boundary(n):
    """
    采样凹槽底面固定边界。
    当前设定中仅凹槽底面施加零位移约束，凹槽侧壁属于自由面。
    """
    return sample_groove_bottom(n)

def sample_groove_bottom_edge_band(n, band_width=edge_surface_band):
    """
    在凹槽底面靠近底缘圆环处加密采样。
    用于固定边界与后处理中的局部边缘评估。
    """
    band_width = min(max(band_width, 1.0e-6), R_groove)
    rho_min = max(R_groove - band_width, 0.0)

    pts_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 256)
        alpha = np.random.uniform(0.0, 2.0 * np.pi, size=(m,))
        rho = np.sqrt(np.random.uniform(rho_min**2, R_groove**2, size=(m,)))
        x = x_groove + rho * np.cos(alpha)
        z = rho * np.sin(alpha)
        y = np.full_like(x, y_groove_bottom)

        mask = in_main_body_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_groove_side_edge_band(n, height_span=edge_surface_band):
    """
    在凹槽侧壁靠近底缘处加密采样。
    用于自由边界与后处理中的局部边缘评估。
    """
    height_span = min(max(height_span, 1.0e-6), groove_depth)
    y_upper = min(y_groove_bottom + height_span, y_groove_top)

    pts_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 256)
        alpha = np.random.uniform(0.0, 2.0 * np.pi, size=(m,))
        y = np.random.uniform(y_groove_bottom, y_upper, size=(m,))
        x = x_groove + R_groove * np.cos(alpha)
        z = R_groove * np.sin(alpha)

        mask = in_main_body_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_free_groove_side_surface(n):
    """
    采样凹槽侧壁自由面。
    对实体域而言，凹槽侧壁的外法向指向凹槽空腔内部：
    n = (-(x-x_groove)/R_groove, 0, -z/R_groove)。
    """
    pts = sample_groove_side(n)
    rho = np.sqrt((pts[:, 0] - x_groove)**2 + pts[:, 2]**2)
    normals = np.stack([
        -(pts[:, 0] - x_groove) / rho,
        np.zeros_like(rho),
        -pts[:, 2] / rho
    ], axis=1)
    return pts, normals

def sample_free_groove_side_edge_surface(n, height_span=edge_surface_band):
    """
    采样凹槽侧壁靠近底缘处的自由面，并返回外法向。
    """
    pts = sample_groove_side_edge_band(n, height_span=height_span)
    rho = np.sqrt((pts[:, 0] - x_groove)**2 + pts[:, 2]**2)
    normals = np.stack([
        -(pts[:, 0] - x_groove) / rho,
        np.zeros_like(rho),
        -pts[:, 2] / rho
    ], axis=1)
    return pts, normals

def sample_interior_groove_edge_neighborhood(
    n,
    radial_span=edge_refine_radius,
    vertical_span=edge_refine_height
):
    """
    在凹槽底缘附近的实体内部加密采样。
    这里围绕交线 rho=R_groove, y=y_groove_bottom 构造局部邻域，
    再剔除空腔点，仅保留实体内部点。
    """
    radial_span = min(max(radial_span, 1.0e-6), 0.5 * R_groove)
    vertical_span = min(max(vertical_span, 1.0e-6), max(y_groove_bottom, H_total - y_groove_bottom))

    pts_list = []
    count = 0
    while count < n:
        m = max(4 * (n - count), 512)
        alpha = np.random.uniform(0.0, 2.0 * np.pi, size=(m,))

        dr = np.random.normal(loc=0.0, scale=0.35 * radial_span, size=(m,))
        dy = np.random.normal(loc=0.0, scale=0.35 * vertical_span, size=(m,))
        dr = np.clip(dr, -radial_span, radial_span)
        dy = np.clip(dy, -vertical_span, vertical_span)

        rho = R_groove + dr
        x = x_groove + rho * np.cos(alpha)
        z = rho * np.sin(alpha)
        y = y_groove_bottom + dy

        mask = in_solid_np(x, y, z)
        accepted = np.stack([x[mask], y[mask], z[mask]], axis=1)
        pts_list.append(accepted)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    return pts

def sample_free_top_surface(n):
    """
    采样自由上表面：
    y = H_total 的扇环顶面，扣除凹槽开口圆盘区域。
    外法向 n = (0, 1, 0)。
    """
    pts_list = []
    normals_list = []
    count = 0
    while count < n:
        m = max(2 * (n - count), 256)
        pts = sample_sector_annulus(m, y_value=H_total)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # 顶面中被凹槽开口占据的圆盘不是实体自由外表面，需剔除。
        rho_g = np.sqrt((x - x_groove)**2 + z**2)
        mask = rho_g > R_groove
        accepted = pts[mask]
        normals = np.zeros_like(accepted)
        normals[:, 1] = 1.0

        pts_list.append(accepted)
        normals_list.append(normals)
        count += accepted.shape[0]

    pts = np.concatenate(pts_list, axis=0)[:n]
    normals = np.concatenate(normals_list, axis=0)[:n]
    return pts, normals

def sample_free_outer_cylindrical_surface(n):
    """
    采样外圆柱自由面：
    r = R_outer, theta in [theta_min, theta_max], y in [0, H_total]。
    外法向 n = (cos(theta), 0, sin(theta))。
    """
    theta = np.random.uniform(theta_min, theta_max, size=(n,))
    y = np.random.uniform(0.0, H_total, size=(n,))
    x = R_outer * np.cos(theta)
    z = R_outer * np.sin(theta)

    pts = np.stack([x, y, z], axis=1)
    normals = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=1)
    return pts, normals

def sample_free_inner_cylindrical_surface(n):
    """
    采样内圆柱自由面：
    r = R_inner, theta in [theta_min, theta_max], y in [0, H_total]。
    对扇环实体而言，内圆柱面的外法向指向半径减小方向：
    n = (-cos(theta), 0, -sin(theta))。
    """
    theta = np.random.uniform(theta_min, theta_max, size=(n,))
    y = np.random.uniform(0.0, H_total, size=(n,))
    x = R_inner * np.cos(theta)
    z = R_inner * np.sin(theta)

    pts = np.stack([x, y, z], axis=1)
    normals = np.stack([-np.cos(theta), np.zeros_like(theta), -np.sin(theta)], axis=1)
    return pts, normals

def sample_free_radial_side_surface(n, theta_value):
    """
    采样两侧径向自由面：
    theta = theta_min 或 theta_max, r in [R_inner, R_outer], y in [0, H_total]。
    theta = theta_max 时外法向为 e_theta = (-sin(theta), 0, cos(theta))；
    theta = theta_min 时外法向为 -e_theta = (sin(theta), 0, -cos(theta))。
    """
    r = np.random.uniform(R_inner, R_outer, size=(n,))
    y = np.random.uniform(0.0, H_total, size=(n,))
    theta = np.full_like(r, fill_value=theta_value)
    x = r * np.cos(theta)
    z = r * np.sin(theta)

    pts = np.stack([x, y, z], axis=1)
    sign = 1.0 if theta_value > 0.0 else -1.0
    normals = sign * np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    return pts, normals

def sample_free_surface_boundary(n):
    """
    采样除底面受力边界与凹槽底面固支边界之外的其余自由外表面，并返回对应外法向。
    自由面包括：
    - 顶面 y=H_total，扣除凹槽开口；
    - 凹槽侧壁；
    - 外圆柱面 r=R_outer；
    - 内圆柱面 r=R_inner；
    - 两个径向侧面 theta=theta_min/theta_max。
    """
    groove_opening_area = np.pi * R_groove**2
    groove_side_area = 2.0 * np.pi * R_groove * groove_depth
    top_area = bottom_area - groove_opening_area
    outer_area = R_outer * (theta_max - theta_min) * H_total
    inner_area = R_inner * (theta_max - theta_min) * H_total
    radial_side_area = (R_outer - R_inner) * H_total

    areas = np.array([
        top_area,
        groove_side_area,
        outer_area,
        inner_area,
        radial_side_area,
        radial_side_area
    ], dtype=np.float64)
    areas = np.maximum(areas, 0.0)
    counts = np.floor(n * areas / np.sum(areas)).astype(int)
    counts[-1] += n - int(np.sum(counts))

    samplers = [
        lambda count: sample_free_top_surface(count),
        lambda count: sample_free_groove_side_surface(count),
        lambda count: sample_free_outer_cylindrical_surface(count),
        lambda count: sample_free_inner_cylindrical_surface(count),
        lambda count: sample_free_radial_side_surface(count, theta_min),
        lambda count: sample_free_radial_side_surface(count, theta_max),
    ]

    pts_parts = []
    normal_parts = []
    for count, sampler in zip(counts, samplers):
        if count <= 0:
            continue
        pts_i, normals_i = sampler(int(count))
        pts_parts.append(pts_i)
        normal_parts.append(normals_i)

    pts = np.concatenate(pts_parts, axis=0)
    normals = np.concatenate(normal_parts, axis=0)

    idx = np.random.permutation(pts.shape[0])
    return pts[idx], normals[idx]

def to_tensor(x_np):
    return torch.tensor(x_np, dtype=dtype, device=device)

def shuffle_points_np(pts):
    if pts.shape[0] <= 1:
        return pts
    idx = np.random.permutation(pts.shape[0])
    return pts[idx]

def shuffle_points_with_normals_np(pts, normals):
    if pts.shape[0] <= 1:
        return pts, normals
    idx = np.random.permutation(pts.shape[0])
    return pts[idx], normals[idx]

# ============================================================
# 3. SIREN 神经网络模块
# ============================================================

class SineLayer(nn.Module):
    """
    SIREN 的正弦激活层。
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # SIREN 首层初始化
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                # SIREN 隐层初始化
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.fill_(0.0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenNet(nn.Module):
    """
    输入：三维坐标 (x, y, z)，单位仍为 m
    输出：三个位移分量 + 六个应力分量，单位仍为 m 和 Pa
    说明：
    1) 前向内部仅做“数值归一化”，不改变物理定义；
    2) 位移输出乘 U_ref，应力输出乘 T_ref，以提高训练初期的数值稳定性。
    """
    def __init__(self,
                 in_features=3,
                 hidden_features=128,
                 hidden_layers=5,
                 out_features=3,
                 first_omega_0=30.0,
                 hidden_omega_0=30.0):
        super().__init__()

        layers = []
        layers.append(SineLayer(in_features, hidden_features,
                                is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)
            final_linear.bias.fill_(0.0)

        layers.append(final_linear)
        self.net = nn.Sequential(*layers)

    def normalize_coords(self, x):
        """
        将坐标缩放到近似 [-1, 1] 区间，减少 SIREN 的训练难度。
        """
        x_in = x.clone()
        x_in[:, 0:1] = x[:, 0:1] / R_outer
        x_in[:, 1:2] = 2.0 * x[:, 1:2] / H_total - 1.0
        x_in[:, 2:3] = x[:, 2:3] / R_outer
        return x_in

    def forward(self, x):
        x_in = self.normalize_coords(x)
        y_hat = self.net(x_in)
        disp_phys = U_ref * y_hat[:, 0:3]
        stress_phys = T_ref * y_hat[:, 3:9]
        return torch.cat([disp_phys, stress_phys], dim=1)

# ============================================================
# 4. PDE 残差与应力应变模块
# ============================================================

def gradients(y, x, create_graph=True):
    """
    计算 dy/dx，y 必须是 shape=(N,1)。
    """
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True
    )[0]

def kinematics_and_stress(points, model, create_graph=True):
    """
    给定空间点 points，返回：
    - 位移 u,v,w
    - 应变 eps_xx, eps_yy, eps_zz, eps_xy, eps_yz, eps_xz
    - 网络应力 sxx, syy, szz, sxy, syz, sxz
    - 本构应力 sxx_const, syy_const, szz_const, sxy_const, syz_const, sxz_const
    """
    points = points.clone().detach().requires_grad_(True)
    pred = model(points)

    u = pred[:, 0:1]
    v = pred[:, 1:2]
    w = pred[:, 2:3]

    sxx = pred[:, 3:4]
    syy = pred[:, 4:5]
    szz = pred[:, 5:6]
    sxy = pred[:, 6:7]
    syz = pred[:, 7:8]
    sxz = pred[:, 8:9]

    grad_u = gradients(u, points, create_graph=create_graph)
    grad_v = gradients(v, points, create_graph=create_graph)
    grad_w = gradients(w, points, create_graph=create_graph)

    du_dx, du_dy, du_dz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    dv_dx, dv_dy, dv_dz = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    dw_dx, dw_dy, dw_dz = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

    # 小变形应变张量
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_zz = dw_dz
    eps_xy = 0.5 * (du_dy + dv_dx)
    eps_yz = 0.5 * (dv_dz + dw_dy)
    eps_xz = 0.5 * (du_dz + dw_dx)

    trace_eps = eps_xx + eps_yy + eps_zz

    # 由位移梯度得到的本构应力，用于约束网络应力。
    sxx_const = lam * trace_eps + 2.0 * mu * eps_xx
    syy_const = lam * trace_eps + 2.0 * mu * eps_yy
    szz_const = lam * trace_eps + 2.0 * mu * eps_zz
    sxy_const = 2.0 * mu * eps_xy
    syz_const = 2.0 * mu * eps_yz
    sxz_const = 2.0 * mu * eps_xz

    return {
        "points": points,
        "u": u, "v": v, "w": w,
        "eps_xx": eps_xx, "eps_yy": eps_yy, "eps_zz": eps_zz,
        "eps_xy": eps_xy, "eps_yz": eps_yz, "eps_xz": eps_xz,
        "sxx": sxx, "syy": syy, "szz": szz,
        "sxy": sxy, "syz": syz, "sxz": sxz,
        "sxx_const": sxx_const, "syy_const": syy_const, "szz_const": szz_const,
        "sxy_const": sxy_const, "syz_const": syz_const, "sxz_const": sxz_const
    }

def pde_residual(points, model):
    """
    线弹性平衡方程：div(sigma) = 0
    返回三个方向的平衡残差：
    rx = dsxx/dx + dsxy/dy + dsxz/dz
    ry = dsxy/dx + dsyy/dy + dsyz/dz
    rz = dsxz/dx + dsyz/dy + dszz/dz
    """
    out = kinematics_and_stress(points, model, create_graph=True)
    pts = out["points"]

    sxx, syy, szz = out["sxx"], out["syy"], out["szz"]
    sxy, syz, sxz = out["sxy"], out["syz"], out["sxz"]

    grad_sxx = gradients(sxx, pts, create_graph=True)
    grad_syy = gradients(syy, pts, create_graph=True)
    grad_szz = gradients(szz, pts, create_graph=True)
    grad_sxy = gradients(sxy, pts, create_graph=True)
    grad_syz = gradients(syz, pts, create_graph=True)
    grad_sxz = gradients(sxz, pts, create_graph=True)

    rx = grad_sxx[:, 0:1] + grad_sxy[:, 1:2] + grad_sxz[:, 2:3]
    ry = grad_sxy[:, 0:1] + grad_syy[:, 1:2] + grad_syz[:, 2:3]
    rz = grad_sxz[:, 0:1] + grad_syz[:, 1:2] + grad_szz[:, 2:3]

    return rx, ry, rz

def energy_balance_loss(x_energy, x_neu, model):
    """
    能量形式损失：
    采用线弹性下的 Clapeyron 定理约束：
        ∫ 0.5 * sigma:epsilon dV = 0.5 * ∫ t·u dS

    - 内应变能密度：0.5 * sigma:epsilon
    - 外力功密度（底面）：t · u

    这里通过 Monte Carlo 积分近似体积分与面积分，
    并将两者之差构造成无量纲损失。
    说明：这里应使用“均匀体采样”的内部点，否则能量积分会因局部加密而偏置。
    """
    out_int = kinematics_and_stress(x_energy, model, create_graph=True)
    strain_energy_density = 0.5 * (
        out_int["sxx_const"] * out_int["eps_xx"] +
        out_int["syy_const"] * out_int["eps_yy"] +
        out_int["szz_const"] * out_int["eps_zz"] +
        2.0 * out_int["sxy_const"] * out_int["eps_xy"] +
        2.0 * out_int["syz_const"] * out_int["eps_yz"] +
        2.0 * out_int["sxz_const"] * out_int["eps_xz"]
    )
    internal_energy = solid_volume * torch.mean(strain_energy_density)

    disp_neu = model(x_neu)[:, 0:3]
    external_work_density = q_load * disp_neu[:, 1:2]
    external_work = bottom_area * torch.mean(external_work_density)

    energy_gap = internal_energy - 0.5 * external_work
    loss_energy = (energy_gap / Energy_ref) ** 2
    return loss_energy, internal_energy, external_work

def bottom_traction_residual(points, model):
    """
    底面 y=0 的 Neumann 边界条件。
    底面外法向量 n = (0, -1, 0)。
    规定外部面力沿 +Y 方向，大小为 q_load，因此目标牵引向量为：
        t = [0, q_load, 0]^T
    而 sigma·n = [-sxy, -syy, -syz]^T
    """
    out = kinematics_and_stress(points, model, create_graph=True)
    tx = -out["sxy"]
    ty = -out["syy"]
    tz = -out["syz"]

    t_target_x = torch.zeros_like(tx)
    t_target_y = torch.full_like(ty, q_load)
    t_target_z = torch.zeros_like(tz)

    rx = tx - t_target_x
    ry = ty - t_target_y
    rz = tz - t_target_z
    return rx, ry, rz

def surface_traction_residual(points, normals, model):
    """
    任意表面的牵引残差：
        t = sigma · n
    对自由表面，目标牵引为零，因此直接返回 tx, ty, tz。
    """
    out = kinematics_and_stress(points, model, create_graph=True)
    nx = normals[:, 0:1]
    ny = normals[:, 1:2]
    nz = normals[:, 2:3]

    tx = out["sxx"] * nx + out["sxy"] * ny + out["sxz"] * nz
    ty = out["sxy"] * nx + out["syy"] * ny + out["syz"] * nz
    tz = out["sxz"] * nx + out["syz"] * ny + out["szz"] * nz
    return tx, ty, tz

def von_mises_from_stress(sxx, syy, szz, sxy, syz, sxz):
    """
    3D von Mises 等效应力。
    """
    vm = torch.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
        + 1e-30
    )
    return vm

# ============================================================
# 5. 损失函数模块
# ============================================================

def compute_losses(
    model,
    x_int,
    x_edge,
    x_energy,
    x_dir,
    x_dir_edge,
    x_neu,
    x_free,
    n_free,
    x_free_edge,
    n_free_edge
):
    """
    总损失 = 控制方程损失 + 能量形式损失 + 各类边界损失
    说明：
    - PDE 残差缩放：T_ref / L_ref
    - 能量损失缩放：Energy_ref
    - Dirichlet 位移缩放：U_ref
    - Neumann 与自由面牵引缩放：T_ref
    这样做仅为改善训练数值条件，不改变物理单位。
    """
    # ---- PDE loss ----
    rx_pde, ry_pde, rz_pde = pde_residual(x_int, model)
    pde_scale = T_ref / L_ref
    loss_pde = torch.mean(
        (rx_pde / pde_scale)**2 +
        (ry_pde / pde_scale)**2 +
        (rz_pde / pde_scale)**2
    )

    # ---- Local PDE loss near groove bottom edge ----
    rx_edge, ry_edge, rz_edge = pde_residual(x_edge, model)
    loss_pde_edge = torch.mean(
        (rx_edge / pde_scale)**2 +
        (ry_edge / pde_scale)**2 +
        (rz_edge / pde_scale)**2
    )

    # ---- Energy loss ----
    loss_energy, internal_energy, external_work = energy_balance_loss(x_energy, x_neu, model)

    # ---- Constitutive consistency loss ----
    out_const = kinematics_and_stress(x_int, model, create_graph=True)
    loss_const = torch.mean(
        ((out_const["sxx"] - out_const["sxx_const"]) / T_ref)**2 +
        ((out_const["syy"] - out_const["syy_const"]) / T_ref)**2 +
        ((out_const["szz"] - out_const["szz_const"]) / T_ref)**2 +
        ((out_const["sxy"] - out_const["sxy_const"]) / T_ref)**2 +
        ((out_const["syz"] - out_const["syz_const"]) / T_ref)**2 +
        ((out_const["sxz"] - out_const["sxz_const"]) / T_ref)**2
    )

    # ---- Dirichlet loss（仅凹槽底面固支）----
    u_dir = model(x_dir)[:, 0:3]
    loss_dir = torch.mean((u_dir / U_ref)**2)

    # ---- Local Dirichlet loss near groove bottom edge ----
    u_dir_edge = model(x_dir_edge)[:, 0:3]
    loss_dir_edge = torch.mean((u_dir_edge / U_ref)**2)

    # ---- Neumann loss（底面向上均布面力）----
    rx_neu, ry_neu, rz_neu = bottom_traction_residual(x_neu, model)
    loss_neu = torch.mean(
        (rx_neu / T_ref)**2 +
        (ry_neu / T_ref)**2 +
        (rz_neu / T_ref)**2
    )

    # ---- Free surface loss（其余外表面零牵引）----
    rx_free, ry_free, rz_free = surface_traction_residual(x_free, n_free, model)
    loss_free = torch.mean(
        (rx_free / T_ref)**2 +
        (ry_free / T_ref)**2 +
        (rz_free / T_ref)**2
    )

    # ---- Local free surface loss near groove side/bottom edge ----
    rx_free_edge, ry_free_edge, rz_free_edge = surface_traction_residual(
        x_free_edge,
        n_free_edge,
        model
    )
    loss_free_edge = torch.mean(
        (rx_free_edge / T_ref)**2 +
        (ry_free_edge / T_ref)**2 +
        (rz_free_edge / T_ref)**2
    )

    loss_total = (
        w_pde * loss_pde
        + w_pde_edge * loss_pde_edge
        + w_energy * loss_energy
        + w_const * loss_const
        + w_dir * loss_dir
        + w_dir_edge * loss_dir_edge
        + w_neu * loss_neu
        + w_free * loss_free
        + w_free_edge * loss_free_edge
    )

    return loss_total, {
        "loss_pde": loss_pde.detach(),
        "loss_pde_edge": loss_pde_edge.detach(),
        "loss_energy": loss_energy.detach(),
        "loss_const": loss_const.detach(),
        "loss_dir": loss_dir.detach(),
        "loss_dir_edge": loss_dir_edge.detach(),
        "loss_neu": loss_neu.detach(),
        "loss_free": loss_free.detach(),
        "loss_free_edge": loss_free_edge.detach(),
        "internal_energy": internal_energy.detach(),
        "external_work": external_work.detach(),
        "loss_total": loss_total.detach()
    }

# ============================================================
# 6. 训练流程模块
# ============================================================

def resample_training_points():
    n_int_edge = min(int(round(N_int * edge_refine_ratio_int)), N_int)
    n_int_bulk = N_int - n_int_edge
    x_int_parts = []
    if n_int_bulk > 0:
        x_int_parts.append(sample_interior(n_int_bulk))
    if n_int_edge > 0:
        x_int_parts.append(sample_interior_groove_edge_neighborhood(n_int_edge))
    x_int = to_tensor(shuffle_points_np(np.concatenate(x_int_parts, axis=0)))

    x_edge = to_tensor(sample_interior_groove_edge_neighborhood(N_edge))
    x_energy = to_tensor(sample_interior(N_energy))

    n_dir_edge = min(int(round(N_dir * edge_refine_ratio_dir)), N_dir)
    n_dir_bulk = N_dir - n_dir_edge
    x_dir_parts = []
    if n_dir_bulk > 0:
        x_dir_parts.append(sample_dirichlet_groove_bottom_boundary(n_dir_bulk))
    if n_dir_edge > 0:
        x_dir_parts.append(sample_groove_bottom_edge_band(n_dir_edge))
    x_dir = to_tensor(shuffle_points_np(np.concatenate(x_dir_parts, axis=0)))

    x_dir_edge = to_tensor(sample_groove_bottom_edge_band(N_dir_edge))
    x_neu = to_tensor(sample_sector_annulus(N_neu, y_value=0.0))

    n_free_edge = min(int(round(N_free * edge_refine_ratio_free)), N_free)
    n_free_bulk = N_free - n_free_edge
    x_free_parts = []
    n_free_parts = []
    if n_free_bulk > 0:
        x_free_bulk_np, n_free_bulk_np = sample_free_surface_boundary(n_free_bulk)
        x_free_parts.append(x_free_bulk_np)
        n_free_parts.append(n_free_bulk_np)
    if n_free_edge > 0:
        x_free_edge_np, n_free_edge_np = sample_free_groove_side_edge_surface(n_free_edge)
        x_free_parts.append(x_free_edge_np)
        n_free_parts.append(n_free_edge_np)
    x_free_np = np.concatenate(x_free_parts, axis=0)
    n_free_np = np.concatenate(n_free_parts, axis=0)
    x_free_np, n_free_np = shuffle_points_with_normals_np(x_free_np, n_free_np)

    x_free = to_tensor(x_free_np)
    n_free = to_tensor(n_free_np)

    x_free_edge_np, n_free_edge_np = sample_free_groove_side_edge_surface(N_free_edge)
    x_free_edge = to_tensor(x_free_edge_np)
    n_free_edge = to_tensor(n_free_edge_np)

    return (
        x_int,
        x_edge,
        x_energy,
        x_dir,
        x_dir_edge,
        x_neu,
        x_free,
        n_free,
        x_free_edge,
        n_free_edge
    )

def evaluate_total_loss_scalar(model, samples):
    loss_total, _ = compute_losses(model, *samples)
    return float(loss_total.detach().cpu().item())

def train_model():
    model = SirenNet(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,fused=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=eta_min
    )

    history = {
        "loss_total": [],
        "loss_pde": [],
        "loss_pde_edge": [],
        "loss_energy": [],
        "loss_const": [],
        "loss_dir": [],
        "loss_dir_edge": [],
        "loss_neu": [],
        "loss_free": [],
        "loss_free_edge": []
    }

    best_state = None
    best_loss = float("inf")

    samples = resample_training_points()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if (epoch == 1) or (epoch % resample_every == 0):
            samples = resample_training_points()

        optimizer.zero_grad(set_to_none=True)

        loss_total, logs = compute_losses(model, *samples)
        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        ltot = float(logs["loss_total"].cpu().item())
        lpde = float(logs["loss_pde"].cpu().item())
        lpde_edge = float(logs["loss_pde_edge"].cpu().item())
        lenergy = float(logs["loss_energy"].cpu().item())
        lconst = float(logs["loss_const"].cpu().item())
        ldir = float(logs["loss_dir"].cpu().item())
        ldir_edge = float(logs["loss_dir_edge"].cpu().item())
        lneu = float(logs["loss_neu"].cpu().item())
        lfree = float(logs["loss_free"].cpu().item())
        lfree_edge = float(logs["loss_free_edge"].cpu().item())

        history["loss_total"].append(ltot)
        history["loss_pde"].append(lpde)
        history["loss_pde_edge"].append(lpde_edge)
        history["loss_energy"].append(lenergy)
        history["loss_const"].append(lconst)
        history["loss_dir"].append(ldir)
        history["loss_dir_edge"].append(ldir_edge)
        history["loss_neu"].append(lneu)
        history["loss_free"].append(lfree)
        history["loss_free_edge"].append(lfree_edge)

        if ltot < best_loss:
            best_loss = ltot
            best_state = copy.deepcopy(model.state_dict())

        if epoch % print_every == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:6d}/{epochs}] | "
                f"Total={ltot:.6e} | PDE={lpde:.6e} | PDEe={lpde_edge:.6e} | "
                f"ENE={lenergy:.6e} | CONST={lconst:.6e} | "
                f"DIR={ldir:.6e} | DIRe={ldir_edge:.6e} | "
                f"NEU={lneu:.6e} | FREE={lfree:.6e} | FREEe={lfree_edge:.6e} | "
                f"lr={current_lr:.3e} | time={elapsed:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    lbfgs_val_before = None
    lbfgs_val_after = None
    lbfgs_kept = False

    if use_lbfgs_finetune:
        print("\n[LBFGS] 开始固定样本精修...")

        pre_lbfgs_state = copy.deepcopy(model.state_dict())
        val_samples = resample_training_points()
        lbfgs_samples = resample_training_points()

        lbfgs_val_before = evaluate_total_loss_scalar(model, val_samples)
        print(f"[LBFGS] 精修前验证总损失 = {lbfgs_val_before:.6e}")

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe"
        )

        lbfgs_iter = [0]

        def closure():
            optimizer_lbfgs.zero_grad()

            loss_total_lbfgs, logs_lbfgs = compute_losses(model, *lbfgs_samples)
            loss_total_lbfgs.backward()

            lbfgs_iter[0] += 1
            if lbfgs_iter[0] == 1 or lbfgs_iter[0] % lbfgs_print_every == 0:
                print(
                    f"[LBFGS] Iter={lbfgs_iter[0]:4d} | "
                    f"Total={float(loss_total_lbfgs.detach().cpu().item()):.6e} | "
                    f"PDE={float(logs_lbfgs['loss_pde'].cpu().item()):.6e} | "
                    f"PDEe={float(logs_lbfgs['loss_pde_edge'].cpu().item()):.6e} | "
                    f"ENE={float(logs_lbfgs['loss_energy'].cpu().item()):.6e} | "
                    f"CONST={float(logs_lbfgs['loss_const'].cpu().item()):.6e} | "
                    f"DIR={float(logs_lbfgs['loss_dir'].cpu().item()):.6e} | "
                    f"DIRe={float(logs_lbfgs['loss_dir_edge'].cpu().item()):.6e} | "
                    f"NEU={float(logs_lbfgs['loss_neu'].cpu().item()):.6e} | "
                    f"FREE={float(logs_lbfgs['loss_free'].cpu().item()):.6e} | "
                    f"FREEe={float(logs_lbfgs['loss_free_edge'].cpu().item()):.6e}"
                )

            return loss_total_lbfgs

        optimizer_lbfgs.step(closure)

        lbfgs_val_after = evaluate_total_loss_scalar(model, val_samples)
        print(f"[LBFGS] 精修后验证总损失 = {lbfgs_val_after:.6e}")

        if lbfgs_val_after <= lbfgs_val_before:
            lbfgs_kept = True
            print("[LBFGS] 验证总损失改善，保留精修后的权重。")
        else:
            model.load_state_dict(pre_lbfgs_state)
            print("[LBFGS] 验证总损失未改善，回退到 Adam 最优权重。")

    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_save_name)
    torch.save(model.state_dict(), model_save_path)

    print(f"[训练结束] Adam阶段最优总损失 = {best_loss:.6e}")
    if use_lbfgs_finetune and lbfgs_val_before is not None and lbfgs_val_after is not None:
        print(
            f"[训练结束] LBFGS验证总损失: "
            f"{lbfgs_val_before:.6e} -> {lbfgs_val_after:.6e} "
            f"({'保留' if lbfgs_kept else '回退'})"
        )
    print(f"[模型保存] 最优模型权重已保存至: {model_save_path}")
    return model, history

# ============================================================
# 7. 后处理与可视化模块
# ============================================================

def evaluate_points_with_stress(model, points_np, batch_size=4096):
    """
    对任意点集评估：
    - 位移 u,v,w
    - 位移模长 |u|
    - von Mises 应力
    """
    model.eval()

    all_disp = []
    all_vm = []
    all_sigma = []

    n_total = points_np.shape[0]
    for i in range(0, n_total, batch_size):
        pts_batch_np = points_np[i:i + batch_size]
        pts = torch.tensor(pts_batch_np, dtype=dtype, device=device, requires_grad=True)

        out = kinematics_and_stress(pts, model, create_graph=False)
        u = torch.cat([out["u"], out["v"], out["w"]], dim=1)
        vm = von_mises_from_stress(out["sxx"], out["syy"], out["szz"],
                                   out["sxy"], out["syz"], out["sxz"])
        sigma = torch.cat([out["sxx"], out["syy"], out["szz"],
                           out["sxy"], out["syz"], out["sxz"]], dim=1)

        all_disp.append(u.detach().cpu().numpy())
        all_vm.append(vm.detach().cpu().numpy())
        all_sigma.append(sigma.detach().cpu().numpy())

    disp = np.vstack(all_disp)
    vm = np.vstack(all_vm)
    sigma = np.vstack(all_sigma)
    umag = np.linalg.norm(disp, axis=1, keepdims=True)
    return disp, umag, vm, sigma

def summarize_extrema_for_points(model, region_name, points_np, batch_size=4096):
    """
    对给定点集统计该区域内的最大位移与最大 von Mises 应力。
    """
    disp, umag, vm, sigma = evaluate_points_with_stress(model, points_np, batch_size=batch_size)

    idx_u = int(np.argmax(umag[:, 0]))
    idx_vm = int(np.argmax(vm[:, 0]))

    return {
        "region": region_name,
        "num_points": int(points_np.shape[0]),
        "u_max": float(umag[idx_u, 0]),
        "u_point": points_np[idx_u].copy(),
        "vm_max": float(vm[idx_vm, 0]),
        "vm_point": points_np[idx_vm].copy()
    }

def estimate_global_extrema(model, n_eval=50000):
    """
    通过“内部 + 各类边界 + 凹槽底缘邻域”的分区域采样，
    更公平地近似评估全域最大位移和最大 von Mises 应力。
    """
    n_surface_eval = max(8000, n_eval // 6)
    n_edge_eval = max(12000, n_eval // 4)

    x_free_eval, _ = sample_free_surface_boundary(n_surface_eval)

    region_specs = [
        ("内部随机采样", sample_interior(n_eval)),
        ("底面受力边界", sample_sector_annulus(n_surface_eval, y_value=0.0)),
        ("凹槽底面固支边界", sample_groove_bottom(n_surface_eval)),
        ("其余自由表面", x_free_eval),
        ("凹槽底缘邻域内部", sample_interior_groove_edge_neighborhood(n_edge_eval)),
        ("凹槽底面近边缘", sample_groove_bottom_edge_band(n_surface_eval)),
        ("凹槽侧壁近底缘", sample_groove_side_edge_band(n_surface_eval))
    ]

    summaries = [
        summarize_extrema_for_points(model, region_name, pts, batch_size=4096)
        for region_name, pts in region_specs
    ]

    best_u = max(summaries, key=lambda item: item["u_max"])
    best_vm = max(summaries, key=lambda item: item["vm_max"])

    print("\n================= 关键工程结果（分区域高密度采样近似） =================")
    print(f"[位移] 全局近似最大值来自：{best_u['region']}")
    print(f"最大位移模长 |u|max ≈ {best_u['u_max']:.6e} m")
    print(
        f"对应坐标 ≈ ({best_u['u_point'][0]:.6f}, "
        f"{best_u['u_point'][1]:.6f}, {best_u['u_point'][2]:.6f}) m"
    )
    print(f"[应力] 全局近似最大值来自：{best_vm['region']}")
    print(f"最大 von Mises 应力 σ_vm,max ≈ {best_vm['vm_max']:.6e} Pa")
    print(
        f"对应坐标 ≈ ({best_vm['vm_point'][0]:.6f}, "
        f"{best_vm['vm_point'][1]:.6f}, {best_vm['vm_point'][2]:.6f}) m"
    )
    print("---------------- 各区域峰值概览 ----------------")
    for item in summaries:
        print(
            f"{item['region']:<18} | 点数={item['num_points']:6d} | "
            f"|u|max={item['u_max']:.6e} m | "
            f"σ_vm,max={item['vm_max']:.6e} Pa"
        )
    print("=======================================================================\n")
    return summaries

def make_mask_xy_section(X, Y, z_const=0.0):
    """
    中心剖面 z=z_const 的实体域掩膜。
    """
    Z = np.full_like(X, fill_value=z_const)
    return in_solid_np(X, Y, Z)

def make_mask_xz_slice(X, Z, y_const):
    """
    水平切片 y=y_const 的实体域掩膜。
    """
    Y = np.full_like(X, fill_value=y_const)
    return in_solid_np(X, Y, Z)

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.semilogy(history["loss_total"], label="Total loss")
    plt.semilogy(history["loss_pde"], label="PDE loss")
    plt.semilogy(history["loss_pde_edge"], label="Edge PDE loss")
    plt.semilogy(history["loss_energy"], label="Energy loss")
    plt.semilogy(history["loss_const"], label="Constitutive loss")
    plt.semilogy(history["loss_dir"], label="Dirichlet loss")
    plt.semilogy(history["loss_dir_edge"], label="Edge Dirichlet loss")
    plt.semilogy(history["loss_neu"], label="Neumann loss")
    plt.semilogy(history["loss_free"], label="Free surface loss")
    plt.semilogy(history["loss_free_edge"], label="Edge free surface loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN Training Loss History")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_center_section(model, nx=250, ny=180):
    """
    绘制中心剖面 z=0 上的：
    1) 位移模长云图
    2) von Mises 应力云图
    """
    x_min, x_max = -0.02, R_outer + 0.02
    y_min, y_max = 0.0, H_total

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    mask = make_mask_xy_section(X, Y, z_const=0.0)
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=1)

    disp, umag, vm, sigma = evaluate_points_with_stress(model, pts, batch_size=4096)

    U_plot = np.full_like(X, np.nan, dtype=np.float64)
    VM_plot = np.full_like(X, np.nan, dtype=np.float64)
    U_plot[mask] = umag[:, 0]
    VM_plot[mask] = vm[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cf1 = axes[0].contourf(X, Y, U_plot, levels=60)
    axes[0].set_title(r"Center section $z=0$: displacement magnitude $|\mathbf{u}|$ (m)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    plt.colorbar(cf1, ax=axes[0])

    cf2 = axes[1].contourf(X, Y, VM_plot, levels=60)
    axes[1].set_title(r"Center section $z=0$: von Mises stress (Pa)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(cf2, ax=axes[1])

    plt.tight_layout()
    plt.show()

def plot_horizontal_slice(model, y_const=0.035, nx=260, nz=240):
    """
    绘制水平切片 y=y_const 的：
    1) 位移模长云图
    2) von Mises 应力云图
    """
    x_min = R_inner * np.cos(theta_max) - 0.03
    x_max = R_outer + 0.02
    z_lim = R_outer * np.sin(theta_max) + 0.03

    x = np.linspace(x_min, x_max, nx)
    z = np.linspace(-z_lim, z_lim, nz)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, fill_value=y_const)

    mask = make_mask_xz_slice(X, Z, y_const=y_const)
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=1)

    disp, umag, vm, sigma = evaluate_points_with_stress(model, pts, batch_size=4096)

    U_plot = np.full_like(X, np.nan, dtype=np.float64)
    VM_plot = np.full_like(X, np.nan, dtype=np.float64)
    U_plot[mask] = umag[:, 0]
    VM_plot[mask] = vm[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cf1 = axes[0].contourf(X, Z, U_plot, levels=60)
    axes[0].set_title(rf"Horizontal slice $y={y_const:.3f}$ m: $|\mathbf{{u}}|$ (m)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_aspect("equal")
    plt.colorbar(cf1, ax=axes[0])

    cf2 = axes[1].contourf(X, Z, VM_plot, levels=60)
    axes[1].set_title(rf"Horizontal slice $y={y_const:.3f}$ m: von Mises stress (Pa)")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(cf2, ax=axes[1])

    plt.tight_layout()
    plt.show()

# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print(f"[设备] {device}")
    print(f"[参考尺度] L_ref={L_ref:.3e} m, T_ref={T_ref:.3e} Pa, U_ref={U_ref:.3e} m")
    print("[说明] 本代码使用的边界：凹槽底面固支 + 底面向上受力 + 凹槽侧壁及其余外表面零牵引自由边界。")

    model, history = train_model()

    # 训练历史
    plot_training_history(history)

    # 输出关键工程指标（通过高密度随机采样近似）
    estimate_global_extrema(model, n_eval=50000)

    # 可视化：中心剖面 z=0
    plot_center_section(model, nx=260, ny=180)

    # 可视化：中部水平截面 y=0.035
    plot_horizontal_slice(model, y_const=0.035, nx=260, nz=240)

    # 可视化：凹槽深度范围内的水平截面 y=0.060
    plot_horizontal_slice(model, y_const=0.060, nx=260, nz=240)
