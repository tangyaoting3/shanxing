import numpy as np
import torch
from pyDOE2 import lhs


device = None
DTYPE = None
R_inner = None
R_outer = None
H_total = None
theta_min = None
theta_max = None
x_groove = None
R_groove = None
y_groove_bottom = None
y_groove_top = None


def configure_sampling(
    *,
    device_in,
    dtype_in,
    R_inner_in,
    R_outer_in,
    H_total_in,
    theta_min_in,
    theta_max_in,
    x_groove_in,
    R_groove_in,
    y_groove_bottom_in,
    y_groove_top_in,
):
    global device, DTYPE
    global R_inner, R_outer, H_total, theta_min, theta_max
    global x_groove, R_groove, y_groove_bottom, y_groove_top

    device = device_in
    DTYPE = dtype_in
    R_inner = R_inner_in
    R_outer = R_outer_in
    H_total = H_total_in
    theta_min = theta_min_in
    theta_max = theta_max_in
    x_groove = x_groove_in
    R_groove = R_groove_in
    y_groove_bottom = y_groove_bottom_in
    y_groove_top = y_groove_top_in


def _ensure_configured():
    if device is None or DTYPE is None:
        raise RuntimeError("采样模块尚未配置，请先调用 configure_sampling(...)。")


def lhs_tensor(dim, n):
    _ensure_configured()
    n = int(n)
    if n <= 0:
        return torch.empty((0, dim), dtype=DTYPE, device=device)
    return torch.tensor(lhs(dim, n), dtype=DTYPE, device=device)


def to_cartesian(r, theta, y):
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    return torch.cat([x, y, z], dim=1)


def groove_rho_torch(x, z):
    return torch.sqrt((x - x_groove) ** 2 + z ** 2 + 1e-30)


def is_in_groove_void_torch(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]
    rho = groove_rho_torch(x, z)
    cond_r = rho <= R_groove
    cond_y = (y >= y_groove_bottom) & (y <= y_groove_top)
    return cond_r & cond_y


def is_in_main_sector_torch(X):
    eps = 1e-10
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]

    r = torch.sqrt(x**2 + z**2 + 1e-30)
    theta = torch.atan2(z, x)

    cond_r = (r >= R_inner - eps) & (r <= R_outer + eps)
    cond_t = (theta >= theta_min - eps) & (theta <= theta_max + eps)
    cond_y = (y >= 0.0 - eps) & (y <= H_total + eps)
    return cond_r & cond_t & cond_y


def is_in_solid_torch(X):
    return is_in_main_sector_torch(X) & (~is_in_groove_void_torch(X))


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
    X = sample_groove_bottom(N_fix)
    return X.requires_grad_(True)


def sample_groove_side_free(Ns):
    X = sample_groove_side(Ns)
    return X.requires_grad_(True)


def sample_side_free(N_side):
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
    pts_list = []
    total = 0
    while total < N_top:
        m = max(2 * (N_top - total), 512)
        raw = lhs_tensor(2, m)
        r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        theta = theta_min + raw[:, 1:2] * (theta_max - theta_min)
        y = torch.full((m, 1), H_total, dtype=DTYPE, device=device)

        X = to_cartesian(r, theta, y)
        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X = torch.cat(pts_list, dim=0)[:N_top]
    return X.requires_grad_(True)


def sample_radial_free(N_rad):
    N_rad = int(N_rad)
    if N_rad <= 0:
        return torch.empty((0, 3), dtype=DTYPE, device=device, requires_grad=True)

    N_each = N_rad // 2
    N_rem = N_rad - N_each

    raw1 = lhs_tensor(2, N_each)
    if N_each > 0:
        r1 = torch.sqrt(raw1[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        y1 = raw1[:, 1:2] * H_total
        th1 = torch.full((N_each, 1), theta_min, dtype=DTYPE, device=device)
        X1 = to_cartesian(r1, th1, y1)
    else:
        X1 = torch.empty((0, 3), dtype=DTYPE, device=device)

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
