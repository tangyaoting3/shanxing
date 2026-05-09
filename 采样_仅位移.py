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
        raise RuntimeError("Sampling module is not configured. Call configure_sampling(...) first.")


def _empty_points():
    return torch.empty((0, 3), dtype=DTYPE, device=device, requires_grad=True)


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


def mirror_points_z(X):
    X_m = X.clone()
    X_m[:, 2:3] = -X_m[:, 2:3]
    return torch.cat([X, X_m], dim=0)


def groove_rho_torch(x, z):
    return torch.sqrt((x - x_groove) ** 2 + z ** 2 + 1.0e-30)


def is_in_main_sector_torch(X):
    eps = 1.0e-10
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]

    r = torch.sqrt(x**2 + z**2 + 1.0e-30)
    theta = torch.atan2(z, x)

    cond_r = (r >= R_inner - eps) & (r <= R_outer + eps)
    cond_t = (theta >= theta_min - eps) & (theta <= theta_max + eps)
    cond_y = (y >= 0.0 - eps) & (y <= H_total + eps)
    return cond_r & cond_t & cond_y


def is_in_solid_torch(X):
    return is_in_main_sector_torch(X)


def is_on_top_free_patch_torch(X):
    del X
    return torch.zeros((0, 1), dtype=torch.bool, device=device)


def sample_interior_points(N_col):
    pts_list = []
    total = 0
    theta_half_span = max(abs(float(theta_min)), abs(float(theta_max)))
    while total < N_col:
        m = max((N_col - total + 1) // 2, 512)
        raw = lhs_tensor(3, m)

        r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        theta = raw[:, 1:2] * theta_half_span
        y = raw[:, 2:3] * H_total

        X = mirror_points_z(to_cartesian(r, theta, y))
        mask = is_in_solid_torch(X).squeeze(1)
        X_ok = X[mask]

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X_col = torch.cat(pts_list, dim=0)[:N_col]
    return X_col.requires_grad_(True)


def sample_groove_edge_interior_points(N_edge):
    del N_edge
    return _empty_points()


def sample_groove_corner_interior_points(N_corner):
    del N_corner
    return _empty_points()


def sample_groove_influence_interior_points(N_inf):
    del N_inf
    return _empty_points()


def sample_bottom_loaded(N_b):
    m = max((N_b + 1) // 2, 1)
    raw = lhs_tensor(2, m)
    r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
    theta = raw[:, 1:2] * max(abs(float(theta_min)), abs(float(theta_max)))
    y = torch.zeros((m, 1), dtype=DTYPE, device=device)
    X = mirror_points_z(to_cartesian(r, theta, y))[:N_b]
    return X.requires_grad_(True)


def sample_groove_bottom(Nb):
    pts_list = []
    total = 0
    while total < Nb:
        m = max((Nb - total + 1) // 2, 128)
        raw = lhs_tensor(2, m)
        r = torch.sqrt(raw[:, 0:1] * (R_outer**2 - R_inner**2) + R_inner**2)
        theta = raw[:, 1:2] * max(abs(float(theta_min)), abs(float(theta_max)))
        y = torch.full((m, 1), H_total, dtype=DTYPE, device=device)
        X_ok = mirror_points_z(to_cartesian(r, theta, y))

        pts_list.append(X_ok)
        total += X_ok.shape[0]

    X = torch.cat(pts_list, dim=0)[:Nb]
    return X


def sample_groove_side(Ns):
    del Ns
    return torch.empty((0, 3), dtype=DTYPE, device=device)


def sample_groove_bottom_fixed(N_fix):
    X = sample_groove_bottom(N_fix)
    return X.requires_grad_(True)


def sample_groove_side_free(Ns):
    del Ns
    return _empty_points()


def sample_side_free(N_side):
    N_each = N_side // 2

    raw_out = lhs_tensor(2, max((N_each + 1) // 2, 1))
    theta_out = raw_out[:, 0:1] * max(abs(float(theta_min)), abs(float(theta_max)))
    y_out = raw_out[:, 1:2] * H_total
    r_out = torch.full((raw_out.shape[0], 1), R_outer, dtype=DTYPE, device=device)
    X_outer = mirror_points_z(to_cartesian(r_out, theta_out, y_out))[:N_each]

    n_inner = N_side - N_each
    raw_in = lhs_tensor(2, max((n_inner + 1) // 2, 1))
    theta_in = raw_in[:, 0:1] * max(abs(float(theta_min)), abs(float(theta_max)))
    y_in = raw_in[:, 1:2] * H_total
    r_in = torch.full((raw_in.shape[0], 1), R_inner, dtype=DTYPE, device=device)
    X_inner = mirror_points_z(to_cartesian(r_in, theta_in, y_in))[:n_inner]

    X = torch.cat([X_outer, X_inner], dim=0)
    return X.requires_grad_(True)


def sample_top_free_uniform(N_top):
    del N_top
    return _empty_points()


def sample_top_groove_ring_free(N_ring):
    return sample_top_free_uniform(N_ring)


def sample_top_free(N_top_uniform, N_top_ring=0):
    X_parts = []
    if N_top_uniform > 0:
        X_parts.append(sample_top_free_uniform(N_top_uniform))
    if N_top_ring > 0:
        X_parts.append(sample_top_groove_ring_free(N_top_ring))

    if not X_parts:
        return _empty_points()

    X = torch.cat(X_parts, dim=0)
    return X.requires_grad_(True)


def sample_radial_free(N_rad):
    N_rad = int(N_rad)
    if N_rad <= 0:
        return _empty_points()

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
    N_col_uniform = 32000
    N_b = 4000
    N_fix = 4000
    N_groove_side = 0
    N_side = 4000
    N_rad = 4000
    N_top_uniform = 0
    N_top_ring = 0

    X_col = sample_interior_points(N_col_uniform)
    X_bottom_load = sample_bottom_loaded(N_b)
    X_groove_bottom_fix = sample_groove_bottom_fixed(N_fix)
    X_groove_side_free = sample_groove_side_free(N_groove_side)
    X_side_free = sample_side_free(N_side)
    X_top_free = sample_top_free(N_top_uniform, N_top_ring)
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
    print("       uniform=32000, groove-related interior samplers disabled")
    print("       total interior=32000")
    print("       top_fixed=4000 (entire top surface)")
    print("       groove_side_free=0")
    print("       top_free=0")
    print("[INFO] validation collocation samples:")
    print("       uniform=9800, groove-related interior samplers disabled")
    print("       total interior=9800")
    print("       top_fixed=1200 (entire top surface)")
    print("       groove_side_free=0")
    print("       top_free=0")


def get_validation_samples():
    N_col_uniform = 9800
    N_b = 1200
    N_fix = 1200
    N_groove_side = 0
    N_side = 1200
    N_rad = 1200
    N_top_uniform = 0
    N_top_ring = 0

    X_col = sample_interior_points(N_col_uniform)
    X_bottom_load = sample_bottom_loaded(N_b)
    X_groove_bottom_fix = sample_groove_bottom_fixed(N_fix)
    X_groove_side_free = sample_groove_side_free(N_groove_side)
    X_side_free = sample_side_free(N_side)
    X_top_free = sample_top_free(N_top_uniform, N_top_ring)
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
