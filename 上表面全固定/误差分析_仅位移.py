import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from 扇形2_仅位移 import (
    PINN3D,
    DTYPE,
    device,
    H_total,
    R_inner,
    R_outer,
    lmda,
    mu,
    theta_max,
    theta_min,
    U_REF,
    adf_power,
    project_to_top_surface,
    top_blend_fraction,
    top_surface_phi,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")
RESULT_STAGE1_DIR = os.path.join(SCRIPT_DIR, "result_第一阶段")
RESULT_STAGE2_START_DIR = os.path.join(SCRIPT_DIR, "result_实验A")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")

MODEL_PATH = os.path.join(WEIGHTS_DIR, "pinn_sector_disp_only_model_stage1_soft (1).pth")
FEM_UTOTAL_PATH = os.path.join(SCRIPT_DIR, "file503u.txt")
FEM_VM_PATH = os.path.join(SCRIPT_DIR, "file503v.txt")
TOP_FIXED_TOL = 1.0e-8
TOP_REGION_FRACTION = top_blend_fraction

# The current FEM sector spans about [-109.8, -70.2] deg, while the PINN sector spans
# about [-19.8, 19.8] deg. Rotate FEM points about y by +90 deg before comparison.
FEM_ROTATE_Y_DEG = 90.0

def infer_bc_mode_from_model_path(model_path):
    filename = os.path.basename(model_path).lower()
    if "stage1_soft" in filename:
        return "soft"
    return "hard_adf"


def infer_result_dir_from_model_path(model_path):
    filename = os.path.basename(model_path).lower()
    if "stage1_soft" in filename:
        return RESULT_STAGE1_DIR
    if "stage2_start" in filename:
        return RESULT_STAGE2_START_DIR
    if filename.startswith("pinn_sector_disp_only_model"):
        return RESULT_DIR
    return RESULT_DIR


CURRENT_RESULT_DIR = infer_result_dir_from_model_path(MODEL_PATH)


class ZeroInitCorrectionNet(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = torch.nn.SiLU()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

        for layer in self.linears[:-1]:
            torch.nn.init.xavier_normal_(layer.weight.data)
            torch.nn.init.zeros_(layer.bias.data)

        torch.nn.init.zeros_(self.linears[-1].weight.data)
        torch.nn.init.zeros_(self.linears[-1].bias.data)

    def forward(self, x_in):

        a = 2.0 * (x_in - torch.tensor([-R_outer, 0.0, -R_outer], dtype=DTYPE, device=device)) / (
            torch.tensor([R_outer, H_total, R_outer], dtype=DTYPE, device=device)
            - torch.tensor([-R_outer, 0.0, -R_outer], dtype=DTYPE, device=device)
        ) - 1.0
        for layer in self.linears[:-1]:
            a = self.activation(layer(a))
        out = self.linears[-1](a)
        return U_REF * out[:, 0:1], U_REF * out[:, 1:2], U_REF * out[:, 2:3]


class HardCorrectionModel(torch.nn.Module):
    def __init__(self, base_model, correction_net):
        super().__init__()
        self.base_model = base_model
        self.correction_net = correction_net
        self.bc_mode = "hard_adf"

    def forward(self, x_in):
        u_base, v_base, w_base = self.base_model(x_in)
        x_top = project_to_top_surface(x_in)
        u_top, v_top, w_top = self.base_model(x_top)
        u_corr_raw, v_corr_raw, w_corr_raw = self.correction_net(x_in)
        phi_top = top_surface_phi(x_in)

        u = u_base - (1.0 - phi_top) * u_top + phi_top * u_corr_raw
        v = v_base - (1.0 - phi_top) * v_top + phi_top * v_corr_raw
        w = w_base - (1.0 - phi_top) * w_top + phi_top * w_corr_raw
        return u, v, w


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "stage2_hard_correction_v1":
        base_model = PINN3D([3, 192, 192, 192, 192, 192, 3], bc_mode="soft").to(device=device, dtype=DTYPE)
        base_model.load_state_dict(checkpoint["base_model_state_dict"])
        correction_net = ZeroInitCorrectionNet([3, 192, 192, 192, 192, 192, 3]).to(device=device, dtype=DTYPE)
        correction_net.load_state_dict(checkpoint["correction_net_state_dict"])
        model = HardCorrectionModel(base_model, correction_net).to(device=device, dtype=DTYPE)
        model.eval()
        print(f"[INFO] loaded model: {os.path.abspath(model_path)}")
        print("[INFO] inferred model type for analysis: stage2_hard_correction_v1")
        print(
            f"[INFO] hard lifting during analysis follows training script: "
            f"localized top blend fraction = {top_blend_fraction:g}, ADF power = {adf_power:g}"
        )
        return model

    bc_mode = infer_bc_mode_from_model_path(model_path)
    model = PINN3D([3, 192, 192, 192, 192, 192, 3], bc_mode=bc_mode).to(device=device, dtype=DTYPE)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"[INFO] loaded model: {os.path.abspath(model_path)}")
    print(f"[INFO] inferred bc_mode for analysis: {bc_mode}")
    return model

def _clean_header(text):
    return text.replace("\ufeff", "").strip()


def read_fem_table(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        raise ValueError(f"File is empty: {path}")

    header = [_clean_header(col) for col in rows[0] if _clean_header(col)]
    data_rows = []
    for row in rows[1:]:
        clean = [col.strip() for col in row if col.strip() != ""]
        if len(clean) >= 5:
            data_rows.append(clean[:5])

    if not data_rows:
        raise ValueError(f"No valid data rows found in file: {path}")

    arr = np.array(data_rows, dtype=np.float64)
    return {
        "node": arr[:, 0].astype(np.int64),
        "x": arr[:, 1],
        "y": arr[:, 2],
        "z": arr[:, 3],
        "value": arr[:, 4],
        "header": header,
    }


def rotate_points_about_y(pts, angle_deg):
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    x_rot = x * c - z * s
    z_rot = x * s + z * c
    return np.column_stack([x_rot, y, z_rot])


def theta_deg_range(pts):
    theta_deg = np.degrees(np.arctan2(pts[:, 2], pts[:, 0]))
    return float(np.min(theta_deg)), float(np.max(theta_deg))


def print_coordinate_summary(name, pts):
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
    th_min, th_max = theta_deg_range(pts)
    print(f"[INFO] {name}")
    print(f"       x range = [{np.min(pts[:, 0]):.6e}, {np.max(pts[:, 0]):.6e}]")
    print(f"       y range = [{np.min(pts[:, 1]):.6e}, {np.max(pts[:, 1]):.6e}]")
    print(f"       z range = [{np.min(pts[:, 2]):.6e}, {np.max(pts[:, 2]):.6e}]")
    print(f"       r range = [{np.min(r):.6e}, {np.max(r):.6e}]")
    print(f"       theta range = [{th_min:.3f}, {th_max:.3f}] deg")


def get_gradients(u, x):
    return torch.autograd.grad(
        u,
        x,
        torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
    )[0]


def stress_from_disp_batch(u, v, w, batch):
    gu = get_gradients(u, batch)
    gv = get_gradients(v, batch)
    gw = get_gradients(w, batch)

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
    return sxx, syy, szz, sxy, syz, sxz


def von_mises_np(sxx, syy, szz, sxy, syz, sxz):
    return np.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def error_stats(ref, pred):
    abs_err = np.abs(pred - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1.0e-12)
    l2_rel = np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1.0e-12)
    return {
        "mean_abs": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_rel": float(np.mean(rel_err)),
        "max_rel": float(np.max(rel_err)),
        "l2_rel": float(l2_rel),
    }


def masked_error_stats(ref, pred, mask):
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    return error_stats(ref[mask], pred[mask])


def format_report_lines(name, stats):
    if stats is None:
        return [
            f"[{name}]",
            "no points in this region",
        ]
    return [
        f"[{name}]",
        f"mean_abs = {stats['mean_abs']:.6e}",
        f"max_abs  = {stats['max_abs']:.6e}",
        f"mean_rel = {stats['mean_rel']:.6e}",
        f"max_rel  = {stats['max_rel']:.6e}",
        f"l2_rel   = {stats['l2_rel']:.6e}",
    ]


def print_report(name, stats):
    for line in format_report_lines(name, stats):
        print(line)


def write_text_report(lines, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"[INFO] saved text report: {save_path}")


def build_region_masks(pts):
    y = pts[:, 1]
    top_region_y = H_total - TOP_REGION_FRACTION * H_total
    top_fixed_mask = np.isclose(y, H_total, atol=1.0e-10)
    top_region_mask = y >= top_region_y - 1.0e-12
    bulk_excluding_top_mask = ~top_region_mask
    return {
        "top_fixed_mask": top_fixed_mask,
        "top_region_mask": top_region_mask,
        "bulk_excluding_top_mask": bulk_excluding_top_mask,
        "top_region_y": top_region_y,
    }


def result_path(filename):
    return os.path.join(CURRENT_RESULT_DIR, filename)


def save_series_plot(series_dict, title, ylabel, save_path):
    idx = np.arange(1, len(next(iter(series_dict.values()))) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, values in series_dict.items():
        ax.plot(idx, values, label=label, linewidth=1.2)
    ax.set_xlabel("Point index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] saved plot: {save_path}")


def save_scatter_plot(ref, pred, title, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(ref, pred, s=14, alpha=0.45, edgecolors="none")

    vmin = float(min(np.min(ref), np.min(pred)))
    vmax = float(max(np.max(ref), np.max(pred)))
    pad = 0.03 * max(vmax - vmin, 1.0e-12)
    lo = vmin - pad
    hi = vmax + pad

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] saved plot: {save_path}")


def predict_on_points(model, pts, batch_size=4096):
    pts_tensor = torch.tensor(pts, dtype=DTYPE, device=device)

    u_list = []
    v_list = []
    w_list = []
    utotal_list = []
    vm_list = []
    syy_list = []

    with torch.enable_grad():
        for i in range(0, len(pts), batch_size):
            batch = pts_tensor[i:i + batch_size].detach().clone().requires_grad_(True)
            u, v, w = model(batch)
            sxx, syy, szz, sxy, syz, sxz = stress_from_disp_batch(u, v, w, batch)

            u_np = u.detach().cpu().numpy().reshape(-1)
            v_np = v.detach().cpu().numpy().reshape(-1)
            w_np = w.detach().cpu().numpy().reshape(-1)

            sxx_np = sxx.detach().cpu().numpy().reshape(-1)
            syy_np = syy.detach().cpu().numpy().reshape(-1)
            szz_np = szz.detach().cpu().numpy().reshape(-1)
            sxy_np = sxy.detach().cpu().numpy().reshape(-1)
            syz_np = syz.detach().cpu().numpy().reshape(-1)
            sxz_np = sxz.detach().cpu().numpy().reshape(-1)

            utotal = np.sqrt(u_np**2 + v_np**2 + w_np**2)
            vm = von_mises_np(sxx_np, syy_np, szz_np, sxy_np, syz_np, sxz_np)

            u_list.append(u_np)
            v_list.append(v_np)
            w_list.append(w_np)
            utotal_list.append(utotal)
            vm_list.append(vm)
            syy_list.append(syy_np)

    return {
        "u": np.concatenate(u_list),
        "v": np.concatenate(v_list),
        "w": np.concatenate(w_list),
        "utotal": np.concatenate(utotal_list),
        "vm": np.concatenate(vm_list),
        "syy": np.concatenate(syy_list),
    }


def build_top_surface_points(n_r=80, n_theta=80):
    r = np.linspace(R_inner, R_outer, n_r, dtype=np.float64)
    theta = np.linspace(theta_min, theta_max, n_theta, dtype=np.float64)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    x = rr * np.cos(tt)
    z = rr * np.sin(tt)
    y = np.full_like(x, H_total)
    return np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])


def build_fem_through_thickness_probe(aligned_pts, fem_utotal, fem_vm, theta_target_deg=0.0, r_target=None):
    if r_target is None:
        r_target = 0.5 * (R_inner + R_outer)

    x = aligned_pts[:, 0]
    y = aligned_pts[:, 1]
    z = aligned_pts[:, 2]
    r = np.sqrt(x**2 + z**2)
    theta_deg = np.degrees(np.arctan2(z, x))

    theta_scale = max(abs(math.degrees(theta_min)), abs(math.degrees(theta_max)), 1.0)
    r_scale = max(R_outer - R_inner, 1.0e-12)
    y_levels = np.sort(np.unique(np.round(y, 12)))

    picked_idx = []
    for y_level in y_levels:
        mask = np.isclose(y, y_level, atol=1.0e-10)
        cand = np.where(mask)[0]
        if cand.size == 0:
            continue
        score = np.abs(theta_deg[cand] - theta_target_deg) / theta_scale + np.abs(r[cand] - r_target) / r_scale
        picked_idx.append(cand[int(np.argmin(score))])

    picked_idx = np.array(picked_idx, dtype=np.int64)
    order = np.argsort(y[picked_idx])
    picked_idx = picked_idx[order]

    return {
        "pts": aligned_pts[picked_idx],
        "y": y[picked_idx],
        "r": r[picked_idx],
        "theta_deg": theta_deg[picked_idx],
        "fem_utotal": fem_utotal[picked_idx],
        "fem_vm": fem_vm[picked_idx],
        "r_target": r_target,
        "theta_target_deg": theta_target_deg,
    }


def save_through_thickness_probe_plot(probe, pred_probe, save_path):
    y_vals = probe["y"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(y_vals, probe["fem_utotal"], "o-", label="FEM", linewidth=1.3, markersize=4)
    ax.plot(y_vals, pred_probe["utotal"], "s--", label="PINN", linewidth=1.3, markersize=4)
    ax.set_title("Through-thickness Utotal")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Total displacement (m)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(y_vals, probe["fem_vm"], "o-", label="FEM", linewidth=1.3, markersize=4)
    ax.plot(y_vals, pred_probe["vm"], "s--", label="PINN", linewidth=1.3, markersize=4)
    ax.set_title("Through-thickness Von Mises")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("Von Mises stress (Pa)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(y_vals, pred_probe["v"], "o-", color="#E45756", linewidth=1.3, markersize=4)
    ax.set_title("PINN v(y)")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("v displacement (m)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(y_vals, pred_probe["syy"], "o-", color="#54A24B", linewidth=1.3, markersize=4)
    ax.set_title("PINN syy(y)")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("syy (Pa)")
    ax.grid(True, alpha=0.25)

    theta_err = float(np.max(np.abs(probe["theta_deg"] - probe["theta_target_deg"])))
    r_err = float(np.max(np.abs(probe["r"] - probe["r_target"])))
    fig.suptitle(
        "Through-thickness probe near theta = "
        f"{probe['theta_target_deg']:.1f} deg, r = {probe['r_target']:.6f} m\n"
        f"selected FEM points: max |dtheta| = {theta_err:.3f} deg, max |dr| = {r_err:.3e} m",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[INFO] saved plot: {save_path}")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RESULT_STAGE1_DIR, exist_ok=True)
    os.makedirs(RESULT_STAGE2_START_DIR, exist_ok=True)
    print(f"[INFO] result output dir = {os.path.abspath(CURRENT_RESULT_DIR)}")
    report_lines = [
        f"[INFO] result output dir = {os.path.abspath(CURRENT_RESULT_DIR)}",
        f"[INFO] model path = {os.path.abspath(MODEL_PATH)}",
    ]

    fem_u = read_fem_table(FEM_UTOTAL_PATH)
    fem_vm = read_fem_table(FEM_VM_PATH)

    xyz_u_raw = np.column_stack([fem_u["x"], fem_u["y"], fem_u["z"]])
    xyz_vm_raw = np.column_stack([fem_vm["x"], fem_vm["y"], fem_vm["z"]])

    if xyz_u_raw.shape != xyz_vm_raw.shape or not np.allclose(xyz_u_raw, xyz_vm_raw, atol=1.0e-12):
        raise ValueError("FEM displacement and stress files do not share the same coordinates.")

    print_coordinate_summary("raw FEM points", xyz_u_raw)
    xyz_u = rotate_points_about_y(xyz_u_raw, FEM_ROTATE_Y_DEG)
    xyz_vm = rotate_points_about_y(xyz_vm_raw, FEM_ROTATE_Y_DEG)
    print(f"[INFO] applied FEM rotation about y = {FEM_ROTATE_Y_DEG:.3f} deg")
    print_coordinate_summary("aligned FEM points", xyz_u)
    print(
        f"[INFO] model theta target = "
        f"[{math.degrees(theta_min):.3f}, {math.degrees(theta_max):.3f}] deg"
    )
    report_lines.extend(
        [
            f"[INFO] compared model path = {os.path.abspath(MODEL_PATH)}",
        ]
    )

    model = load_model()
    pred = predict_on_points(model, xyz_u)

    region_masks = build_region_masks(xyz_u)

    stats_u = error_stats(fem_u["value"], pred["utotal"])
    stats_vm = error_stats(fem_vm["value"], pred["vm"])
    stats_u_top_fixed = masked_error_stats(fem_u["value"], pred["utotal"], region_masks["top_fixed_mask"])
    stats_u_top_region = masked_error_stats(fem_u["value"], pred["utotal"], region_masks["top_region_mask"])
    stats_u_bulk = masked_error_stats(fem_u["value"], pred["utotal"], region_masks["bulk_excluding_top_mask"])
    stats_vm_top_fixed = masked_error_stats(fem_vm["value"], pred["vm"], region_masks["top_fixed_mask"])
    stats_vm_top_region = masked_error_stats(fem_vm["value"], pred["vm"], region_masks["top_region_mask"])
    stats_vm_bulk = masked_error_stats(fem_vm["value"], pred["vm"], region_masks["bulk_excluding_top_mask"])

    print(f"[INFO] compared points = {xyz_u.shape[0]}")
    print(
        f"[INFO] top-region split: y >= {region_masks['top_region_y']:.6e} m "
        f"({TOP_REGION_FRACTION:.2%} of thickness from the top)"
    )
    print(
        f"[INFO] top_fixed points = {np.count_nonzero(region_masks['top_fixed_mask'])} | "
        f"top_region points = {np.count_nonzero(region_masks['top_region_mask'])} | "
        f"bulk_excluding_top points = {np.count_nonzero(region_masks['bulk_excluding_top_mask'])}"
    )
    report_lines.extend(
        [
            f"[INFO] compared points = {xyz_u.shape[0]}",
            (
                f"[INFO] top-region split: y >= {region_masks['top_region_y']:.6e} m "
                f"({TOP_REGION_FRACTION:.2%} of thickness from the top)"
            ),
            (
                f"[INFO] top_fixed points = {np.count_nonzero(region_masks['top_fixed_mask'])} | "
                f"top_region points = {np.count_nonzero(region_masks['top_region_mask'])} | "
                f"bulk_excluding_top points = {np.count_nonzero(region_masks['bulk_excluding_top_mask'])}"
            ),
            "",
        ]
    )

    for name, stats in [
        ("Total displacement", stats_u),
        ("Total displacement - top fixed only", stats_u_top_fixed),
        ("Total displacement - top region", stats_u_top_region),
        ("Total displacement - bulk excluding top region", stats_u_bulk),
        ("Von Mises from displacement gradients", stats_vm),
        ("Von Mises - top fixed only", stats_vm_top_fixed),
        ("Von Mises - top region", stats_vm_top_region),
        ("Von Mises - bulk excluding top region", stats_vm_bulk),
    ]:
        print()
        print_report(name, stats)
        report_lines.extend(format_report_lines(name, stats))
        report_lines.append("")

    save_series_plot(
        {"FEM": fem_u["value"], "PINN": pred["utotal"]},
        "FEM vs PINN Total Displacement (Disp Only, Aligned)",
        "Total displacement (m)",
        result_path("fem_pinn_utotal_compare_disp_only_aligned.png"),
    )
    save_series_plot(
        {"FEM": fem_vm["value"], "PINN vm_disp": pred["vm"]},
        "FEM vs PINN Von Mises (Disp Only, Aligned)",
        "Von Mises stress (Pa)",
        result_path("fem_pinn_vm_compare_disp_only_aligned.png"),
    )
    save_scatter_plot(
        fem_u["value"],
        pred["utotal"],
        "FEM vs PINN Total Displacement Scatter (Disp Only, Aligned)",
        "FEM total displacement (m)",
        "PINN total displacement (m)",
        result_path("fem_pinn_utotal_scatter_disp_only_aligned.png"),
    )
    save_scatter_plot(
        fem_vm["value"],
        pred["vm"],
        "FEM vs PINN Von Mises Scatter (Disp Only, Aligned)",
        "FEM Von Mises stress (Pa)",
        "PINN Von Mises stress (Pa)",
        result_path("fem_pinn_vm_scatter_disp_only_aligned.png"),
    )

    probe = build_fem_through_thickness_probe(xyz_u, fem_u["value"], fem_vm["value"])
    pred_probe = predict_on_points(model, probe["pts"])
    print(
        "[INFO] through-thickness probe near "
        f"theta={probe['theta_target_deg']:.1f} deg, r={probe['r_target']:.6f} m"
    )
    print(
        f"[INFO] selected {probe['pts'].shape[0]} FEM levels | "
        f"theta range = [{np.min(probe['theta_deg']):.3f}, {np.max(probe['theta_deg']):.3f}] deg | "
        f"r range = [{np.min(probe['r']):.6e}, {np.max(probe['r']):.6e}] m"
    )
    save_through_thickness_probe_plot(
        probe,
        pred_probe,
        result_path("through_thickness_probe_disp_only.png"),
    )

    top_pts = build_top_surface_points()
    top_pred = predict_on_points(model, top_pts)
    top_utotal = top_pred["utotal"]
    top_abs_components = np.column_stack(
        [np.abs(top_pred["u"]), np.abs(top_pred["v"]), np.abs(top_pred["w"])]
    )

    print(f"\n[Top surface displacement diagnostic - current model]")
    print(f"points = {top_pts.shape[0]}")
    print(f"utotal mean = {np.mean(top_utotal):.6e}")
    print(f"utotal max  = {np.max(top_utotal):.6e}")
    print(f"|u| max = {np.max(top_abs_components[:, 0]):.6e}")
    print(f"|v| max = {np.max(top_abs_components[:, 1]):.6e}")
    print(f"|w| max = {np.max(top_abs_components[:, 2]):.6e}")
    print(f"fraction(utotal < {TOP_FIXED_TOL:.1e}) = {np.mean(top_utotal < TOP_FIXED_TOL):.6f}")
    if hasattr(model, "bc_mode") and model.bc_mode == "soft":
        print("[INFO] current model uses soft Dirichlet treatment on the top surface during analysis.")
        top_mode_line = "[INFO] current model uses soft Dirichlet treatment on the top surface during analysis."
    else:
        print("[INFO] current model uses exact top Dirichlet enforcement via the stage-2 ADF correction form.")
        top_mode_line = "[INFO] current model uses exact top Dirichlet enforcement via the stage-2 ADF correction form."

    report_lines.extend(
        [
            "[Top surface displacement diagnostic - current model]",
            f"points = {top_pts.shape[0]}",
            f"utotal mean = {np.mean(top_utotal):.6e}",
            f"utotal max  = {np.max(top_utotal):.6e}",
            f"|u| max = {np.max(top_abs_components[:, 0]):.6e}",
            f"|v| max = {np.max(top_abs_components[:, 1]):.6e}",
            f"|w| max = {np.max(top_abs_components[:, 2]):.6e}",
            f"fraction(utotal < {TOP_FIXED_TOL:.1e}) = {np.mean(top_utotal < TOP_FIXED_TOL):.6f}",
            top_mode_line,
        ]
    )

    save_series_plot(
        {
            "|u|": np.abs(top_pred["u"]),
            "|v|": np.abs(top_pred["v"]),
            "|w|": np.abs(top_pred["w"]),
            "utotal": top_utotal,
        },
        "Top surface displacement diagnostic (Disp Only, Current Model)",
        "Displacement magnitude (m)",
        result_path("top_fixed_displacement_diagnostic_disp_only.png"),
    )
    write_text_report(report_lines, result_path("error_report_disp_only.txt"))


if __name__ == "__main__":
    main()
