import os
import numpy as np
import torch
import pyvista as pv

from 扇形 import (
    SirenNet,
    device,
    dtype,
    lam,
    mu,
    R_inner,
    R_outer,
    H_total,
    theta_min,
    theta_max,
    x_groove,
    R_groove,
    y_groove_bottom,
    y_groove_top,
    hidden_features,
    hidden_layers,
    first_omega_0,
    hidden_omega_0,
    in_features,
    model_save_name,
    evaluate_points_with_stress,
    von_mises_from_stress,
)


# =========================================================
# 1. 加载扇形.py 训练得到的 SIREN 模型
# =========================================================
def infer_checkpoint_out_features(state):
    final_weight_keys = [
        key for key in state.keys()
        if key.startswith("net.") and key.endswith(".weight")
    ]
    if not final_weight_keys:
        raise ValueError("无法从模型权重中识别 SIREN 输出层。")

    final_weight_key = max(
        final_weight_keys,
        key=lambda key: int(key.split(".")[1])
    )
    return int(state[final_weight_key].shape[0])


def load_sector_siren_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            model_save_name,
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    state = torch.load(model_path, map_location=device)
    checkpoint_out_features = infer_checkpoint_out_features(state)

    model = SirenNet(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=checkpoint_out_features,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)

    model.load_state_dict(state)
    model.output_mode = "mixed" if checkpoint_out_features == 9 else "disp_only"
    model.eval()

    print(f"成功加载模型权重: {model_path}")
    print(f"模型输出维度: {checkpoint_out_features} ({model.output_mode})")
    return model


model = load_sector_siren_model("pinn_sector_siren_model (1).pth")



# =========================================================
# 2. 在给定点上做 PINN 推理：位移 + 等效应力
# 新混合模型直接输出应力；旧位移模型则由位移梯度反算应力。
# =========================================================
def evaluate_legacy_disp_model_with_stress(model, points_np, batch_size=4096):
    model.eval()

    all_disp = []
    all_vm = []
    all_sigma = []

    for i in range(0, points_np.shape[0], batch_size):
        pts_np = points_np[i:i + batch_size]
        pts = torch.tensor(pts_np, dtype=dtype, device=device, requires_grad=True)

        disp_all = model(pts)
        disp = disp_all[:, 0:3]
        u = disp[:, 0:1]
        v = disp[:, 1:2]
        w = disp[:, 2:3]

        grad_u = torch.autograd.grad(
            u, pts,
            grad_outputs=torch.ones_like(u),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_v = torch.autograd.grad(
            v, pts,
            grad_outputs=torch.ones_like(v),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_w = torch.autograd.grad(
            w, pts,
            grad_outputs=torch.ones_like(w),
            create_graph=False,
            retain_graph=True,
            only_inputs=True
        )[0]

        du_dx, du_dy, du_dz = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        dv_dx, dv_dy, dv_dz = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        dw_dx, dw_dy, dw_dz = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

        eps_xx = du_dx
        eps_yy = dv_dy
        eps_zz = dw_dz
        eps_xy = 0.5 * (du_dy + dv_dx)
        eps_yz = 0.5 * (dv_dz + dw_dy)
        eps_xz = 0.5 * (du_dz + dw_dx)

        trace_eps = eps_xx + eps_yy + eps_zz
        sxx = lam * trace_eps + 2.0 * mu * eps_xx
        syy = lam * trace_eps + 2.0 * mu * eps_yy
        szz = lam * trace_eps + 2.0 * mu * eps_zz
        sxy = 2.0 * mu * eps_xy
        syz = 2.0 * mu * eps_yz
        sxz = 2.0 * mu * eps_xz

        vm = von_mises_from_stress(sxx, syy, szz, sxy, syz, sxz)
        sigma = torch.cat([sxx, syy, szz, sxy, syz, sxz], dim=1)

        all_disp.append(disp.detach().cpu().numpy())
        all_vm.append(vm.detach().cpu().numpy())
        all_sigma.append(sigma.detach().cpu().numpy())

    disp = np.vstack(all_disp)
    vm = np.vstack(all_vm)
    sigma = np.vstack(all_sigma)
    umag = np.linalg.norm(disp, axis=1, keepdims=True)
    return disp, umag, vm, sigma


def predict_on_points(model, pts, stress_scale=1.0, disp_scale=1.0, batch_size=4096):
    if getattr(model, "output_mode", "mixed") == "disp_only":
        disp, umag, vm, sigma = evaluate_legacy_disp_model_with_stress(
            model=model,
            points_np=pts,
            batch_size=batch_size,
        )
    else:
        disp, umag, vm, sigma = evaluate_points_with_stress(
            model=model,
            points_np=pts,
            batch_size=batch_size,
        )

    return {
        "u": disp[:, 0] * disp_scale,
        "v": disp[:, 1] * disp_scale,
        "w": disp[:, 2] * disp_scale,
        "utotal": umag[:, 0] * disp_scale,
        "vm": vm[:, 0] * stress_scale,
        "sxx": sigma[:, 0] * stress_scale,
        "syy": sigma[:, 1] * stress_scale,
        "szz": sigma[:, 2] * stress_scale,
        "sxy": sigma[:, 3] * stress_scale,
        "syz": sigma[:, 4] * stress_scale,
        "sxz": sigma[:, 5] * stress_scale,
    }


# =========================================================
# 3. 几何辅助函数
# =========================================================
def groove_rho_np(x, z):
    return np.sqrt((x - x_groove) ** 2 + z ** 2)


def top_surface_valid_mask(xx, zz, tol=1e-12):
    """
    顶面 y=H_total 需要扣掉凹槽开口。
    """
    rho = groove_rho_np(xx, zz)
    return rho >= (R_groove + tol)


# =========================================================
# 4. 构造扇形模型各个真实边界面的参数网格
# =========================================================
def build_sector_surface_parametric_grids(
    R_inner,
    R_outer,
    H_total,
    theta_min,
    theta_max,
    x_groove,
    R_groove,
    y_groove_bottom,
    y_groove_top,
    n_theta=220,
    n_y=120,
    n_r=120,
    n_alpha=180,
    n_groove_y=80,
    n_groove_r=80,
):
    theta = np.linspace(theta_min, theta_max, n_theta, endpoint=True)
    y_line = np.linspace(0.0, H_total, n_y)
    alpha_closed = np.linspace(0.0, 2.0 * np.pi, n_alpha + 1, endpoint=True)

    TH_o, YY_o = np.meshgrid(theta, y_line, indexing="ij")
    XX_o = R_outer * np.cos(TH_o)
    ZZ_o = R_outer * np.sin(TH_o)

    TH_i, YY_i = np.meshgrid(theta, y_line, indexing="ij")
    XX_i = R_inner * np.cos(TH_i)
    ZZ_i = R_inner * np.sin(TH_i)

    r_bottom = np.linspace(R_inner, R_outer, n_r)
    RR_b, TH_b = np.meshgrid(r_bottom, theta, indexing="ij")
    XX_b = RR_b * np.cos(TH_b)
    ZZ_b = RR_b * np.sin(TH_b)
    YY_b = np.zeros_like(XX_b)

    r_top = np.linspace(R_inner, R_outer, n_r)
    RR_t, TH_t = np.meshgrid(r_top, theta, indexing="ij")
    XX_t = RR_t * np.cos(TH_t)
    ZZ_t = RR_t * np.sin(TH_t)
    YY_t = np.full_like(XX_t, H_total)
    MASK_top = top_surface_valid_mask(XX_t, ZZ_t)

    r_rad = np.linspace(R_inner, R_outer, n_r)
    y_rad = np.linspace(0.0, H_total, n_y)
    RR_rm, YY_rm = np.meshgrid(r_rad, y_rad, indexing="ij")
    TH_rm = np.full_like(RR_rm, theta_min)
    XX_rm = RR_rm * np.cos(TH_rm)
    ZZ_rm = RR_rm * np.sin(TH_rm)

    RR_rp, YY_rp = np.meshgrid(r_rad, y_rad, indexing="ij")
    TH_rp = np.full_like(RR_rp, theta_max)
    XX_rp = RR_rp * np.cos(TH_rp)
    ZZ_rp = RR_rp * np.sin(TH_rp)

    rho_gb = np.linspace(0.0, R_groove, n_groove_r)
    RHO_gb, ALPHA_gb = np.meshgrid(rho_gb, alpha_closed, indexing="ij")
    XX_gb = x_groove + RHO_gb * np.cos(ALPHA_gb)
    ZZ_gb = RHO_gb * np.sin(ALPHA_gb)
    YY_gb = np.full_like(XX_gb, y_groove_bottom)

    y_gs = np.linspace(y_groove_bottom, y_groove_top, n_groove_y)
    ALPHA_gs, YY_gs = np.meshgrid(alpha_closed, y_gs, indexing="ij")
    XX_gs = x_groove + R_groove * np.cos(ALPHA_gs)
    ZZ_gs = R_groove * np.sin(ALPHA_gs)

    return {
        "outer": {"xx": XX_o, "yy": YY_o, "zz": ZZ_o, "mask": None},
        "inner": {"xx": XX_i, "yy": YY_i, "zz": ZZ_i, "mask": None},
        "bottom": {"xx": XX_b, "yy": YY_b, "zz": ZZ_b, "mask": None},
        "top": {"xx": XX_t, "yy": YY_t, "zz": ZZ_t, "mask": MASK_top},
        "radial_min": {"xx": XX_rm, "yy": YY_rm, "zz": ZZ_rm, "mask": None},
        "radial_max": {"xx": XX_rp, "yy": YY_rp, "zz": ZZ_rp, "mask": None},
        "groove_bottom": {"xx": XX_gb, "yy": YY_gb, "zz": ZZ_gb, "mask": None},
        "groove_side": {"xx": XX_gs, "yy": YY_gs, "zz": ZZ_gs, "mask": None},
    }


# =========================================================
# 5. 规则曲面 -> PyVista StructuredGrid
# =========================================================
def make_structured_surface(xx, yy, zz, field_dict, deformation_vis_scale=0.0):
    u = field_dict["u"].reshape(xx.shape)
    v = field_dict["v"].reshape(xx.shape)
    w = field_dict["w"].reshape(xx.shape)
    utotal = field_dict["utotal"].reshape(xx.shape)
    vm = field_dict["vm"].reshape(xx.shape)

    x_plot = xx + deformation_vis_scale * u
    y_plot = yy + deformation_vis_scale * v
    z_plot = zz + deformation_vis_scale * w

    grid = pv.StructuredGrid(x_plot, y_plot, z_plot)
    grid["u"] = u.ravel(order="F")
    grid["v"] = v.ravel(order="F")
    grid["w"] = w.ravel(order="F")
    grid["utotal"] = utotal.ravel(order="F")
    grid["vm"] = vm.ravel(order="F")

    surf = grid.extract_surface().triangulate().clean(tolerance=1e-10)
    surf = surf.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )
    return surf


# =========================================================
# 6. 带孔洞裁切的参数面 -> PolyData
# =========================================================
def make_masked_quad_surface(xx, yy, zz, valid_mask, field_dict, deformation_vis_scale=0.0):
    assert xx.shape == yy.shape == zz.shape == valid_mask.shape

    u = field_dict["u"].reshape(xx.shape)
    v = field_dict["v"].reshape(xx.shape)
    w = field_dict["w"].reshape(xx.shape)
    utotal = field_dict["utotal"].reshape(xx.shape)
    vm = field_dict["vm"].reshape(xx.shape)

    x_plot = xx + deformation_vis_scale * u
    y_plot = yy + deformation_vis_scale * v
    z_plot = zz + deformation_vis_scale * w

    n1, n2 = xx.shape
    points = np.column_stack([
        x_plot.ravel(order="F"),
        y_plot.ravel(order="F"),
        z_plot.ravel(order="F"),
    ])

    def idx(i, j):
        return i + n1 * j

    faces = []
    for i in range(n1 - 1):
        for j in range(n2 - 1):
            if valid_mask[i, j] and valid_mask[i + 1, j] and valid_mask[i + 1, j + 1] and valid_mask[i, j + 1]:
                faces.extend([4, idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    surf = pv.PolyData(points, np.array(faces, dtype=np.int64))
    surf["u"] = u.ravel(order="F")
    surf["v"] = v.ravel(order="F")
    surf["w"] = w.ravel(order="F")
    surf["utotal"] = utotal.ravel(order="F")
    surf["vm"] = vm.ravel(order="F")

    surf = surf.triangulate().clean(tolerance=1e-10)
    surf = surf.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )
    return surf


# =========================================================
# 7. 构造扇形模型所有边界面的 PyVista 网格
# =========================================================
def build_sector_surfaces_pyvista(
    model,
    stress_scale=1.0,
    disp_scale=1.0,
    deformation_vis_scale=0.0,
    n_theta=220,
    n_y=120,
    n_r=120,
    n_alpha=180,
    n_groove_y=80,
    n_groove_r=80,
    batch_size=4096,
):
    surf_grids = build_sector_surface_parametric_grids(
        R_inner=R_inner,
        R_outer=R_outer,
        H_total=H_total,
        theta_min=theta_min,
        theta_max=theta_max,
        x_groove=x_groove,
        R_groove=R_groove,
        y_groove_bottom=y_groove_bottom,
        y_groove_top=y_groove_top,
        n_theta=n_theta,
        n_y=n_y,
        n_r=n_r,
        n_alpha=n_alpha,
        n_groove_y=n_groove_y,
        n_groove_r=n_groove_r,
    )

    mesh_dict = {}
    for name, data in surf_grids.items():
        xx = data["xx"]
        yy = data["yy"]
        zz = data["zz"]
        mask = data["mask"]

        pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        pred = predict_on_points(
            model=model,
            pts=pts,
            stress_scale=stress_scale,
            disp_scale=disp_scale,
            batch_size=batch_size,
        )

        if mask is None:
            mesh_dict[name] = make_structured_surface(
                xx,
                yy,
                zz,
                pred,
                deformation_vis_scale=deformation_vis_scale,
            )
        else:
            mesh_dict[name] = make_masked_quad_surface(
                xx,
                yy,
                zz,
                mask,
                pred,
                deformation_vis_scale=deformation_vis_scale,
            )

        print(
            f"[完成] {name:<14} | "
            f"|u|max={np.max(pred['utotal']):.6e} m | "
            f"vm,max={np.max(pred['vm']):.6e} Pa"
        )

    return mesh_dict


# =========================================================
# 8. 合并与绘图
# =========================================================
def merge_sector_surfaces(mesh_dict):
    merged = None
    for mesh in mesh_dict.values():
        merged = mesh.copy() if merged is None else merged.merge(mesh, merge_points=True, tolerance=1e-9)

    merged = merged.clean(tolerance=1e-9)
    merged = merged.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )
    return merged


def _add_common_scene(
    plotter,
    merged_mesh,
    field_name,
    title,
    cmap,
    clim,
    show_edges,
    smooth_shading,
    parallel_projection,
    background,
):
    plotter.set_background(background)
    plotter.add_mesh(
        merged_mesh,
        scalars=field_name,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        smooth_shading=smooth_shading,
        specular=0.1,
        diffuse=1.0,
        ambient=0.2,
        scalar_bar_args={
            "title": field_name,
            "vertical": True,
            "position_x": 0.87,
            "position_y": 0.12,
            "height": 0.74,
            "width": 0.06,
            "label_font_size": 12,
            "title_font_size": 14,
        },
    )
    plotter.add_axes()
    plotter.add_title(title, font_size=16)

    if parallel_projection:
        plotter.enable_parallel_projection()

    plotter.view_isometric()
    plotter.render()


def plot_sector_field_pyvista(
    mesh_dict,
    field_name="vm",
    title="Von Mises Stress",
    cmap="jet",
    clim=None,
    show_edges=False,
    smooth_shading=True,
    parallel_projection=True,
    background="white",
    screenshot=None,
    show_window=True,
):
    merged_mesh = merge_sector_surfaces(mesh_dict)
    all_vals = merged_mesh[field_name]

    if clim is None:
        clim = [float(np.min(all_vals)), float(np.max(all_vals))]

    if screenshot is not None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), screenshot)
        save_plotter = pv.Plotter(off_screen=True, window_size=(1400, 1000))
        _add_common_scene(
            save_plotter,
            merged_mesh,
            field_name,
            title,
            cmap,
            clim,
            show_edges,
            smooth_shading,
            parallel_projection,
            background,
        )
        save_plotter.screenshot(save_path)
        save_plotter.close()
        print(f"截图已保存到: {save_path}")

    if show_window:
        view_plotter = pv.Plotter(window_size=(1400, 1000))
        _add_common_scene(
            view_plotter,
            merged_mesh,
            field_name,
            title,
            cmap,
            clim,
            show_edges,
            smooth_shading,
            parallel_projection,
            background,
        )
        view_plotter.show()
        view_plotter.close()


def plot_sector_disp_and_vm_pyvista(
    model,
    stress_scale=1.0,
    disp_scale=1.0,
    deformation_vis_scale=0.0,
    n_theta=220,
    n_y=120,
    n_r=120,
    n_alpha=180,
    n_groove_y=80,
    n_groove_r=80,
    batch_size=4096,
):
    mesh_dict = build_sector_surfaces_pyvista(
        model=model,
        stress_scale=stress_scale,
        disp_scale=disp_scale,
        deformation_vis_scale=deformation_vis_scale,
        n_theta=n_theta,
        n_y=n_y,
        n_r=n_r,
        n_alpha=n_alpha,
        n_groove_y=n_groove_y,
        n_groove_r=n_groove_r,
        batch_size=batch_size,
    )

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="utotal",
        title="Total Deformation (SIREN Sector + Groove)",
        screenshot="sector_siren_total_deformation_pyvista.png",
        show_window=True,
    )

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="vm",
        title="Von Mises Stress (SIREN Sector + Groove)",
        screenshot="sector_siren_von_mises_pyvista.png",
        show_window=True,
    )

    mesh_bottom = mesh_dict["bottom"]
    plot_sector_field_pyvista(
        mesh_dict={"bottom": mesh_bottom},
        field_name="v",
        title="Bottom Surface Y-Displacement",
        screenshot="sector_siren_bottom_v.png",
        show_window=True,
    )

    plot_sector_field_pyvista(
        mesh_dict={"bottom": mesh_bottom},
        field_name="utotal",
        title="Bottom Surface Total Displacement",
        screenshot="sector_siren_bottom_utotal.png",
        show_window=True,
    )


# =========================================================
# 9. 运行示例
# =========================================================
if __name__ == "__main__":
    stress_scale = 1.0
    disp_scale = 1.0

    plot_sector_disp_and_vm_pyvista(
        model=model,
        stress_scale=stress_scale,
        disp_scale=disp_scale,
        deformation_vis_scale=0.0,
        n_theta=220,
        n_y=120,
        n_r=120,
        n_alpha=180,
        n_groove_y=80,
        n_groove_r=80,
        batch_size=4096,
    )
