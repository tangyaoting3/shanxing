import os
import torch
import numpy as np
import pyvista as pv

# =========================================================
# 你自己的模型导入
# 请把模块名“扇形2”改成你实际训练脚本对应的文件名
# 要保证下面这些变量在训练脚本里是模块级可导入的
# =========================================================
from 扇形2 import (
    PINN3D, DTYPE, device,
    R_inner, R_outer, H_total,
    theta_min, theta_max,
    x_groove, R_groove,
    y_groove_bottom, y_groove_top,
    E_val, q_load, U_REF, SIGMA_REF,
    lmda, mu
)

# =========================================================
# 1. 加载模型
# =========================================================
model = PINN3D([3,192, 192, 192, 192, 192, 9]).to(device=device, dtype=DTYPE)

model_path = "pinn_sector_groove_model (14).pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("成功加载模型权重！")
    print(f"[INFO] model_path = {os.path.abspath(model_path)}")
    print(f"[INFO] E_val      = {E_val:.6e} Pa")
    print(f"[INFO] q_load     = {q_load:.6e} Pa")
    print(f"[INFO] U_REF      = {U_REF:.6e} m")
    print(f"[INFO] SIGMA_REF  = {SIGMA_REF:.6e} Pa")
else:
    raise FileNotFoundError(f"找不到模型文件: {model_path}")

model.eval()


# =========================================================
# 2. 在给定点上做 PINN 推理：位移 + 等效应力
# 坐标系严格按你的扇形模型：
#   x: 水平向右
#   y: 竖直向上（高度方向）
#   z: 垂直于 x-y 平面
# 位移分量：
#   u, v, w 分别对应 x, y, z 方向
# =========================================================
def von_mises_np(sxx, syy, szz, sxy, syz, sxz):
    return np.sqrt(
        0.5 * (
            (sxx - syy) ** 2 +
            (syy - szz) ** 2 +
            (szz - sxx) ** 2 +
            6.0 * (sxy**2 + syz**2 + sxz**2)
        )
    )


def get_gradients(u, x):
    return torch.autograd.grad(
        u,
        x,
        torch.ones_like(u),
        create_graph=False,
        retain_graph=True,
    )[0]


def predict_on_points(model, pts, stress_scale=1.0, disp_scale=1.0, batch_size=4096):
    pts_tensor = torch.tensor(pts, dtype=DTYPE, device=device)

    u_list, v_list, w_list = [], [], []
    utotal_list = []
    vm_net_list, vm_disp_list = [], []
    vm_abs_diff_list, vm_rel_diff_list = [], []

    with torch.enable_grad():
        for i in range(0, len(pts), batch_size):
            batch = pts_tensor[i:i + batch_size].detach().clone().requires_grad_(True)
            u, v, w, sxx, syy, szz, sxy, syz, sxz = model(batch)

            gu = get_gradients(u, batch)
            gv = get_gradients(v, batch)
            gw = get_gradients(w, batch)

            ux, uy, uz = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
            vx, vy, vz = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
            wx, wy, wz = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]
            div_u = ux + vy + wz

            sxx_disp = lmda * div_u + 2.0 * mu * ux
            syy_disp = lmda * div_u + 2.0 * mu * vy
            szz_disp = lmda * div_u + 2.0 * mu * wz
            sxy_disp = mu * (uy + vx)
            syz_disp = mu * (vz + wy)
            sxz_disp = mu * (uz + wx)

            u_np = u.detach().cpu().numpy().reshape(-1) * disp_scale
            v_np = v.detach().cpu().numpy().reshape(-1) * disp_scale
            w_np = w.detach().cpu().numpy().reshape(-1) * disp_scale

            sxx_np = sxx.detach().cpu().numpy().reshape(-1) * stress_scale
            syy_np = syy.detach().cpu().numpy().reshape(-1) * stress_scale
            szz_np = szz.detach().cpu().numpy().reshape(-1) * stress_scale
            sxy_np = sxy.detach().cpu().numpy().reshape(-1) * stress_scale
            syz_np = syz.detach().cpu().numpy().reshape(-1) * stress_scale
            sxz_np = sxz.detach().cpu().numpy().reshape(-1) * stress_scale

            sxx_disp_np = sxx_disp.detach().cpu().numpy().reshape(-1) * stress_scale
            syy_disp_np = syy_disp.detach().cpu().numpy().reshape(-1) * stress_scale
            szz_disp_np = szz_disp.detach().cpu().numpy().reshape(-1) * stress_scale
            sxy_disp_np = sxy_disp.detach().cpu().numpy().reshape(-1) * stress_scale
            syz_disp_np = syz_disp.detach().cpu().numpy().reshape(-1) * stress_scale
            sxz_disp_np = sxz_disp.detach().cpu().numpy().reshape(-1) * stress_scale

            utotal = np.sqrt(u_np**2 + v_np**2 + w_np**2)
            vm_net = von_mises_np(sxx_np, syy_np, szz_np, sxy_np, syz_np, sxz_np)
            vm_disp = von_mises_np(
                sxx_disp_np, syy_disp_np, szz_disp_np,
                sxy_disp_np, syz_disp_np, sxz_disp_np
            )
            vm_abs_diff = np.abs(vm_net - vm_disp)
            vm_rel_diff = vm_abs_diff / np.maximum(np.maximum(np.abs(vm_net), np.abs(vm_disp)), 1.0)

            u_list.append(u_np)
            v_list.append(v_np)
            w_list.append(w_np)
            utotal_list.append(utotal)
            vm_net_list.append(vm_net)
            vm_disp_list.append(vm_disp)
            vm_abs_diff_list.append(vm_abs_diff)
            vm_rel_diff_list.append(vm_rel_diff)

    return {
        "u": np.concatenate(u_list),
        "v": np.concatenate(v_list),
        "w": np.concatenate(w_list),
        "utotal": np.concatenate(utotal_list),
        "vm": np.concatenate(vm_net_list),
        "vm_net": np.concatenate(vm_net_list),
        "vm_disp": np.concatenate(vm_disp_list),
        "vm_abs_diff": np.concatenate(vm_abs_diff_list),
        "vm_rel_diff": np.concatenate(vm_rel_diff_list),
    }


# =========================================================
# 3. 几何辅助函数
# =========================================================
def groove_rho_np(x, z):
    return np.sqrt((x - x_groove) ** 2 + z ** 2)

def top_surface_valid_mask(xx, zz, tol=1e-12):
    """
    顶面 y=H_total 需要扣掉凹槽开口：
        sqrt((x-x_groove)^2 + z^2) <= R_groove
    这个区域在顶面上是空的，因此顶面有效区域是其补集。
    """
    rho = groove_rho_np(xx, zz)
    return rho >= (R_groove + tol)


# =========================================================
# 4. 构造扇形模型各个真实边界面的参数网格
#
# 包括：
#   1) 外圆柱面
#   2) 内圆柱面
#   3) 底面
#   4) 顶面（扣掉凹槽开口）
#   5) 两个径向侧面
#   6) 凹槽底面
#   7) 凹槽侧壁
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

    # groove 圆周参数，闭合
    alpha_closed = np.linspace(0.0, 2.0 * np.pi, n_alpha + 1, endpoint=True)

    # -----------------------------------------------------
    # 外圆柱面：r = R_outer, 参数(theta, y)
    # x = R_outer cos(theta), z = R_outer sin(theta), y = y
    # -----------------------------------------------------
    TH_o, YY_o = np.meshgrid(theta, y_line, indexing="ij")
    XX_o = R_outer * np.cos(TH_o)
    ZZ_o = R_outer * np.sin(TH_o)

    # -----------------------------------------------------
    # 内圆柱面：r = R_inner
    # -----------------------------------------------------
    TH_i, YY_i = np.meshgrid(theta, y_line, indexing="ij")
    XX_i = R_inner * np.cos(TH_i)
    ZZ_i = R_inner * np.sin(TH_i)

    # -----------------------------------------------------
    # 底面：y = 0, 参数(r, theta)
    # -----------------------------------------------------
    r_bottom = np.linspace(R_inner, R_outer, n_r)
    RR_b, TH_b = np.meshgrid(r_bottom, theta, indexing="ij")
    XX_b = RR_b * np.cos(TH_b)
    ZZ_b = RR_b * np.sin(TH_b)
    YY_b = np.zeros_like(XX_b)

    # -----------------------------------------------------
    # 顶面：y = H_total, 参数(r, theta)
    # 但需要扣掉凹槽开口
    # -----------------------------------------------------
    r_top = np.linspace(R_inner, R_outer, n_r)
    RR_t, TH_t = np.meshgrid(r_top, theta, indexing="ij")
    XX_t = RR_t * np.cos(TH_t)
    ZZ_t = RR_t * np.sin(TH_t)
    YY_t = np.full_like(XX_t, H_total)
    MASK_top = top_surface_valid_mask(XX_t, ZZ_t)

    # -----------------------------------------------------
    # 两个径向侧面：theta = theta_min / theta_max, 参数(r, y)
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 凹槽底面：y = y_groove_bottom, 参数(rho, alpha)
    # x = x_groove + rho cos(alpha), z = rho sin(alpha)
    # -----------------------------------------------------
    rho_gb = np.linspace(0.0, R_groove, n_groove_r)
    RHO_gb, ALPHA_gb = np.meshgrid(rho_gb, alpha_closed, indexing="ij")
    XX_gb = x_groove + RHO_gb * np.cos(ALPHA_gb)
    ZZ_gb = RHO_gb * np.sin(ALPHA_gb)
    YY_gb = np.full_like(XX_gb, y_groove_bottom)

    # -----------------------------------------------------
    # 凹槽侧壁：rho = R_groove, 参数(alpha, y)
    # -----------------------------------------------------
    y_gs = np.linspace(y_groove_bottom, y_groove_top, n_groove_y)
    ALPHA_gs, YY_gs = np.meshgrid(alpha_closed, y_gs, indexing="ij")
    XX_gs = x_groove + R_groove * np.cos(ALPHA_gs)
    ZZ_gs = R_groove * np.sin(ALPHA_gs)

    return {
        "outer": {
            "xx": XX_o, "yy": YY_o, "zz": ZZ_o, "mask": None
        },
        "inner": {
            "xx": XX_i, "yy": YY_i, "zz": ZZ_i, "mask": None
        },
        "bottom": {
            "xx": XX_b, "yy": YY_b, "zz": ZZ_b, "mask": None
        },
        "top": {
            "xx": XX_t, "yy": YY_t, "zz": ZZ_t, "mask": MASK_top
        },
        "radial_min": {
            "xx": XX_rm, "yy": YY_rm, "zz": ZZ_rm, "mask": None
        },
        "radial_max": {
            "xx": XX_rp, "yy": YY_rp, "zz": ZZ_rp, "mask": None
        },
        "groove_bottom": {
            "xx": XX_gb, "yy": YY_gb, "zz": ZZ_gb, "mask": None
        },
        "groove_side": {
            "xx": XX_gs, "yy": YY_gs, "zz": ZZ_gs, "mask": None
        },
    }


# =========================================================
# 5. 规则曲面 -> PyVista StructuredGrid
# 用于没有“孔洞裁切”的规则面
# =========================================================
def make_structured_surface(xx, yy, zz, field_dict, deformation_vis_scale=0.0):
    u = field_dict["u"].reshape(xx.shape)
    v = field_dict["v"].reshape(xx.shape)
    w = field_dict["w"].reshape(xx.shape)

    utotal = field_dict["utotal"].reshape(xx.shape)
    vm = field_dict["vm"].reshape(xx.shape)
    vm_net = field_dict["vm_net"].reshape(xx.shape)
    vm_disp = field_dict["vm_disp"].reshape(xx.shape)
    vm_abs_diff = field_dict["vm_abs_diff"].reshape(xx.shape)
    vm_rel_diff = field_dict["vm_rel_diff"].reshape(xx.shape)

    # 注意：这里位移分量对应 x/y/z
    x_plot = xx + deformation_vis_scale * u
    y_plot = yy + deformation_vis_scale * v
    z_plot = zz + deformation_vis_scale * w

    grid = pv.StructuredGrid(x_plot, y_plot, z_plot)

    grid["u"] = u.ravel(order="F")
    grid["v"] = v.ravel(order="F")
    grid["w"] = w.ravel(order="F")
    grid["utotal"] = utotal.ravel(order="F")
    grid["vm"] = vm.ravel(order="F")
    grid["vm_net"] = vm_net.ravel(order="F")
    grid["vm_disp"] = vm_disp.ravel(order="F")
    grid["vm_abs_diff"] = vm_abs_diff.ravel(order="F")
    grid["vm_rel_diff"] = vm_rel_diff.ravel(order="F")

    surf = grid.extract_surface().triangulate().clean(tolerance=1e-10)

    # 避开 splitting 参数，兼容你之前遇到的 PyVista 版本差异
    surf = surf.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

    return surf


# =========================================================
# 6. 带“孔洞裁切”的参数面 -> PolyData
# 这里只给顶面用，因为顶面需要扣除凹槽开口
# =========================================================
def make_masked_quad_surface(xx, yy, zz, valid_mask, field_dict, deformation_vis_scale=0.0):
    """
    对规则参数网格，只保留 valid_mask=True 的四边形单元。
    适用于：顶面扣除凹槽开口。
    """
    assert xx.shape == yy.shape == zz.shape == valid_mask.shape

    u = field_dict["u"].reshape(xx.shape)
    v = field_dict["v"].reshape(xx.shape)
    w = field_dict["w"].reshape(xx.shape)
    utotal = field_dict["utotal"].reshape(xx.shape)
    vm = field_dict["vm"].reshape(xx.shape)
    vm_net = field_dict["vm_net"].reshape(xx.shape)
    vm_disp = field_dict["vm_disp"].reshape(xx.shape)
    vm_abs_diff = field_dict["vm_abs_diff"].reshape(xx.shape)
    vm_rel_diff = field_dict["vm_rel_diff"].reshape(xx.shape)

    x_plot = xx + deformation_vis_scale * u
    y_plot = yy + deformation_vis_scale * v
    z_plot = zz + deformation_vis_scale * w

    n1, n2 = xx.shape

    # Fortran 顺序扁平化，与 StructuredGrid 的数据顺序保持一致
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
                faces.extend([
                    4,
                    idx(i, j),
                    idx(i + 1, j),
                    idx(i + 1, j + 1),
                    idx(i, j + 1)
                ])

    faces = np.array(faces, dtype=np.int64)

    surf = pv.PolyData(points, faces)

    surf["u"] = u.ravel(order="F")
    surf["v"] = v.ravel(order="F")
    surf["w"] = w.ravel(order="F")
    surf["utotal"] = utotal.ravel(order="F")
    surf["vm"] = vm.ravel(order="F")
    surf["vm_net"] = vm_net.ravel(order="F")
    surf["vm_disp"] = vm_disp.ravel(order="F")
    surf["vm_abs_diff"] = vm_abs_diff.ravel(order="F")
    surf["vm_rel_diff"] = vm_rel_diff.ravel(order="F")

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
    R_inner,
    R_outer,
    H_total,
    theta_min,
    theta_max,
    x_groove,
    R_groove,
    y_groove_bottom,
    y_groove_top,
    stress_scale=1.0,
    disp_scale=1.0,
    deformation_vis_scale=0.0,
    n_theta=220,
    n_y=120,
    n_r=120,
    n_alpha=180,
    n_groove_y=80,
    n_groove_r=80,
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
        )

        if mask is None:
            mesh_dict[name] = make_structured_surface(
                xx, yy, zz, pred,
                deformation_vis_scale=deformation_vis_scale
            )
        else:
            mesh_dict[name] = make_masked_quad_surface(
                xx, yy, zz, mask, pred,
                deformation_vis_scale=deformation_vis_scale
            )

    return mesh_dict


# =========================================================
# 8. 合并所有边界面
# =========================================================
def merge_sector_surfaces(mesh_dict):
    merged = None
    for mesh in mesh_dict.values():
        if merged is None:
            merged = mesh.copy()
        else:
            merged = merged.merge(mesh, merge_points=True, tolerance=1e-9)

    merged = merged.clean(tolerance=1e-9)

    merged = merged.compute_normals(
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

    return merged


def print_vm_consistency_report(mesh_dict):
    print("\n================ VM consistency report ================")
    for name, mesh in mesh_dict.items():
        vm_net = np.asarray(mesh["vm_net"])
        vm_disp = np.asarray(mesh["vm_disp"])
        rel = np.asarray(mesh["vm_rel_diff"])
        net_i = int(np.argmax(vm_net))
        disp_i = int(np.argmax(vm_disp))
        print(
            f"{name:14s} | "
            f"max(vm_net)={vm_net[net_i]:.6e} Pa | "
            f"max(vm_disp)={vm_disp[disp_i]:.6e} Pa | "
            f"mean(rel_diff)={np.mean(rel):.3e} | "
            f"p95(rel_diff)={np.percentile(rel, 95):.3e}"
        )


# =========================================================
# 9. 公共绘图场景
# =========================================================
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
        lighting=False,
        specular=0.0,
        diffuse=0.0,
        ambient=1.0,
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


# =========================================================
# 10. 显示单个物理量
# =========================================================
def plot_sector_field_pyvista(
    mesh_dict,
    field_name="vm",
    title="Von Mises Stress",
    cmap="jet",
    clim=None,
    show_edges=False,
    smooth_shading=False,
    parallel_projection=True,
    background="white",
    screenshot=None,
    show_window=True,
):
    merged_mesh = merge_sector_surfaces(mesh_dict)

    all_vals = merged_mesh[field_name]
    if clim is None:
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        clim = [vmin, vmax]

    save_path = None
    if screenshot is not None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            screenshot
        )

        # 离屏保存
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

        if os.path.exists(save_path):
            print(f"截图已成功保存到: {save_path}")
        else:
            print(f"截图保存失败，目标路径: {save_path}")

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


# =========================================================
# 11. 同时画：总位移 + 等效应力
# =========================================================
def plot_sector_disp_and_vm_pyvista(
    model,
    R_inner,
    R_outer,
    H_total,
    theta_min,
    theta_max,
    x_groove,
    R_groove,
    y_groove_bottom,
    y_groove_top,
    stress_scale=1.0,
    disp_scale=1.0,
    deformation_vis_scale=0.0,
    n_theta=220,
    n_y=120,
    n_r=120,
    n_alpha=180,
    n_groove_y=80,
    n_groove_r=80,
):
    mesh_dict = build_sector_surfaces_pyvista(
        model=model,
        R_inner=R_inner,
        R_outer=R_outer,
        H_total=H_total,
        theta_min=theta_min,
        theta_max=theta_max,
        x_groove=x_groove,
        R_groove=R_groove,
        y_groove_bottom=y_groove_bottom,
        y_groove_top=y_groove_top,
        stress_scale=stress_scale,
        disp_scale=disp_scale,
        deformation_vis_scale=deformation_vis_scale,
        n_theta=n_theta,
        n_y=n_y,
        n_r=n_r,
        n_alpha=n_alpha,
        n_groove_y=n_groove_y,
        n_groove_r=n_groove_r,
    )

    print_vm_consistency_report(mesh_dict)

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="utotal",
        title="Total Deformation (Sector + Groove, PyVista)",
        screenshot="sector_total_deformation_pyvista.png",
        show_window=True
    )

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="vm",
        title="Von Mises Stress from Network Stress Head",
        screenshot="sector_von_mises_pyvista.png",
        show_window=True
    )

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="vm_disp",
        title="Von Mises Stress from Displacement Gradients",
        screenshot="sector_von_mises_from_disp_pyvista.png",
        show_window=True
    )

    plot_sector_field_pyvista(
        mesh_dict=mesh_dict,
        field_name="vm_rel_diff",
        title="Relative Difference: Network VM vs Displacement-Gradient VM",
        cmap="viridis",
        screenshot="sector_von_mises_relative_difference.png",
        show_window=True
    )

    mesh_bottom = mesh_dict["bottom"]

    plot_sector_field_pyvista(
        mesh_dict={"bottom": mesh_bottom},
        field_name="v",
        title="Bottom Surface Y-Displacement",
        screenshot="bottom_v.png",
        show_window=True
    )
    plot_sector_field_pyvista(
    mesh_dict={"bottom": mesh_bottom},
    field_name="utotal",
    title="Bottom Surface Total Displacement",
    screenshot="bottom_utotal.png",
    show_window=True
    )
# =========================================================
# 12. 运行示例
# =========================================================
if __name__ == "__main__":
    # 如果你的 PINN 训练输出已经是物理量（Pa、m），就用 1.0
    # 如果你训练时做过额外缩放，再按你的实际缩放恢复
    stress_scale = 1.0
    disp_scale = 1.0

    # deformation_vis_scale = 0 表示只看真实几何上的场分布
    # deformation_vis_scale = 1 表示按真实位移叠加变形
    # 若想夸张显示可设 10 / 50 / 100，但那不再是真实变形比例
    plot_sector_disp_and_vm_pyvista(
        model=model,
        R_inner=R_inner,
        R_outer=R_outer,
        H_total=H_total,
        theta_min=theta_min,
        theta_max=theta_max,
        x_groove=x_groove,
        R_groove=R_groove,
        y_groove_bottom=y_groove_bottom,
        y_groove_top=y_groove_top,
        stress_scale=stress_scale,
        disp_scale=disp_scale,
        deformation_vis_scale=0.0,
        n_theta=220,
        n_y=120,
        n_r=120,
        n_alpha=180,
        n_groove_y=80,
        n_groove_r=80,
    )
