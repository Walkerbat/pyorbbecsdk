"""点云配准 + 重建 + STL 导出 —— 工件三维建模第二步。

输入：capture_scan.py 采集的扫描目录（每帧一个 pointcloud.ply）。
输出：
    fused.ply        多帧配准融合后的完整点云（去噪、体素降采样）
    mesh.ply / mesh.stl   Poisson 重建的三角网格（STL 可直接进 MuJoCo）
    dims.json        重建件的 AABB 尺寸 / 体积 / 中心，供 scene.xml 填参数

配准策略：
    - 转台（--turntable）：相邻帧之间近似绕 z 轴旋转一个固定角，用
      RANSAC+ICP（point-to-plane）配准；转台帧间重叠大、初值好，最稳。
    - 手持（默认）：首帧作参考，后续帧用 ICP 逐帧对齐（点云有足够重叠
      时收敛良好）。
    - --with-color：配准时用彩色 ICP（对低纹理工件更稳），否则用几何 ICP。

重建：
    - 融合点云 -> 体素降采样 -> 统计离群点去除（可选）-> 估计法向
      -> Poisson 重建 -> 按密度裁剪漂浮面片 -> 保存 mesh。
    - 转台工件的底面无扫描数据（转台遮挡），Poisson 会自动封口，生成
      一个近似平面底；若要精确底面，用 --crop-bottom 按 AABB 底部裁剪。

用法：
    python reconstruct.py --scan scan_my_part [--turntable] [--with-color]
                          [--voxel 0.0015] [--crop-bottom]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import open3d as o3d


def _flip_transform() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def load_frames(scan_dir: str) -> list[o3d.geometry.PointCloud]:
    ply_files = sorted(
        os.path.join(scan_dir, d, "pointcloud.ply")
        for d in os.listdir(scan_dir)
        if os.path.isdir(os.path.join(scan_dir, d))
        and os.path.exists(os.path.join(scan_dir, d, "pointcloud.ply"))
    )
    if not ply_files:
        raise SystemExit(f"在 {scan_dir} 下没找到任何 pointcloud.ply，先跑 capture_scan.py")

    flip = _flip_transform()
    pcds = []
    for p in ply_files:
        pc = o3d.io.read_point_cloud(p)
        pc.transform(flip)
        pc.remove_non_finite_points()
        pcds.append(pc)
        print(f"载入 {os.path.basename(os.path.dirname(p))}: {len(pc.points)} 点")
    return pcds


def _preprocess(pc, voxel):
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)
    )
    return pc


def icp_align(source, target, voxel, with_color, init=None):
    """把 source 对齐到 target，返回 (变换后的 source, 配准矩阵, fitness)。"""
    src = _preprocess(source, voxel)
    tgt = _preprocess(target, voxel)

    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if with_color and src.has_colors() and tgt.has_colors():
        est = o3d.pipelines.registration.TransformationEstimationForColoredICP()

    if init is not None:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, voxel * 8, init,
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
    else:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, voxel * 8, np.eye(4),
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

    src.transform(reg.transformation)
    return src, reg.transformation, reg.fitness


def turntable_init(i: int, n: int) -> np.ndarray:
    """第 i 帧相对第 0 帧绕 z 轴旋转 i/n*360 度的刚体变换（转台初值）。"""
    if i == 0:
        return np.eye(4)
    theta = np.deg2rad(360.0 * i / n)
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0],
         [np.sin(theta), np.cos(theta), 0],
         [0, 0, 1]]
    )
    T = np.eye(4)
    T[:3, :3] = R
    return T


def main():
    ap = argparse.ArgumentParser(description="ICP 配准 + Poisson 重建 + STL 导出")
    ap.add_argument("--scan", required=True, help="扫描目录")
    ap.add_argument("--turntable", action="store_true", help="转台采集，用绕 z 旋转初值")
    ap.add_argument("--with-color", action="store_true", help="彩色 ICP")
    ap.add_argument("--voxel", type=float, default=0.0015, help="体素降采样边长(m)")
    ap.add_argument("--no-outlier", action="store_true", help="跳过统计离群点去除")
    ap.add_argument("--crop-bottom", type=float, default=None,
                    help="按 AABB 底部 z 裁剪掉转台平面（如 0.005，单位 m）")
    args = ap.parse_args()

    pcds = load_frames(args.scan)
    n = len(pcds)
    print(f"\n共 {n} 帧，开始配准 ...")

    fused = o3d.geometry.PointCloud()
    reference = pcds[0]

    for i in range(n):
        if i == 0:
            fused += pcds[0]
            print(f"[{i}] 参考帧就位")
            continue
        if args.turntable:
            init = turntable_init(i, n)
            aligned, T, fit = icp_align(pcds[i], reference, args.voxel,
                                        args.with_color, init=init)
        else:
            aligned, T, fit = icp_align(pcds[i], reference, args.voxel,
                                        args.with_color)
        fused += aligned
        print(f"[{i}] 对齐到参考帧: fitness={fit:.3f}")

    print(f"融合点云: {len(fused.points)} 点")

    # 去噪 + 降采样
    fused = fused.voxel_down_sample(args.voxel)
    if not args.no_outlier:
        fused, _ = fused.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    print(f"降采样+去噪后: {len(fused.points)} 点")

    if args.crop_bottom is not None:
        pts = np.asarray(fused.points)
        zmin = pts[:, 2].min()
        keep = pts[:, 2] > zmin + args.crop_bottom
        fused = fused.select_by_index(np.where(keep)[0])
        print(f"裁剪底部后: {len(fused.points)} 点")

    fused_path = os.path.join(args.scan, "fused.ply")
    o3d.io.write_point_cloud(fused_path, fused)
    print(f"融合点云已保存: {fused_path}")

    # 重建
    print("估计法向 ...")
    fused.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 5, max_nn=30)
    )
    fused.orient_normals_consistent_tangent_plane(30)

    print("Poisson 重建 ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        fused, depth=9
    )
    # 按密度裁剪漂浮面片：保留密度前 95% 的顶点
    dens = np.asarray(densities)
    thresh = np.quantile(dens, 0.05)
    verts_to_remove = dens < thresh
    mesh.remove_vertices_by_mask(verts_to_remove)

    mesh_path = os.path.join(args.scan, "mesh.ply")
    stl_path = os.path.join(args.scan, "mesh.stl")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(stl_path, mesh)
    print(f"mesh 已保存: {mesh_path}")
    print(f"STL 已保存: {stl_path}")

    # 尺寸 / 体积统计
    aabb = mesh.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    try:
        vol = mesh.get_volume()
        vol_note = "watertight"
    except RuntimeError:
        # Poisson 在底面/未观测区常留缺口，网格非封闭时 get_volume 会抛错。
        # 退化为 AABB 体积作为上界估计。
        vol = float(ext[0] * ext[1] * ext[2])
        vol_note = "AABB(非封闭，取包围盒上界)"
    dims = {
        "aabb_min": aabb.get_min_bound().tolist(),
        "aabb_max": aabb.get_max_bound().tolist(),
        "extent_xyz": ext.tolist(),
        "center": aabb.get_center().tolist(),
        "volume_m3": float(vol),
        "num_vertices": len(mesh.vertices),
        "num_triangles": len(mesh.triangles),
    }
    dims_path = os.path.join(args.scan, "dims.json")
    with open(dims_path, "w", encoding="utf-8") as f:
        json.dump(dims, f, ensure_ascii=False, indent=2)
    print("\n=== 尺寸统计 ===")
    print(f"长(x) {ext[0]*1000:.1f} mm, 宽(y) {ext[1]*1000:.1f} mm, 高(z) {ext[2]*1000:.1f} mm")
    print(f"中心 {aabb.get_center()} (m)")
    print(f"体积 {vol*1e6:.1f} cm^3 ({vol_note})")
    print(f"面片 {dims['num_triangles']}")
    print(f"尺寸详情已写: {dims_path}")

    print("\n下一步：把 mesh.stl 复制到 Bin_grasp/simulation/meshes/，"
          "在 scene.xml 里加 <mesh> asset 和 <geom type=\"mesh\">，"
          "并按 dims.json 的 extent_xyz 填 mass / 半高（见 modeling_notes）。")


if __name__ == "__main__":
    main()
