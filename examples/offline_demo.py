"""离线验证 —— 无需相机，验证 reconstruct.py 的重建链路是否可用。

用程序生成一个「扁平盒状工件」的合成点云，模拟转台绕 z 轴旋转拍的多帧，
再走一遍 ICP 配准 + Poisson 重建 + STL 导出。跑通说明 open3d 环境和
reconstruct.py 的逻辑没问题，之后再插真相机扫真件。

用法：
    python offline_demo.py
    python reconstruct.py --scan offline_demo_scan --turntable --voxel 0.002
"""
from __future__ import annotations

import os
import time

import numpy as np
import open3d as o3d


def make_box_pcd(hx=0.060, hy=0.040, hz=0.015, n_per_face=800):
    """生成一个盒子的密集表面点云（无顶面，模拟转台扫描看到的样子）。"""
    pts = []
    cols = []
    rng = np.random.default_rng(0)

    def add_face(samples, face_normal, color):
        # samples: (N,3) 落在某个平面上的点
        p = samples
        pts.append(p)
        cols.append(np.tile(color, (len(p), 1)))

    # 底面 z=-hz
    xs = rng.uniform(-hx, hx, n_per_face)
    ys = rng.uniform(-hy, hy, n_per_face)
    add_face(np.column_stack([xs, ys, np.full(n_per_face, -hz)]), (0, 0, -1), (0.7, 0.7, 0.7))
    # 顶面 z=+hz
    xs = rng.uniform(-hx, hx, n_per_face)
    ys = rng.uniform(-hy, hy, n_per_face)
    add_face(np.column_stack([xs, ys, np.full(n_per_face, hz)]), (0, 0, 1), (0.7, 0.7, 0.7))
    # 四个侧面（法向分别沿 +y / -y / +x / -x）
    for sign, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
        n = n_per_face
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        if axis == 0:  # 法向沿 y：x、z 变化，y 固定
            x = u * hx
            y = np.full(n, sign * hy)
            z = v * hz
        else:  # 法向沿 x：y、z 变化，x 固定
            x = np.full(n, sign * hx)
            y = u * hy
            z = v * hz
        pts.append(np.column_stack([x, y, z]))
        cols.append(np.tile([0.55, 0.55, 0.55], (n, 1)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(cols))
    return pcd


def main():
    out = "offline_demo_scan"
    n_views = 12  # 模拟 12 个转台视角

    for i in range(n_views):
        sub = os.path.join(out, f"frame_0000_{i:03d}")
        os.makedirs(sub, exist_ok=True)
        pcd = make_box_pcd()
        # 绕 z 轴旋转 i/n*360 度（转台）
        theta = np.deg2rad(360.0 * i / n_views)
        R = pcd.get_rotation_matrix_from_xyz((0, 0, theta))
        pcd.rotate(R, center=(0, 0, 0))
        o3d.io.write_point_cloud(os.path.join(sub, "pointcloud.ply"), pcd)
        print(f"生成 {sub}")

    print(f"\n已生成 {n_views} 帧合成扫描 -> {out}/")
    print("现在运行: python reconstruct.py --scan offline_demo_scan --turntable --voxel 0.002")


if __name__ == "__main__":
    main()
