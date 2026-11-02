"""Orbbec 多视角扫描采集 —— 工件三维建模第一步。

用途：
    绕着工件（或转动工件）从多个视角各拍一帧 RGB-D，逐帧保存为
    带颜色的 PLY 点云，供 reconstruct.py 做 ICP 配准 + Poisson 重建。

    - 转动工件：工件放转台上自转，相机固定。
    - 转动相机：工件静止，相机绕其一周（每 15~30° 拍一帧）。

用法：
    conda activate bin_grasp
    python capture_scan.py --out scan_my_part

按键（OpenCV 预览窗口）：
    空格     抓取当前帧并保存
    r        把采集模式在「转台/手持」间切换（决定配准时用绕 z 旋转的初值）
    q / ESC  结束采集
    s        结束并打印已采集帧清单

设计取舍：
    1. 深度对齐到彩色流（AlignFilter -> COLOR），内参取彩色内参，这样
       颜色图、深度图、点云三者在同一像素坐标系下，配准与后处理一致。
    2. 点云格式 RGB_POINT，直接存 PLY（含颜色，供重建时降采样/可视化）。
    3. 每帧同时落一份 color.png / depth.png / intrinsics.npy，便于离线
       复查、用别的 SfM/MVS 工具二次处理，不依赖 Orbbec SDK 在场。
    4. 相机坐标系转 Open3D 惯用系（翻转 Y/Z），与 capture_pointcloud.py 一致。
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBError,
    OBFormat,
    OBSensorType,
    OBStreamType,
    Pipeline,
    PointCloudFilter,
    save_point_cloud_to_ply,
)


def _flip_transform() -> np.ndarray:
    """OpenCV 相机系 -> Open3D 惯用系（Y 向上、Z 朝观察者）。"""
    return np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _color_to_bgr(color_frame, color_profile) -> np.ndarray | None:
    """把彩色帧解码成 BGR ndarray 供预览 / 落盘。"""
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    if color_profile.get_format() == OBFormat.MJPG:
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    w, h = color_frame.get_width(), color_frame.get_height()
    img = np.ascontiguousarray(raw.reshape((h, w, 3)))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Orbbec multi-view workpiece scan")
    ap.add_argument("--out", default="scan", help="输出目录名（默认 scan）")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    pipeline = Pipeline()
    config = Config()

    # 深度流
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is None:
        print("找不到深度传感器，无法扫描")
        return
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    # 彩色流（可选，但强烈建议有——带色点云对配准/分割友好）
    has_color = False
    color_profile = None
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color = True
    except OBError as e:
        print(f"彩色传感器不可用: {e}")

    pipeline.enable_frame_sync()
    pipeline.start(config)
    print("相机已启动")

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()
    if has_color:
        point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
    else:
        point_cloud_filter.set_create_point_format(OBFormat.POINT)

    cv2.namedWindow("Scan Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scan Preview", 960, 540)

    turntable = True  # 默认假设转台（绕 z 旋转）
    frames = []  # 已采集的 (subdir, ply_path)
    flip = _flip_transform()

    print("\n操作：空格=抓取 | r=切换转台/手持 | s=结束并列清单 | q=退出")

    while True:
        fr = pipeline.wait_for_frames(100)
        if fr is None:
            continue
        depth_frame = fr.get_depth_frame()
        color_frame = fr.get_color_frame() if has_color else None
        if depth_frame is None or (has_color and color_frame is None):
            continue

        if has_color:
            img = _color_to_bgr(color_frame, color_profile)
            if img is not None:
                cv2.imshow("Scan Preview", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r"):
            turntable = not turntable
            print(f"采集模式 -> {'转台(绕z旋转)' if turntable else '手持(自由视角)'}")
        elif key in (ord("s"), 32):
            # 空格 = 抓帧；s = 结束并抓最后一帧（同一路径）
            aligned = align_filter.process(fr)
            pc_frame = point_cloud_filter.process(aligned)
            if pc_frame is None:
                print("生成点云失败，跳过")
                continue

            sub = time.strftime("frame_%Y%m%d_%H%M%S_") + f"{len(frames):03d}"
            subdir = os.path.join(out_root, sub)
            os.makedirs(subdir, exist_ok=True)

            ply = os.path.join(subdir, "pointcloud.ply")
            save_point_cloud_to_ply(ply, pc_frame)

            # 落 RGB / depth / intrinsics（供离线复查）
            if has_color and img is not None:
                cv2.imwrite(os.path.join(subdir, "color.png"), img)
            aligned_depth = aligned.get_depth_frame()
            d_raw = np.frombuffer(aligned_depth.get_data(), dtype=np.uint16)
            d_img = d_raw.reshape(
                (aligned_depth.get_height(), aligned_depth.get_width())
            )
            cv2.imwrite(os.path.join(subdir, "depth.png"), d_img)

            cam = pipeline.get_camera_param()
            ci = cam.rgb_intrinsic if has_color else cam.depth_intrinsic
            K = np.array([[ci.fx, 0, ci.cx], [0, ci.fy, ci.cy], [0, 0, 1]])
            np.save(os.path.join(subdir, "intrinsics.npy"), K)

            frames.append((sub, ply))
            print(f"[{len(frames):02d}] 已保存 {sub}")

            # 简单核对：读取点云，翻转坐标系，统计点数
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(ply)
                pcd.transform(flip)
                print(f"      点数 {len(pcd.points)}")
            except Exception:
                pass

            if key == ord("s"):
                break

    cv2.destroyAllWindows()
    pipeline.stop()

    print("\n=== 采集完成 ===")
    print(f"目录: {out_root}")
    print(f"模式: {'转台(绕z旋转)' if turntable else '手持(自由视角)'}")
    print(f"帧数: {len(frames)}")
    for i, (sub, ply) in enumerate(frames):
        print(f"  {i}: {sub}")
    print(f"\n下一步: python reconstruct.py --scan {out_root} "
          f"{'--turntable' if turntable else ''}")


if __name__ == "__main__":
    main()
