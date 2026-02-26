import os
import time
import numpy as np
import cv2
import open3d as o3d
from pyorbbecsdk import *

# --- 1. 准备数据集保存总目录 ---
dataset_dir = os.path.join(os.getcwd(), "suctionnet_dataset")
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"📁 已创建数据集目录: {dataset_dir}")


def main():
    pipeline = Pipeline()
    config = Config()

    # --- 配置深度与彩色流 (保持之前的成功配置) ---
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    has_color_sensor = False
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor = True
    except OBError as e:
        pass

    pipeline.enable_frame_sync()
    pipeline.start(config)
    print("📸 相机已启动！")

    # 硬件层面对齐滤波器 (将深度对齐到彩色分辨率)
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()

    cv2.namedWindow("Live Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Preview", 960, 540)
    print("\n💡 操作提示：[空格键] -> 抓取 SuctionNet 全套数据 | [ESC/Q] -> 退出\n")

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None: continue
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame() if has_color_sensor else None
        if depth_frame is None or (has_color_sensor and color_frame is None): continue

        # --- OpenCV 实时预览 ---
        raw_data = color_frame.get_data()
        color_data = np.frombuffer(raw_data, dtype=np.uint8)
        w, h = color_frame.get_width(), color_frame.get_height()

        if color_profile.get_format() == OBFormat.MJPG:
            img_preview = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
        else:
            img_preview = np.ascontiguousarray(color_data.reshape((h, w, 3)))
            img_preview = cv2.cvtColor(img_preview, cv2.COLOR_RGB2BGR)

        cv2.imshow("Live Preview", img_preview)

        # --- 监听按键 ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32:  # 空格键保存全家桶
            print("\n⏳ 正在生成 SuctionNet 算法所需数据集...")

            # 1. 创建独立的时间戳文件夹
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(dataset_dir, f"scene_{timestamp}")
            os.makedirs(save_path)

            # 2. 对齐深度和彩色帧
            aligned_frames = align_filter.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()

            # =============== [核心修改：保存全套数据] ===============

            # [A] 保存彩色图 (RGB)
            color_path = os.path.join(save_path, "color.png")
            cv2.imwrite(color_path, img_preview)

            # [B] 保存对齐后的深度图 (16-bit PNG，单位：毫米)
            depth_path = os.path.join(save_path, "depth.png")
            depth_raw = np.frombuffer(aligned_depth_frame.get_data(), dtype=np.uint16)
            depth_img = depth_raw.reshape((aligned_depth_frame.get_height(), aligned_depth_frame.get_width()))
            cv2.imwrite(depth_path, depth_img)

            # [C] 保存相机内参矩阵 (Intrinsics)
            # 因为我们把深度对齐到了彩色流，所以这里提取彩色的内参
            camera_param = pipeline.get_camera_param()
            color_int = camera_param.rgb_intrinsic
            intrinsics_matrix = np.array([
                [color_int.fx, 0, color_int.cx],
                [0, color_int.fy, color_int.cy],
                [0, 0, 1]
            ])
            intrinsics_path = os.path.join(save_path, "intrinsics.npy")
            np.save(intrinsics_path, intrinsics_matrix)

            # [D] 顺手保存点云，方便人工核对
            point_format = OBFormat.RGB_POINT if has_color_sensor else OBFormat.POINT
            point_cloud_filter.set_create_point_format(point_format)
            point_cloud_frame = point_cloud_filter.process(aligned_frames)
            ply_path = os.path.join(save_path, "pointcloud.ply")
            save_point_cloud_to_ply(ply_path, point_cloud_frame)

            print(f"✅ 数据抓取成功！已保存至: {save_path}")
            print(f"  ├── color.png (彩色图)")
            print(f"  ├── depth.png (16位深度图)")
            print(f"  ├── intrinsics.npy (相机内参)")
            print(f"  └── pointcloud.ply (参考点云)")

            # 立即弹窗核对点云
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                pcd.transform(flip_transform)
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, axes], window_name="数据核对 - 按ESC继续拍摄")
            except Exception as e:
                pass

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()