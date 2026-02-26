import os
import time
import numpy as np
import cv2
import open3d as o3d
from pyorbbecsdk import *

# --- 1. 准备保存目录 ---
save_points_dir = os.path.join(os.getcwd(), "point_clouds")
if not os.path.exists(save_points_dir):
    os.makedirs(save_points_dir)
    print(f"📁 已创建保存目录: {save_points_dir}")


def main():
    pipeline = Pipeline()
    config = Config()

    # --- 2. 配置深度流 ---
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is None:
        print("❌ 找不到深度传感器配置，无法生成点云")
        return
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    has_color_sensor = False
    try:
        # --- 3. 配置彩色流 ---
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            # 默认配置通常能满足大部分需求
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor = True
    except OBError as e:
        print(f"⚠️ 彩色传感器配置警告: {e}")

    # 开启帧同步并启动相机
    pipeline.enable_frame_sync()
    pipeline.start(config)
    print("📸 相机已启动！")

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()

    # --- 4. 初始化预览窗口 ---
    cv2.namedWindow("Live Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Preview", 960, 540)
    print("\n💡 操作提示：")
    print("  [空格键] -> 抓取、保存并查看 3D 点云")
    print("  [ESC/Q] -> 退出程序\n")

    while True:
        # 实时获取数据流
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        color_frame = frames.get_color_frame()
        if has_color_sensor and color_frame is None:
            continue

        # --- 5. 在 OpenCV 中实时预览彩色画面 ---
        if has_color_sensor and color_frame is not None:
            raw_data = color_frame.get_data()
            color_data = np.frombuffer(raw_data, dtype=np.uint8)

            # 根据格式解码图像以便预览
            if color_profile.get_format() == OBFormat.MJPG:
                img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            else:
                w, h = color_frame.get_width(), color_frame.get_height()
                img = np.ascontiguousarray(color_data.reshape((h, w, 3)))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if img is not None:
                cv2.imshow("Live Preview", img)

        # --- 6. 监听键盘按键 ---
        key = cv2.waitKey(1) & 0xFF

        # 按下 ESC 或 Q 退出
        if key == 27 or key == ord('q'):
            break

        # 按下 空格键 (Space) 保存并查看点云
        elif key == 32:
            print("⏳ 正在生成并对齐点云数据...")

            # 硬件级对齐深度和彩色帧
            aligned_frames = align_filter.process(frames)

            # 设置点云格式 (有彩色传感器就生成带颜色的点云)
            point_format = OBFormat.RGB_POINT if has_color_sensor else OBFormat.POINT
            point_cloud_filter.set_create_point_format(point_format)

            # 生成 3D 点云
            point_cloud_frame = point_cloud_filter.process(aligned_frames)
            if point_cloud_frame is None:
                print("❌ 生成点云失败，请重试")
                continue

            # 使用时间戳生成唯一文件名，防止覆盖
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ply_filename = os.path.join(save_points_dir, f"point_cloud_{timestamp}.ply")

            # 保存 PLY 文件 (保持你原有的调用格式)
            save_point_cloud_to_ply(ply_filename, point_cloud_frame)
            print(f"✅ 点云已保存: {ply_filename}")

            # 立即使用 Open3D 读取并弹窗显示刚刚保存的点云
            print("👀 正在打开 3D 视图... (在 3D 窗口中按 ESC 即可关闭并继续拍摄)")
            try:
                # 读取刚刚保存的点云
                pcd = o3d.io.read_point_cloud(ply_filename)

                # ==========================================
                # 【坐标系翻转修正】
                # 将相机的 OpenCV 坐标系 (Y轴向下) 转换为 Open3D 坐标系 (Y轴向上)
                # 通过矩阵变换翻转 Y 轴和 Z 轴
                # ==========================================
                flip_transform = [[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 0, 0, 1]]
                pcd.transform(flip_transform)

                # 创建坐标轴辅助显示 (红X, 绿Y, 蓝Z)
                # size 调大了点，方便在大场景里看清坐标系原点
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, axes], window_name=f"3D Viewer - {timestamp}")
            except Exception as e:
                print(f"Open3D 显示错误: {e}")

    # --- 7. 安全释放资源 ---
    print("🔌 正在停止相机并清理资源...")
    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()