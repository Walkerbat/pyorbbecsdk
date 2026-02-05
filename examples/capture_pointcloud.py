import os
import time
import numpy as np
from pyorbbecsdk import *

def save_points_to_ply(vertices, filename):
    """将点云数据保存为 PLY 格式"""
    if vertices is None or len(vertices) == 0:
        return False
    
    # 过滤掉坐标全为 0 的无效点
    valid_points = vertices[np.any(vertices != 0, axis=1)]
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(valid_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in valid_points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    return True

def main():
    # 1. 准备文件夹
    save_path = "capture_pointclouds"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 已创建点云文件夹: {save_path}")

    pipeline = Pipeline()
    config = Config()

    try:
        # 2. 配置深度流
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        
        # 3. 启动 Pipeline
        pipeline.start(config)
        print(f"✅ 相机已启动，正在生成点云...")

        # 4. 创建点云过滤器
        pc_filter = PointCloudFilter()
        # 获取相机外参（用于点云坐标转换）
        camera_param = pipeline.get_camera_param()
        pc_filter.set_camera_param(camera_param)

        # 5. 等待并获取一帧
        frames = pipeline.wait_for_frames(1000)
        if frames:
            # 计算点云
            pc_filter.set_point_cloud_type(OBPointValueType.XYZ)
            frame_set = pc_filter.calculate_pointcloud(frames)
            
            if frame_set:
                # 获取点云位置数据
                points_data = frame_set.get_data()
                
                # --- 解决内存不连续问题 ---
                # 将原始 buffer 转为 numpy 数组并强制深拷贝
                tmp_arr = np.frombuffer(points_data, dtype=np.float32)
                # 每三个值代表一个 (x, y, z) 坐标
                vertices = np.array(tmp_arr, copy=True, order='C').reshape(-1, 3)
                
                # 保存文件
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(save_path, f"pc_{timestamp}.ply")
                
                if save_points_to_ply(vertices, filename):
                    print(f"🎉 点云保存成功: {filename}")
                    print(f"💡 提示：该文件包含 {len(vertices)} 个点，可使用 MeshLab 查看。")
                else:
                    print("❌ 无效的点云数据")
        else:
            print("❌ 获取帧超时")

    except Exception as e:
        print(f"❌ 运行报错: {e}")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()