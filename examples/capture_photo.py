import cv2
import numpy as np
import os
import time
from pyorbbecsdk import *

def main():
    # --- 1. 准备文件夹 ---
    save_path = "capture_photos"  # 文件夹名称
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 已创建文件夹: {save_path}")

    pipeline = Pipeline()
    config = Config()

    try:
        # --- 2. 配置彩色流 ---
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # 1080P 在 Femto Bolt 上通常是 MJPG 格式
            color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
        except:
            color_profile = profile_list.get_default_video_stream_profile()
        
        config.enable_stream(color_profile)
        pipeline.start(config)
        print(f"📸 相机已启动: {color_profile.get_width()}x{color_profile.get_height()}")

        # --- 3. 等待画面稳定 (跳过前 30 帧) ---
        for _ in range(30):
            pipeline.wait_for_frames(100)

        # --- 4. 获取并处理图像 ---
        frames = pipeline.wait_for_frames(1000)
        if frames:
            color_frame = frames.get_color_frame()
            if color_frame:
                # 获取原始数据并解决内存不连续问题
                raw_buffer = color_frame.get_data()
                tmp_arr = np.frombuffer(raw_buffer, dtype=np.uint8)
                # 强制转换为 C 连续内存，彻底解决 ndarray 报错
                final_data = np.array(tmp_arr, copy=True, order='C')
                
                # 解码图片
                img = cv2.imdecode(final_data, cv2.IMREAD_COLOR)

                if img is not None:
                    # 使用时间戳生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"bolt_{timestamp}.jpg"
                    # 组合完整路径
                    full_save_path = os.path.join(save_path, filename)
                    
                    cv2.imwrite(full_save_path, img)
                    print(f"✅ 照片已保存至: {full_save_path}")
                else:
                    print("❌ 图像解码失败")
        else:
            print("❌ 获取帧超时")

    except Exception as e:
        print(f"❌ 运行报错: {e}")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()