import cv2
import numpy as np
import os
import time
from pyorbbecsdk import *


def main():
    save_path = "photos"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 已创建文件夹: {save_path}")

    pipeline = Pipeline()
    config = Config()

    try:
        # --- 1. 配置并启动相机 ---
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
        except:
            color_profile = profile_list.get_default_video_stream_profile()

        config.enable_stream(color_profile)
        pipeline.start(config)
        print(f"📸 相机已启动: {color_profile.get_width()}x{color_profile.get_height()}")
        print("💡 提示：按【空格键】拍照，按【ESC】或【q】退出")

        cv2.namedWindow("Femto Bolt Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Femto Bolt Preview", 960, 540)

        # --- 2. 实时预览循环 ---
        while True:
            # 获取帧数据，设置较短的超时时间保证画面流畅
            frames = pipeline.wait_for_frames(100)
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 获取原始数据并转换为 numpy 数组
            raw_data = color_frame.get_data()
            data_ndarray = np.frombuffer(raw_data, dtype=np.uint8)

            # 格式解码
            if color_profile.get_format() == OBFormat.MJPG:
                img = cv2.imdecode(data_ndarray, cv2.IMREAD_COLOR)
            else:
                w, h = color_frame.get_width(), color_frame.get_height()
                img = data_ndarray.reshape((h, w, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 只有在解码成功时才显示画面
            if img is not None:
                cv2.imshow("Femto Bolt Preview", img)

                # 监听键盘按键 (等待 1 毫秒)
                key = cv2.waitKey(1) & 0xFF

                # 按下空格键 (ASCII 码 32) 保存照片
                if key == 32:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"bolt_{timestamp}.jpg"
                    full_save_path = os.path.join(save_path, filename)
                    cv2.imwrite(full_save_path, img)
                    print(f"✅ 照片已保存: {full_save_path}")

                # 按下 ESC (ASCII 码 27) 或 'q' 退出循环
                elif key == 27 or key == ord('q'):
                    print("退出预览...")
                    break

    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")

    finally:
        # --- 3. 释放资源 ---
        cv2.destroyAllWindows()
        pipeline.stop()
        print("🔌 相机与窗口已安全释放")


if __name__ == "__main__":
    main()