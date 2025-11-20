import cv2
import numpy as np

def process_video(filepath):
    # 1. 载入视频
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件。请检查路径。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            # 视频结束，循环播放或退出
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 如果想循环播放取消注释
            break
        
        # 获取画面宽高
        height, width = frame.shape[:2]
        
        # 2. 预处理：转灰度 -> 高斯模糊 -> 边缘检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150) # 阈值根据视频光照情况可能需要微调

        # 3. 检测直线 (霍夫变换)
        # minLineLength: 线段最小长度 (滤除噪点)
        # maxLineGap: 线段间允许的最大断裂距离
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=200, maxLineGap=20)

        # 初始化上下边缘的Y坐标 (默认值，防止未检测到线时报错)
        # 假设上方绿线大约在 1/3 处，下方红线大约在 2/3 处
        y_green_candidates = []
        y_red_candidates = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算斜率，只保留水平线
                if x2 - x1 == 0: continue # 防止除零
                slope = abs((y2 - y1) / (x2 - x1))
                
                # 只要非常平的线 (斜率小于 0.1)
                if slope < 0.1:
                    avg_y = (y1 + y2) // 2
                    
                    # 根据位置分类：上半部分归为绿线(天空)，下半部分归为红线(车厢)
                    # 这里的 height/2 是简单粗暴的分界线，实际可能需要根据视频调整
                    if avg_y < height / 2:
                        y_green_candidates.append(avg_y)
                    else:
                        y_red_candidates.append(avg_y)

        # 计算最终的 Y 坐标 (取平均值让线条更稳定)
        final_y_green = int(np.mean(y_green_candidates)) if y_green_candidates else int(height * 0.3)
        final_y_red = int(np.mean(y_red_candidates)) if y_red_candidates else int(height * 0.8)

        # ----------------- 3. 实时展示逻辑 -----------------
        
        # 窗口1：绘制直线
        display_frame = frame.copy()
        cv2.line(display_frame, (0, final_y_green), (width, final_y_green), (0, 255, 0), 3)
        cv2.line(display_frame, (0, final_y_red), (width, final_y_red), (0, 0, 255), 3)

        # 缩小函数
        def shrink(img, scale=0.5):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w*scale), int(h*scale)))

        # 窗口2：裁剪出的集装箱部分 (绿线和红线中间)
        if final_y_red > final_y_green:
            crop_container = frame[final_y_green:final_y_red, :]
        else:
            crop_container = frame

        # 窗口3：裁剪出的货车部分 (红线以下)
        crop_carriage = frame[final_y_red:height, :]

        # 显示窗口，全部缩小
        cv2.imshow('Window 1: Detection (Green=Sky, Red=Carriage)', shrink(display_frame))
        if crop_container.size > 0:
            cv2.imshow('Window 2: Container Crop', shrink(crop_container))
        if crop_carriage.size > 0:
            cv2.imshow('Window 3: Carriage Crop', shrink(crop_carriage))

        # 按 'q' 退出，延时 30ms (约30帧/秒)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filepath = 'video3.ts'
    process_video(filepath)