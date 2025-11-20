import cv2
import numpy as np

# 加载人脸和眼睛检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 磨皮函数（双边滤波）
def skin_smooth(img, level=15):
    return cv2.bilateralFilter(img, level, level*2, level/2)

# 大眼效果
def enlarge_eyes(img, eyes, scale=1.3):
    result = img.copy()
    h, w = img.shape[:2]

    for (ex, ey, ew, eh) in eyes:
        # 眼睛中心和半径
        center_x = ex + ew // 2
        center_y = ey + eh // 2
        radius = int(ew * 0.6)

        # 确保区域在图像范围内
        if radius < 5 or center_x < radius or center_y < radius:
            continue
        if center_x + radius >= w or center_y + radius >= h:
            continue

        # 局部放大
        for i in range(center_y - radius, center_y + radius):
            for j in range(center_x - radius, center_x + radius):
                dx = j - center_x
                dy = i - center_y
                distance = np.sqrt(dx**2 + dy**2)

                if distance < radius:
                    # 计算偏移量
                    ratio = (radius - distance) / radius
                    ratio = ratio * (1 - 1/scale)

                    offset_x = int(dx * ratio)
                    offset_y = int(dy * ratio)

                    src_x = j - offset_x
                    src_y = i - offset_y

                    if 0 <= src_x < w and 0 <= src_y < h:
                        result[i, j] = img[src_y, src_x]

    return result

# 打开摄像头
cap = cv2.VideoCapture(0)

print("按 'q' 退出程序")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图用于检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 创建副本用于处理
    original = frame.copy()
    smoothed = frame.copy()
    big_eyes = frame.copy()
    final = frame.copy()

    for (x, y, w, h) in faces:
        # 在原始图像上画人脸框和关键点
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(original, (x+w//4, y+h//3), 3, (0, 0, 255), -1)  # 左眼位置
        cv2.circle(original, (x+3*w//4, y+h//3), 3, (0, 0, 255), -1)  # 右眼位置
        cv2.circle(original, (x+w//2, y+2*h//3), 3, (0, 0, 255), -1)  # 嘴巴位置

        # 提取人脸区域
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 磨皮处理（只处理人脸区域）
        smoothed[y:y+h, x:x+w] = skin_smooth(roi_color)

        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        if len(eyes) > 0:
            # 调整眼睛坐标到原图
            eyes_adjusted = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]

            # 大眼效果（基于原图）
            big_eyes = enlarge_eyes(frame, eyes_adjusted, scale=1.25)

            # 最终效果：磨皮+大眼
            final = enlarge_eyes(smoothed, eyes_adjusted, scale=1.25)

    # 显示4个窗口
    cv2.imshow('1. Original with Keypoints', original)
    cv2.imshow('2. Skin Smoothed', smoothed)
    cv2.imshow('3. Big Eyes', big_eyes)
    cv2.imshow('4. Final (Smooth + Big Eyes)', final)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()