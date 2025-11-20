import cv2  # 导入OpenCV库

cap = cv2.VideoCapture(1)  # 打开索引为1的外接摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 设置分辨率为1920宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置分辨率为1080高
cap.set(cv2.CAP_PROP_FPS, 60)             # 尝试设为60帧

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
cv2.resizeWindow("Camera", 1280, 720)         # 设置窗口初始显示尺寸

while True:  # 主循环
    ret, frame = cap.read()  # 读取一帧画面
    if not ret:              # 如果失败则退出
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将BGR颜色空间转换为灰度图（单通道）
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # 将BGR颜色空间转换为HSV（便于做颜色阈值）

    # 为了把灰度图与HSV图并排显示，需要都转为三通道（否则hconcat会因通道数不一致报错）
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 灰度单通道转为BGR三通道
    hsv_bgr  = cv2.cvtColor(hsv,  cv2.COLOR_HSV2BGR)   # HSV再转回BGR用于可视化对比

    # 将三幅图像在水平方向拼接（原图 | 灰度 | HSV可视化）
    concat = cv2.hconcat([frame, gray_bgr, hsv_bgr])  # hconcat要求高度一致、通道一致

    # 显示拼接后的对比画面
    cv2.imshow("Camera", concat)

    # 按下'q'退出；waitKey(1)表示每1ms轮询键盘一次     & 提取按键值的低8位
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()              # 释放摄像头
cv2.destroyAllWindows()    # 关闭窗口
