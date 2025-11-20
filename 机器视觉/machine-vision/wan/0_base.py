import cv2

# 打开默认摄像头，0是默认摄像头索引
cap = cv2.VideoCapture(1)

# 先设置分辨率（某些摄像头需要先设分辨率再设帧率）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 设置宽度为1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置高度为1080

# 然后设置帧率为60 FPS
cap.set(cv2.CAP_PROP_FPS, 60)

# 创建可调整大小的窗口
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
# 设置窗口大小为屏幕适合的尺寸（例如1280x720）
cv2.resizeWindow("Camera", 1280, 720)


while True:
    ret, frame = cap.read()    # 读取一帧图像，ret表示读取是否成功，frame是读取到的图像帧
    if not ret:      # 如果读取失败（如摄像头断开），则退出循环
        break

    # 显示画面
    cv2.imshow("Camera", frame)

    # 等待1毫秒并检测键盘输入，如果按下q键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源，让其他程序可以使用
cap.release()
# 关闭所有由OpenCV创建的窗口
cv2.destroyAllWindows()
