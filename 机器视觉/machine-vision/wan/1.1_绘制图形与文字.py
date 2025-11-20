import cv2  # 导入OpenCV库

cap = cv2.VideoCapture(1)  # 打开索引为1的外接摄像头（你的外置镜头通常是1）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 设置摄像头采集宽度为1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置摄像头采集高度为1080
cap.set(cv2.CAP_PROP_FPS, 60)             # 尝试将帧率设置为60（是否生效取决于硬件/驱动）

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
cv2.resizeWindow("Camera", 1280, 720)         # 将窗口初始尺寸设置为1280×720，便于显示

while True:  # 主循环，逐帧读取并显示
    ret, frame = cap.read()  # 从摄像头读取一帧，ret为读取是否成功，frame为图像
    if not ret:              # 如果读取失败（例如摄像头被拔出），就跳出循环
        break

    # 在画面左上角画一个绿色矩形（x1=100,y1=100 到 x2=300,y2=300，线宽=2）
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # 在画面中心画一个蓝色圆（中心坐标(960,540)大约为1080p的中心，半径=60，线宽=2）
    cv2.circle(frame, (960, 540), 60, (255, 0, 0), 2)

    # 在左上角写一行黄色文字，字号=1，线宽=2
    cv2.putText(frame, "SAO-Lite UI Overlay", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 将带有叠加图形与文字的画面显示到窗口
    cv2.imshow("Camera", frame)

    # 轮询键盘事件1毫秒；如果按下'q'键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()              # 释放摄像头资源
cv2.destroyAllWindows()    # 销毁所有OpenCV创建的窗口
