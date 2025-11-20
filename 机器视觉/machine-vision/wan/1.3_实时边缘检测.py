import cv2  # 导入OpenCV库

cap = cv2.VideoCapture(1)  # 打开索引为1的外接摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 设置分辨率为1920×1080
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置分辨率为1920×1080
cap.set(cv2.CAP_PROP_FPS, 60)             # 尝试设定帧率为60

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
cv2.resizeWindow("Camera", 1280, 720)         # 将窗口调整为1280×720显示

while True:  # 主循环
    ret, frame = cap.read()  # 从摄像头读取一帧
    if not ret:              # 若失败则退出
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # 转灰度（边缘检测通常在灰度图上进行）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)             # 使用高斯模糊降噪（减少伪边缘）
    edges = cv2.Canny(blur, 50, 150)                     # Canny边缘检测（低阈值50，高阈值150）

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 将单通道边缘图转为三通道便于拼接
    concat = cv2.hconcat([frame, edges_bgr])             # 将原图与边缘图水平拼接显示

    cv2.imshow("Camera", concat)                         # 显示并排画面

    if cv2.waitKey(1) & 0xFF == ord('q'):               # 按'q'退出
        break

cap.release()              # 释放摄像头资源
cv2.destroyAllWindows()    # 关闭所有窗口
