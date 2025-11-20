import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 推荐加 CAP_DSHOW

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)  # 显示摄像头画面

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
