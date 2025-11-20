import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 拆分通道（BGR）并生成3通道的纯色通道图 ---
    b, g, r = cv2.split(frame)
    zeros = np.zeros_like(b)
    blue_img  = cv2.merge([b, zeros, zeros])     # 纯蓝
    green_img = cv2.merge([zeros, g, zeros])     # 纯绿
    red_img   = cv2.merge([zeros, zeros, r])     # 纯红

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 二值图是单通道，拼接前转为3通道（否则和前面三张3通道图通道数不一致会报错）
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    shang=cv2.hconcat([blue_img,green_img])
    xia=cv2.hconcat([red_img,binary_bgr])
    ct=cv2.vconcat([shang,xia])

    cv2.imshow("B-G-R + Binary(Otsu)", ct)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
