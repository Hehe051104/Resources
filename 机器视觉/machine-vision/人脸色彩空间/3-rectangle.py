import cv2
import numpy as np

# 模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')
mouth_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    eye_hsv = np.zeros((50,50,3), np.uint8)
    nose_ycrcb = np.zeros((50,50,3), np.uint8)
    mouth_gray = np.zeros((50,50), np.uint8)

    if len(faces) > 0:
        (fx, fy, fw, fh) = max(faces, key=lambda r: r[2]*r[3])
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

        face_gray  = gray[fy:fy+fh, fx:fx+fw]
        face_color = frame[fy:fy+fh, fx:fx+fw]

        # 眼睛
        roi_gray_up  = face_gray[0:fh//2, :]
        roi_color_up = face_color[0:fh//2, :]
        eyes = eye_cascade.detectMultiScale(roi_gray_up, 1.1, 8, minSize=(25, 25))
        eyes = sorted(list(eyes), key=lambda r:r[2]*r[3], reverse=True)[:2]

        eye_hsv_list = []
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color_up, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            eye_bgr = roi_color_up[ey:ey+eh, ex:ex+ew]
            if eye_bgr.size > 0:
                eye_hsv_list.append(cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2HSV))
        if len(eye_hsv_list) == 1:
            eye_hsv = eye_hsv_list[0]
        elif len(eye_hsv_list) == 2:
            h = eye_hsv_list[0].shape[0]
            w = eye_hsv_list[1].shape[1] * h // eye_hsv_list[1].shape[0]
            eye2_resized = cv2.resize(eye_hsv_list[1], (w, h))
            eye_hsv = cv2.hconcat([eye_hsv_list[0], eye2_resized])

        # 鼻子
        noses = nose_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(30, 30))
        if len(noses) > 0:
            (nx, ny, nw, nh) = max(noses, key=lambda r:r[2]*r[3])
            cv2.rectangle(face_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
            nose_bgr = face_color[ny:ny+nh, nx:nx+nw]
            if nose_bgr.size > 0:
                nose_ycrcb = cv2.cvtColor(nose_bgr, cv2.COLOR_BGR2YCrCb)

        # 嘴巴
        roi_gray_down  = face_gray[fh//2:, :]
        roi_color_down = face_color[fh//2:, :]
        mouths = mouth_cascade.detectMultiScale(roi_gray_down, 1.1, 20, minSize=(40, 40))
        if len(mouths) > 0:
            (mx, my, mw, mh) = max(mouths, key=lambda r:r[2]*r[3])
            cv2.rectangle(roi_color_down, (mx, my), (mx+mw, my+mh), (255, 255, 0), 2)
            mouth_gray = roi_gray_down[my:my+mh, mx:mx+mw]

    # ===== 四图拼接窗口 =====
    h, w = 200, 200
    frame_small = cv2.resize(frame, (w, h))
    eye_hsv_small = cv2.resize(eye_hsv, (w, h))
    nose_ycrcb_small = cv2.resize(nose_ycrcb, (w, h))
    mouth_gray_color = cv2.cvtColor(mouth_gray, cv2.COLOR_GRAY2BGR)
    mouth_gray_small = cv2.resize(mouth_gray_color, (w, h))

    plc1 = cv2.hconcat([frame_small, eye_hsv_small])
    plc2 = cv2.hconcat([nose_ycrcb_small, mouth_gray_small])
    plc  = cv2.vconcat([plc1, plc2])
    cv2.imshow('result', plc)

    # ===== Joker 窗口 =====
    joker = frame.copy()
    if len(faces) > 0:
        face_j = joker[fy:fy+fh, fx:fx+fw]

        # Step 1 白底妆
        hsv = cv2.cvtColor(face_j, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = (hsv[..., 1] * 0.2).astype(np.uint8)
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.25, 0, 255).astype(np.uint8)
        face_j[:]   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        paint = np.zeros_like(face_j)

        # Step 2 黑眼圈 + 红眉
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            axes   = (int(ew * 0.7), int(eh * 0.9))
            cv2.ellipse(paint, center, axes, 0, 0, 360, (0, 0, 0), -1)
            brow_c = (center[0], max(0, ey - int(0.7 * eh)))
            brow_a = (int(ew * 0.9), int(eh * 0.6))
            cv2.ellipse(paint, brow_c, brow_a, 0, 200, 340, (0, 0, 255), 10)

        # Step 3 小红嘴
        if len(faces) > 0 and 'mouths' in locals() and len(mouths) > 0:
            (mx, my, mw, mh) = max(mouths, key=lambda r:r[2]*r[3])
            my_face = fh // 2 + my
            mc = (mx + mw // 2, my_face + mh // 2)
            axes = (int(mw * 0.4), int(mh * 0.4))  # 再缩小一点
            cv2.ellipse(paint, mc, axes, 0, 0, 360, (0, 0, 255), -1)

        # Step 4 红鼻子
        if len(noses) > 0:
            (nx, ny, nw, nh) = max(noses, key=lambda r:r[2]*r[3])
            center = (nx + nw // 2, ny + nh // 2)
            radius = int(min(nw, nh) * 0.3)
            cv2.circle(paint, center, radius, (0, 0, 255), -1)

        # Step 5 融合
        paint_blur = cv2.GaussianBlur(paint, (0, 0), 3)
        alpha = (paint_blur.max(axis=2) > 0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 2)
        alpha3 = np.dstack([alpha, alpha, alpha])
        face_j[:] = (face_j.astype(np.float32) * (1 - alpha3) +
                     paint_blur.astype(np.float32) * alpha3).astype(np.uint8)

        # Step 6 绿色头发（脸上方区域整体染绿）
        top = joker[max(0, fy- int(fh*0.6)) : fy, fx:fx+fw]  # 头发大致区域
        if top.size > 0:
            overlay = top.copy()
            overlay[:,:,1] = np.clip(overlay[:,:,1]*1.6, 0, 255)   # 提高绿色
            overlay[:,:,0] = (overlay[:,:,0]*0.6).astype(np.uint8) # 降低蓝色
            overlay[:,:,2] = (overlay[:,:,2]*0.6).astype(np.uint8) # 降低红色
            joker[max(0, fy- int(fh*0.6)) : fy, fx:fx+fw] = cv2.addWeighted(top, 0.4, overlay, 0.6, 0)

    cv2.imshow("joker", joker)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
