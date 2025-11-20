import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

dog_nose = cv2.imread('dog_nose.png')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 图1
    yuantu = frame.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(yuantu, (x, y), (x+w, y+h), (255, 0, 0), 2)


    #Alpha融合
    frame_alpha = frame.copy()
    for (x, y, w, h) in faces:
        nose_x = x + w//3
        nose_y = y + h//2
        nose_w = w//3
        nose_h = h//4

        sticker = cv2.resize(dog_nose, (nose_w, nose_h))

        # Alpha融合
        roi = frame_alpha[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w]
        if roi.shape[:2] == sticker.shape[:2]:
            blended = cv2.addWeighted(roi, 0.5, sticker, 0.5, 0)
            frame_alpha[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w] = blended

    #金字塔融合
    frame_pyramid = frame.copy()
    for (x, y, w, h) in faces:
        nose_x = x + w//3
        nose_y = y + h//2
        nose_w = w//3
        nose_h = h//4

        roi = frame_pyramid[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w]
        sticker = cv2.resize(dog_nose, (nose_w, nose_h))

        if roi.shape[:2] == sticker.shape[:2]:
            pyr1 = cv2.pyrDown(roi)
            pyr2 = cv2.pyrDown(sticker)
            blended_pyr = cv2.addWeighted(pyr1, 0.5, pyr2, 0.5, 0)
            blended = cv2.pyrUp(blended_pyr)


            blended = cv2.resize(blended, (nose_w, nose_h))
            frame_pyramid[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w] = blended

    #泊松融合
    frame_poisson = frame.copy()
    for (x, y, w, h) in faces:
        nose_x = x + w//3
        nose_y = y + h//2
        nose_w = w//3
        nose_h = h//4

        sticker = cv2.resize(dog_nose, (nose_w, nose_h))
        mask = 255 * np.ones(sticker.shape, sticker.dtype)

        center = (nose_x + nose_w//2, nose_y + nose_h//2)

        frame_poisson = cv2.seamlessClone(sticker, frame_poisson, mask, center, cv2.NORMAL_CLONE)


    top = np.hstack([yuantu, frame_alpha])
    bottom = np.hstack([frame_pyramid, frame_poisson])
    final = np.vstack([top, bottom])

    cv2.imshow('final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()