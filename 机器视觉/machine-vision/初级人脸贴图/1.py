import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

dog_nose_path = 'dog_nose.png'

dog_nose = cv2.imread(dog_nose_path)

if dog_nose.shape[2] == 3:
    alpha_channel = np.ones((dog_nose.shape[0], dog_nose.shape[1], 1), dtype=dog_nose.dtype) * 255
    dog_nose = np.concatenate([dog_nose, alpha_channel], axis=2)

    nose_original = dog_nose.copy()
    nose_alpha = dog_nose.copy()
    nose_gold = dog_nose.copy()
    nose_lap = dog_nose.copy()


def tietu(bg, fg, x, y, model='normal'):
    h, w = fg.shape[:2]

    if y + h > bg.shape[0] or x + w > bg.shape[1] or x < 0 or y < 0:
        return bg

    roi = bg[y:y+h, x:x+w]

    if model == 'normal':
        bg[y:y+h, x:x+w] = cv2.cvtColor(fg[:,:,:3], cv2.COLOR_RGBA2BGR)

    elif model == 'alpha':
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = alpha * fg[:, :, c] + (1 - alpha) * roi[:, :, c]

    elif model == 'pyramid':
        mask = cv2.GaussianBlur((fg[:,:,3] / 255.0).astype(np.float32), (5,5), 0)

        for c in range(3):
            blurred_fg = cv2.GaussianBlur(fg[:,:,c].astype(np.float32), (5,5), 0)
            blurred_roi = cv2.GaussianBlur(roi[:,:,c].astype(np.float32), (5,5), 0)
            roi[:,:,c] = mask * blurred_fg + (1 - mask) * blurred_roi

    elif model == 'laplacian':
        alpha = (fg[:, :, 3] / 255.0)

        for c in range(3):
            fg_lap = cv2.Laplacian(fg[:,:,c].astype(np.float32), cv2.CV_32F)
            roi_lap = cv2.Laplacian(roi[:,:,c].astype(np.float32), cv2.CV_32F)

            blended = alpha * (fg[:,:,c] + fg_lap * 0.3) + (1 - alpha) * (roi[:,:,c] + roi_lap * 0.3)
            roi[:,:,c] = np.clip(blended, 0, 255)

    bg[y:y+h, x:x+w] = roi
    return bg

print("按 'q' 退出程序")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame_original = frame.copy()
    frame_alpha = frame.copy()
    frame_pyramid = frame.copy()
    frame_laplacian = frame.copy()

    for (x, y, w, h) in faces:
        # 调整贴图大小
        size = w // 2

        # 鼻子位置
        nose_x = x + w // 2 - size // 2
        nose_y = y + h // 2

        # 1.
        cv2.rectangle(frame_original, (x, y), (x+w, y+h), (255, 0, 0), 2)
        nose_resized = cv2.resize(nose_original, (size, size))
        frame_original = tietu(frame_original, nose_resized, nose_x, nose_y, 'normal')

        # 2.
        nose_resized = cv2.resize(nose_alpha, (size, size))
        frame_alpha = tietu(frame_alpha, nose_resized, nose_x, nose_y, 'alpha')
        cv2.rectangle(frame_alpha, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 3.
        nose_resized = cv2.resize(nose_gold, (size, size))
        frame_pyramid = tietu(frame_pyramid, nose_resized, nose_x, nose_y, 'pyramid')
        cv2.rectangle(frame_pyramid, (x, y), (x+w, y+h), (0, 215, 255), 2)

        # 4.
        nose_resized = cv2.resize(nose_lap, (size, size))
        frame_laplacian = tietu(frame_laplacian, nose_resized, nose_x, nose_y, 'laplacian')
        cv2.rectangle(frame_laplacian, (x, y), (x+w, y+h), (255, 0, 255), 2)


    top = np.hstack([frame_original, frame_alpha])
    bottom = np.hstack([frame_pyramid, frame_laplacian])
    final = np.vstack([top, bottom])

    cv2.imshow('Face Sticker', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()