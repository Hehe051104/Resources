import mediapipe
import cv2
import numpy as np

mp_faces = mediapipe.solutions.face_mesh
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

faces = mp_faces.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Mesh", 1280, 720)

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = faces.process(frame_rgb)

    # 在原始图像上绘制人脸网格
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 绘制人脸网格
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_faces.FACEMESH_TESSELATION,  # 绘制网格连接
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # 可选：绘制轮廓（眼睛、嘴唇等）
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_faces.FACEMESH_CONTOURS,  # 绘制轮廓
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # 显示带有绘制的图像
    cv2.imshow("Face Mesh", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()