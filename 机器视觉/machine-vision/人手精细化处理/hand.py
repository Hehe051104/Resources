import cv2
import mediapipe as mp
import numpy as np

# --------------------------- 0) 参数区 --------------------------------------------------------
KSIZE = 7             # 卷积核大小（奇数：5/7/9）
GAUSS_SIGMA = 0       # 高斯滤波 σ（0 表示自动计算）
CAM_INDEX = 0         # 摄像头索引
BOX_PAD = 10          # 外接框边缘留白像素

# --------------------------- 1) 初始化 MediaPipe Hands ---------------------------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,                # 最多检测两只手
    min_detection_confidence=0.5,   # 检测阈值
    min_tracking_confidence=0.5     # 跟踪阈值
)

# --------------------------- 2) 工具函数 -------------------------------------------------------
def make_hand_mask(frame_shape, landmarks):
    """根据关键点生成手部掩膜（255=手部；0=背景）"""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if not landmarks:
        return mask
    h, w = frame_shape[:2]
    for hand in landmarks:
        pts = np.array([(int(p.x * w), int(p.y * h)) for p in hand.landmark], np.int32)
        hull = cv2.convexHull(pts)  #用 OpenCV 计算凸包（Convex Hull）：就是把所有关键点“包起来”，形成一个封闭的凸多边形。凸包 ≈ 手掌的外轮廓（比矩形框更贴手的形状）。如果直接用矩形框，会把背景多带进去，而凸包贴合手的边缘。
        cv2.fillConvexPoly(mask, hull, 255)  #在黑色 mask 图上，把这个凸包区域填充为白色（255）
    return mask

def filter_on_hand(frame, mask, mode):
    """仅对手部区域滤波，背景保持纯黑"""
    if mode == "mean":
        blur = cv2.blur(frame, (KSIZE, KSIZE))
    elif mode == "gauss":
        blur = cv2.GaussianBlur(frame, (KSIZE, KSIZE), GAUSS_SIGMA)
    else:  # median
        blur = cv2.medianBlur(frame, KSIZE)
    out = np.zeros_like(frame)
    out[mask == 255] = blur[mask == 255]
    return out

# --------------------------- 3) 主循环 --------------------------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # 转RGB进行检测
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 原图副本：绘制关键点与矩形框
    view = frame.copy()
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # 绘制关键点与骨架
            mp_draw.draw_landmarks(view, hand, mp_hands.HAND_CONNECTIONS)
            # 绘制外接矩形框
            xs = [int(p.x * w) for p in hand.landmark]
            ys = [int(p.y * h) for p in hand.landmark]
            x1, x2 = max(min(xs) - BOX_PAD, 0), min(max(xs) + BOX_PAD, w - 1)
            y1, y2 = max(min(ys) - BOX_PAD, 0), min(max(ys) + BOX_PAD, h - 1)
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 构造手部掩膜
    mask = make_hand_mask(frame.shape, results.multi_hand_landmarks)

    # 三种滤波（仅手部）
    mean_hand   = filter_on_hand(frame, mask, "mean")
    gauss_hand  = filter_on_hand(frame, mask, "gauss")
    median_hand = filter_on_hand(frame, mask, "median")

    # 显示结果
    plg1 = cv2.hconcat([view, mean_hand])
    plg2 = cv2.hconcat([gauss_hand, median_hand])
    out  = cv2.vconcat([plg1, plg2])
    cv2.imshow("Hand Smoothing (q: quit)", out)


    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
