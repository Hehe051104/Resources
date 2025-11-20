import cv2, numpy as np, mediapipe as mp
from collections import deque
import time
rng = np.random.default_rng(42)

# ----------------- MediaPipe Hands -----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ----------------- 小工具 -----------------
def draw_glow_line(canvas, p1, p2, color=(0,255,255), thickness=3, glow=10):
    if p1 is None or p2 is None: return canvas
    h, w = canvas.shape[:2]
    layer = np.zeros((h,w,3), dtype=np.uint8)
    cv2.line(layer, p1, p2, color, thickness, cv2.LINE_AA)
    blur = cv2.GaussianBlur(layer, (0,0), glow)
    out = cv2.addWeighted(canvas, 1.0, blur, 0.85, 0)
    cv2.line(out, p1, p2, color, max(1, thickness//2), cv2.LINE_AA)
    return out

def draw_lightning(canvas, a, b, segments=10, jitter=12, color=(255,255,0), thickness=2, glow=10, forks=2):
    if a is None or b is None: return canvas
    ax, ay = a; bx, by = b
    pts = []
    for i in range(segments+1):
        t = i/segments
        x = int(ax*(1-t) + bx*t)
        y = int(ay*(1-t) + by*t)
        dx, dy = bx-ax, by-ay
        L = (dx*dx+dy*dy)**0.5 + 1e-6
        nx, ny = -dy/L, dx/L
        j = rng.normal(0, jitter*(1-abs(0.5-t)*1.8))
        x = int(x + nx*j); y = int(y + ny*j)
        pts.append([x,y])
    pts = np.array(pts, np.int32)

    h, w = canvas.shape[:2]
    layer = np.zeros((h,w,3), dtype=np.uint8)
    cv2.polylines(layer, [pts], False, color, thickness, cv2.LINE_AA)

    # 分叉
    for _ in range(forks):
        k = rng.integers(2, max(3, segments-1))
        p = pts[k]
        ang = np.arctan2(by-ay, bx-ax) + rng.normal(0, 0.7)
        length = rng.integers(18, 42)
        q = (int(p[0] + np.cos(ang)*length), int(p[1] + np.sin(ang)*length))
        cv2.line(layer, tuple(p), q, color, max(1, thickness-1), cv2.LINE_AA)

    blur = cv2.GaussianBlur(layer, (0,0), glow)
    out = cv2.addWeighted(canvas, 1.0, blur, 0.9, 0)
    cv2.polylines(out, [pts], False, color, 1, cv2.LINE_AA)
    return out

def to_px(lm, W, H):
    return (int(lm.x*W), int(lm.y*H))

def palm_center(landmarks, W, H):
    arr = np.array([(lm.x*W, lm.y*H) for lm in landmarks], dtype=np.float32)
    c = arr.mean(axis=0)
    return (int(c[0]), int(c[1]))

# 指尖索引（拇指到小指）
TIP_IDS = [4, 8, 12, 16, 20]

# ----------------- 主循环 -----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("摄像头未打开。")

fps_hist = deque(maxlen=12); t_prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    img_string = frame.copy()   # 2.1：拉丝（指尖↔指尖发光丝）
    img_between = frame.copy()  # 2.2：两手之间多指雷电
    img_inside  = frame.copy()  # 2.3：每只手内部雷电

    hands_pts = []   # [{'center':(x,y), 'tips':{id:(x,y)}, 'all':[(x,y),...]}]
    if res.multi_hand_landmarks:
        for hlm in res.multi_hand_landmarks:
            tips = {idx: to_px(hlm.landmark[idx], W, H) for idx in TIP_IDS}
            center = palm_center(hlm.landmark, W, H)
            all_pts = [to_px(p, W, H) for p in hlm.landmark]
            hands_pts.append({'center': center, 'tips': tips, 'all': all_pts})

            mp_drawing.draw_landmarks(
                img_string, hlm, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

    # -------- 2.1：拉丝：五个指尖对接为“发光丝” --------
    if len(hands_pts) == 2:
        L, R = hands_pts[0], hands_pts[1]
        for k in TIP_IDS:
            img_string = draw_glow_line(img_string, L['tips'][k], R['tips'][k],
                                        color=(0,255,255), thickness=3, glow=10)

    # -------- 2.2：两手之间（无论识别多少根手指，都有雷电） --------
    # 规则：对应指尖都画；若某些指尖缺失，至少画掌心↔掌心一道。
    if len(hands_pts) == 2:
        (L, R) = hands_pts[0], hands_pts[1]
        any_drawn = False
        for k in TIP_IDS:
            pL, pR = L['tips'].get(k), R['tips'].get(k)
            if pL and pR:
                img_between = draw_lightning(
                    img_between, pL, pR,
                    segments=12, jitter=int(0.03*W), forks=2
                )
                any_drawn = True
        if not any_drawn:  # 兜底
            img_between = draw_lightning(
                img_between, L['center'], R['center'],
                segments=12, jitter=int(0.03*W), forks=3
            )

    # -------- 2.3：每只手内部（雷电不跨手） --------
    # 规则：每只手“掌心→每个指尖”都画一道雷。
    for hand in hands_pts:
        c = hand['center']
        for k in TIP_IDS:
            p = hand['tips'][k]
            img_inside = draw_lightning(
                img_inside, c, p,
                segments=9, jitter=int(0.02*W), forks=1
            )

    # 三窗口
    cv2.imshow("2.1", img_string)
    cv2.imshow("2.2", img_between)
    cv2.imshow("2.3", img_inside)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
