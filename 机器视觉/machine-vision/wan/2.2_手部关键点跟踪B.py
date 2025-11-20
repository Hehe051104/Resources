# ========================== B：捏合手势（镜像显示 + 正向文字） ==========================
# 目标：
# 1) 画面以“自拍镜像”显示（举左手→屏幕左侧也举左手）
# 2) 文字（FPS、左右手标签、PINCH状态）全部在“镜像后的帧”上绘制 → 文字不被镜像
# 3) 左右手标签做“镜像语义纠偏”（可选）
# 4) 捏合检测：拇指指尖(4) 与 食指指尖(8) 的像素距离 < 阈值 → 认为“捏合”
# ======================================================================================

import cv2                          # OpenCV：视频采集、显示
import mediapipe as mp              # MediaPipe：手部关键点
import time                         # FPS计算
import math                         # 两点欧氏距离

# --------------------------- 0) 可调参数区 -----------------------------------------------
CAM_INDEX = 0                       # 摄像头索引
CAP_WIDTH  = 1920                   # 采集宽
CAP_HEIGHT = 1080                   # 采集高
CAP_FPS    = 60                     # 目标帧率

SELFIE_VIEW = True                  # 是否镜像显示（像镜子）
SWAP_LABEL_FOR_SELFIE = True        # 镜像显示时，是否交换左右手标签文本
PINCH_THRESHOLD_PX = 40             # 捏合判定阈值（像素）：距离小于此值视为捏合

# --------------------------- 1) 初始化 MediaPipe Hands ------------------------------------
mp_hands = mp.solutions.hands       # Hands 模块
mp_draw  = mp.solutions.drawing_utils       # 绘制工具
mp_style = mp.solutions.drawing_styles      # 绘制样式

hands = mp_hands.Hands(
    static_image_mode=False,        # 视频流模式
    max_num_hands=2,                # 最多两只手
    model_complexity=1,             # 模型复杂度
    min_detection_confidence=0.5,   # 检测阈值
    min_tracking_confidence=0.5     # 跟踪阈值
)

# --------------------------- 2) 打开摄像头并设置参数 ---------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)   # 打开摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)   # 请求宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 请求高
cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)     # 请求帧率

cv2.namedWindow("Hands+Pinch", cv2.WINDOW_NORMAL)  # 可缩放窗口
cv2.resizeWindow("Hands+Pinch", 1280, 720)         # 显示尺寸

# --------------------------- 3) 工具函数：坐标镜像换算 ------------------------------------
def mirror_x(x_px: int, width_px: int, enable_mirror: bool) -> int:
    """把原始帧的x坐标换算到镜像显示帧的x坐标（仅水平翻转）。"""
    if not enable_mirror:           # 不镜像 → 坐标不变
        return x_px
    return (width_px - 1) - x_px    # 镜像 → x' = W - 1 - x

# --------------------------- 4) 主循环：采集→识别→绘制→显示 -------------------------------
prev_time = time.time()             # 上一帧时间戳

while True:
    ret, frame_bgr = cap.read()     # 读取一帧（BGR）
    if not ret:                     # 读取失败则退出
        break

    h, w = frame_bgr.shape[:2]      # 当前帧高、宽（像素）

    # —— 识别阶段：用“未镜像的原始帧”输入模型 ——
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # BGR→RGB
    results = hands.process(frame_rgb)                      # 执行检测/跟踪

    # —— 图形阶段：在原始帧上绘制骨架/点（线条镜像也不会“倒字”）——
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(                         # 在原始帧绘制骨架
                frame_bgr,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

    # —— 显示前：得到“用于显示”的帧（根据开关做水平镜像）——
    display = cv2.flip(frame_bgr, 1) if SELFIE_VIEW else frame_bgr

    # —— 文本阶段：所有文字都在“显示帧”上绘制（保证文字正向）——
    pinch_text = "PINCH: NO"        # 默认未捏合
    pinch_color = (0, 0, 255)        # 默认红色

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(results.multi_hand_landmarks,
                                    results.multi_handedness):
            # 1) 左右手标签（镜像纠偏）
            raw_label = handed.classification[0].label     # 模型原始左右手
            score     = handed.classification[0].score     # 置信度
            label = ("Left" if raw_label == "Right" else "Right") if (SELFIE_VIEW and SWAP_LABEL_FOR_SELFIE) else raw_label
            tag_text = f"{label} ({score:.2f})"            # 标签文本

            # 取手腕（索引0）原始像素坐标
            wrist = hand_lms.landmark[0]
            cx_raw = int(wrist.x * w)
            cy_raw = int(wrist.y * h)
            # 换算到显示帧坐标（镜像后的位置）
            cx_disp = mirror_x(cx_raw, w, SELFIE_VIEW)
            cy_disp = cy_raw

            # 在显示帧上绘制标签文字（正向显示）
            cv2.putText(display, tag_text, (cx_disp + 10, cy_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # 2) 捏合检测：拇指指尖(4) 与 食指指尖(8) 的距离（基于原始坐标计算）
            thumb_tip = hand_lms.landmark[4]               # 拇指尖（归一化）
            index_tip = hand_lms.landmark[8]               # 食指尖（归一化）
            x1_raw, y1_raw = int(thumb_tip.x * w), int(thumb_tip.y * h)  # 原始像素坐标
            x2_raw, y2_raw = int(index_tip.x * w), int(index_tip.y * h)  # 原始像素坐标

            # 计算原始帧中的像素距离
            dist_px = math.hypot(x2_raw - x1_raw, y2_raw - y1_raw)       # 欧氏距离

            # 为了让连线和点位在“显示帧”中位置正确，需要把x坐标镜像换算
            x1_disp = mirror_x(x1_raw, w, SELFIE_VIEW)    # 拇指尖在显示帧的x
            x2_disp = mirror_x(x2_raw, w, SELFIE_VIEW)    # 食指尖在显示帧的x
            y1_disp = y1_raw                               # y不变
            y2_disp = y2_raw                               # y不变

            # 在显示帧上画两个蓝点（指尖）
            cv2.circle(display, (x1_disp, y1_disp), 6, (255, 0, 0), -1)   # 拇指尖
            cv2.circle(display, (x2_disp, y2_disp), 6, (255, 0, 0), -1)   # 食指尖
            # 在显示帧上画一条浅蓝连线（两指尖之间）
            cv2.line(display, (x1_disp, y1_disp), (x2_disp, y2_disp), (255, 200, 0), 2)
            # 在显示帧上标注距离文本（正向可读）
            cv2.putText(display, f"dist={dist_px:.1f}px", (x2_disp + 10, y2_disp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

            # 根据阈值判定是否捏合（基于原始距离判定不会受镜像影响）
            if dist_px < PINCH_THRESHOLD_PX:
                pinch_text = "PINCH: YES"                 # 修改状态文本
                pinch_color = (0, 255, 0)                 # 绿色

    # FPS 计算（基于时间差）
    now = time.time()                                     # 当前时间
    fps = 1.0 / max(now - prev_time, 1e-6)                # FPS = 1/Δt
    prev_time = now                                       # 更新时间戳

    # 在显示帧上绘制FPS与捏合状态（正向可读）
    cv2.putText(display, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display, pinch_text, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, pinch_color, 2, cv2.LINE_AA)

    # 显示最终结果（镜像+正向文字）
    cv2.imshow("Hands+Pinch", display)                    # 展示窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):                 # 按 q 退出
        break

# --------------------------- 5) 资源清理 ---------------------------------------------------
cap.release()                      # 释放摄像头
hands.close()                      # 关闭模型
cv2.destroyAllWindows()            # 销毁窗口
