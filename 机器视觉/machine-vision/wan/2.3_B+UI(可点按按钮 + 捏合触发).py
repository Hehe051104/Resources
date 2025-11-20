# ====================== B 脚本 · UI 交互版（镜像显示 + 正向文字 + 捏合触发按钮） ======================
# 目标：
# 1) 通过 MediaPipe Hands 实时检测手部关键点（21点）并进行“捏合手势”（拇指尖↔食指尖）的判定
# 2) 屏幕上绘制三个 UI 按钮（“拔剑”“开门”“确认”），在“镜像显示”的同时保证文字正向可读
# 3) 当“捏合”发生且指尖位置落入某个按钮区域内时，触发该按钮的“事件”（以提示文本的方式显示）
# 4) 具备“消抖/冷却”机制，避免一次捏合触发多次
# ==================================================================================================

import cv2                   # OpenCV：用于视频采集、图像处理、绘制与窗口显示
import mediapipe as mp       # MediaPipe：用于手部关键点检测（Hands）
import time                  # time：用于FPS统计、提示信息计时、冷却计时
import math                  # math：用于计算两点欧氏距离（捏合距离）

# ----------------------------- 0) 可调参数（按需修改） -------------------------------------
CAM_INDEX = 0                # 摄像头索引：0=内置，1=外接（你当前使用外接，设为1）
CAP_WIDTH  = 1920            # 请求的采集分辨率宽（像素）（是否生效取决于驱动）
CAP_HEIGHT = 1080            # 请求的采集分辨率高（像素）
CAP_FPS    = 60              # 请求帧率（能否达到取决于设备/驱动/带宽/光照）

SELFIE_VIEW = True           # 是否“自拍镜像显示”（True=水平翻转后显示，更符合人类直觉）
SWAP_LABEL_FOR_SELFIE = True # 镜像显示时，是否交换左右手标签文本（让“看起来正确”）
PINCH_THRESHOLD_PX = 40      # 捏合判定阈值（像素）：拇指尖与食指尖距离 < 该值 → 认为“捏合”
PINCH_COOLDOWN_SEC = 0.6     # 捏合触发冷却时间（秒）：防止一次捏合触发多次

WINDOW_NAME = "Hands+UI"     # 显示窗口名称
WIN_W, WIN_H = 1280, 720     # 显示窗口初始大小（不影响采集分辨率）

# UI 按钮的几何参数（在“显示帧”的坐标系中定义，单位=像素）
# 说明：由于我们最终在“显示帧”上绘制按钮，因此按钮的x/y坐标天然与镜像保持一致，无需额外换算
BUTTONS = [
    {"label": "拔剑", "rect": (60,  80, 220, 80)},   # (x, y, w, h)：左上角坐标 + 宽高
    {"label": "开门", "rect": (60,  180, 220, 80)},
    {"label": "确认", "rect": (60,  280, 220, 80)},
]

# 提示信息参数（用于在触发按钮后，屏幕上方显示“事件已触发”的文本提示）
TIP_TEXT_DURATION = 1.5      # 提示文本显示时长（秒）
TIP_TEXT = ""                # 当前的提示文本（空字符串表示不显示）
TIP_DEADLINE = 0.0           # 提示文本截止显示的时间戳（超过则不再显示）

# ----------------------------- 1) 工具函数 -------------------------------------------------
def mirror_x(x_px: int, width_px: int, enable_mirror: bool) -> int:
    """
    将“原始帧”的 x 像素坐标转换为“显示帧”（可能镜像）的 x 坐标。
    - x_px：原始帧中的x坐标（像素）
    - width_px：图像宽度（像素）
    - enable_mirror：是否启用水平镜像（SELFIE_VIEW）
    返回：显示帧中的x坐标（像素）
    """
    if not enable_mirror:          # 不镜像 → 坐标不变
        return x_px
    return (width_px - 1) - x_px   # 镜像：x' = W - 1 - x

def point_in_rect(px: int, py: int, rect: tuple) -> bool:
    """
    判断点 (px, py) 是否落在矩形 rect=(x,y,w,h) 内（包含边界）。
    这里的坐标系与“显示帧”的坐标一致。
    """
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)

def draw_button(display_img, rect: tuple, label: str, hovered: bool, active: bool):
    """
    在显示帧上绘制一个按钮：
    - hovered=True  表示手指悬停（指尖在矩形内）
    - active=True   表示刚刚被触发（捏合成功）
    视觉规则：
    - 默认：灰底白字
    - 悬停：黄边
    - 触发：绿底黑字（瞬时）
    """
    x, y, w, h = rect
    # 设置颜色（BGR）
    base_color = (60, 60, 60)             # 默认底色：深灰
    text_color = (255, 255, 255)          # 默认字色：白
    border_color = (160, 160, 160)        # 默认边框：浅灰
    if hovered:
        border_color = (0, 255, 255)      # 悬停时边框：黄色
    if active:
        base_color = (0, 200, 0)          # 触发时底色：绿色
        text_color = (0, 0, 0)            # 触发时字色：黑色

    # 画按钮底矩形（实心）
    cv2.rectangle(display_img, (x, y), (x + w, y + h), base_color, thickness=-1)
    # 画按钮边框（1~2像素）
    cv2.rectangle(display_img, (x, y), (x + w, y + h), border_color, thickness=2)
    # 文字居中放置（粗略居中）
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2 - 4
    cv2.putText(display_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

# ----------------------------- 2) 初始化 MediaPipe Hands -----------------------------------
mp_hands = mp.solutions.hands                 # Hands 模块命名空间
mp_draw  = mp.solutions.drawing_utils         # 绘制工具（关键点+连线）
mp_style = mp.solutions.drawing_styles        # 绘制样式（配色等）

hands = mp_hands.Hands(                       # 创建Hands检测器（实时视频）
    static_image_mode=False,                  # False：视频流模式（追踪加速）
    max_num_hands=2,                          # 最多2只手
    model_complexity=1,                       # 模型复杂度（0/1/2）
    min_detection_confidence=0.5,             # 首帧检测阈值
    min_tracking_confidence=0.5               # 跟踪阶段阈值
)

# ----------------------------- 3) 打开摄像头并设置参数 --------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)             # 打开索引为 CAM_INDEX 的摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH) # 请求宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)# 请求高
cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)   # 请求帧率

# 创建可缩放窗口（仅影响显示大小，不改变采集分辨率）
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)

# ----------------------------- 4) 主循环：采集→识别→绘制→显示 ------------------------------
prev_time = time.time()                       # 上一帧时间戳（FPS计算用）
last_trigger_ts = 0.0                         # 上一次触发按钮的时间戳（防抖/冷却）
prev_pinch = False                            # 上一帧是否处于“捏合状态”（用于检测上升沿）

while True:
    ret, frame_bgr = cap.read()               # 读取一帧（BGR）
    if not ret:                               # 读取失败（如设备断开/占用）→ 退出
        break

    h, w = frame_bgr.shape[:2]                # 获取当前帧的高、宽（像素）
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # 模型需要RGB输入，故转换

    results = hands.process(frame_rgb)        # 执行手部关键点检测/跟踪

    # ---- 在原始帧上绘制骨架与关键点（线条镜像无碍，不会“倒字”） ----
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame_bgr,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

    # ---- 生成“显示帧”（根据 SELFIE_VIEW 决定是否镜像） ----
    display = cv2.flip(frame_bgr, 1) if SELFIE_VIEW else frame_bgr

    # ---- 计算捏合状态，并得到“用于指示点击位置”的显示坐标 ----
    pinch_now = False                         # 当前帧是否处于捏合
    cursor_x_disp, cursor_y_disp = None, None # “指针”位置（使用两指尖中点，更稳定）

    if results.multi_hand_landmarks:
        # 遍历每只手，计算其捏合距离，选取“距离最小”的那只手作为输入源
        best_dist = 1e9                       # 先设一个极大值用于比较最小距离
        best_pair = None                      # 记录最佳手的两指尖坐标
        for hand_lms in results.multi_hand_landmarks:
            # 取拇指指尖(4) 与 食指指尖(8) 的“原始像素坐标”
            x1_raw = int(hand_lms.landmark[4].x * w)  # 拇指尖x（原始帧坐标）
            y1_raw = int(hand_lms.landmark[4].y * h)  # 拇指尖y
            x2_raw = int(hand_lms.landmark[8].x * w)  # 食指尖x
            y2_raw = int(hand_lms.landmark[8].y * h)  # 食指尖y

            # 计算两点间的欧氏距离（单位：像素）
            dist_px = math.hypot(x2_raw - x1_raw, y2_raw - y1_raw)

            # 找到“距离最小”的那只手（更可能是你正在操作的手）
            if dist_px < best_dist:
                best_dist = dist_px
                best_pair = (x1_raw, y1_raw, x2_raw, y2_raw)

        # 如果找到了“最佳手”，基于它来判断捏合与指针位置
        if best_pair is not None:
            x1_raw, y1_raw, x2_raw, y2_raw = best_pair
            # 判断“是否捏合”
            if best_dist < PINCH_THRESHOLD_PX:
                pinch_now = True

            # 用两指尖“中点”作为“指针位置”（在原始帧坐标）
            mid_x_raw = (x1_raw + x2_raw) // 2
            mid_y_raw = (y1_raw + y2_raw) // 2

            # 将原始x坐标换算到“显示帧”的x坐标（因为显示可能启用镜像）
            cursor_x_disp = mirror_x(mid_x_raw, w, SELFIE_VIEW)
            cursor_y_disp = mid_y_raw  # y坐标水平镜像不变

            # 在“显示帧”上画出两个指尖（蓝色）与连线（浅蓝），以及中点（白色）
            x1_disp = mirror_x(x1_raw, w, SELFIE_VIEW)  # 拇指尖x（显示坐标）
            x2_disp = mirror_x(x2_raw, w, SELFIE_VIEW)  # 食指尖x（显示坐标）
            cv2.circle(display, (x1_disp, y1_raw), 6, (255, 0, 0), -1)       # 拇指尖点
            cv2.circle(display, (x2_disp, y2_raw), 6, (255, 0, 0), -1)       # 食指尖点
            cv2.line(display, (x1_disp, y1_raw), (x2_disp, y2_raw), (255, 200, 0), 2)  # 连线
            cv2.circle(display, (cursor_x_disp, cursor_y_disp), 6, (255, 255, 255), -1) # 中点

            # 在中点附近显示当前距离值（用于调参参考）
            cv2.putText(display, f"dist={best_dist:.1f}px", (cursor_x_disp + 10, cursor_y_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

    # ---- 绘制 UI 按钮，并判断“悬停/触发” ----
    # 规则：只有当 cursor_x_disp/cursor_y_disp 有效（检测到手）时才进行命中测试
    hovered_idx = -1                             # -1 表示当前没有悬停在任何按钮上
    active_idx = -1                              # -1 表示当前没有新触发的按钮
    now_ts = time.time()                         # 当前时间戳（用于冷却判断）

    for i, btn in enumerate(BUTTONS):
        rect = btn["rect"]                       # 取出按钮矩形
        label = btn["label"]                     # 取出按钮文本
        # 判断是否悬停：指针位置是否在按钮矩形内
        is_hovered = (cursor_x_disp is not None) and point_in_rect(cursor_x_disp, cursor_y_disp, rect)
        if is_hovered:
            hovered_idx = i

        # 触发条件：1) 当前帧处于“捏合”状态；2) 上一帧不在捏合（上升沿）；3) 在冷却时间之外；4) 指针在按钮内
        just_triggered = (pinch_now and (not prev_pinch) and
                          (now_ts - last_trigger_ts > PINCH_COOLDOWN_SEC) and is_hovered)
        if just_triggered:
            active_idx = i                       # 记录被触发的按钮序号
            last_trigger_ts = now_ts             # 更新时间戳，进入冷却
            # 设置提示文本与截止时间（用于顶部显示“事件已触发”的提示）
            TIP_TEXT = f"事件：{label} 已触发！"   # 例如 “事件：开门 已触发！”
            TIP_DEADLINE = now_ts + TIP_TEXT_DURATION

        # 绘制按钮（根据 hovered/active 状态改变样式）
        draw_button(display, rect, label, hovered=is_hovered, active=(i == active_idx))

    # ---- 左右手标签（文字始终在“显示帧”上绘制，保证正向） ----
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 读模型原始左右手标签与置信度
            raw_label = handed.classification[0].label
            score     = handed.classification[0].score
            # 如果是镜像显示，且需要纠偏，则交换左右手文本
            label = ("Left" if raw_label == "Right" else "Right") if (SELFIE_VIEW and SWAP_LABEL_FOR_SELFIE) else raw_label
            tag_text = f"{label} ({score:.2f})"
            # 获取手腕（索引0）的原始坐标，并换算到“显示帧”坐标
            cx_raw = int(hand_lms.landmark[0].x * w)
            cy_raw = int(hand_lms.landmark[0].y * h)
            cx_disp = mirror_x(cx_raw, w, SELFIE_VIEW)
            cy_disp = cy_raw
            # 在显示帧上绘制标签文本（黄色）
            cv2.putText(display, tag_text, (cx_disp + 10, cy_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # ---- FPS 计算与显示（文字在显示帧上绘制，保证正向） ----
    now = time.time()                               # 当前时间戳
    fps = 1.0 / max(now - prev_time, 1e-6)         # FPS = 1 / Δt（避免除零）
    prev_time = now                                 # 更新时间戳
    cv2.putText(display, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # ---- 顶部提示文本（在 TIP_DEADLINE 前显示） ----
    if time.time() < TIP_DEADLINE and TIP_TEXT:
        # 画一个半透明条做背景（提升可读性）
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)  # 顶部黑条
        alpha = 0.4                                                            # 透明度
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)        # 融合到 display
        # 在顶部中央写提示文字
        (tw, th), _ = cv2.getTextSize(TIP_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = (display.shape[1] - tw) // 2
        ty = 40
        cv2.putText(display, TIP_TEXT, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ---- 显示最终结果（镜像 + 正向文字） ----
    cv2.imshow(WINDOW_NAME, display)               # 把“显示帧”绘制到窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):          # 窗口聚焦时按 q 退出
        break

    # ---- 更新“上一帧捏合状态”（用于检测上升沿） ----
    prev_pinch = pinch_now

# ----------------------------- 5) 资源清理 -------------------------------------------------
cap.release()                                      # 释放摄像头
hands.close()                                      # 关闭 MediaPipe Hands
cv2.destroyAllWindows()                            # 销毁所有 OpenCV 窗口
