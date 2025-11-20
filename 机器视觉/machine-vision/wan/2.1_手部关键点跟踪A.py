# ========================== A：MediaPipe Hands（镜像显示 + 正向文字） ==========================
# 目标：
# 1) 摄像头画面以“自拍镜像”方式显示（举左手→屏幕左侧也举左手），方便人类直觉
# 2) 所有文字（FPS、左右手标签）在“镜像后的帧”上绘制 → 文字不被镜像，保持正常可读
# 3) 左右手标签做“镜像语义纠偏”——镜像显示时，把Left/Right标签交换以符合人眼直觉
# ============================================================================================

import cv2                          # OpenCV：视频采集、图像处理、显示
import mediapipe as mp              # MediaPipe：手部关键点检测
import time                         # 时间工具：计算FPS

# --------------------------- 0) 可调参数区 -----------------------------------------------
CAM_INDEX = 0                       # 摄像头索引（外接摄像头常是1）
CAP_WIDTH  = 1920                   # 请求采集分辨率宽（像素）
CAP_HEIGHT = 1080                   # 请求采集分辨率高（像素）
CAP_FPS    = 60                     # 请求帧率（能否达成取决于设备/驱动/带宽/光照）

SELFIE_VIEW = True                  # 是否开启“自拍镜像显示”（True=水平翻转后显示）
SWAP_LABEL_FOR_SELFIE = True        # 镜像显示时，是否交换左右手标签文本（建议True）

# --------------------------- 1) 初始化 MediaPipe Hands ------------------------------------
mp_hands = mp.solutions.hands       # Hands 模块命名空间
mp_draw  = mp.solutions.drawing_utils       # 绘制工具
mp_style = mp.solutions.drawing_styles      # 绘制样式

# 创建Hands检测器（实时流模式）
hands = mp_hands.Hands(
    static_image_mode=False,        # 是否启用静态图片模式- True：每一帧都重新检测手部，耗时高，适合单张图片检测。False：视频流模式（内部启用追踪，加速稳定）
    max_num_hands=2,                # 最多检测两只手
    model_complexity=1,             # 模型复杂度（0/1/2，越大越准但越慢）
    min_detection_confidence=0.5,   # 首帧检测阈值 值越高 → 检测更严格，可能漏检 在光照差、背景杂乱时可设低点，比如 0.5。- 只想检测特别确定的手势 → 0.8
    min_tracking_confidence=0.5     # 跟踪阶段阈值  常用：0.5~0.7 快速运动场景建议 0.7。
)

# --------------------------- 2) 打开摄像头并设置参数 ---------------------------------------
cap = cv2.VideoCapture(CAM_INDEX)   # 打开指定索引的摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)   # 请求宽
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 请求高
cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)     # 请求帧率

# 创建可缩放窗口（不影响采集分辨率，仅影响显示尺寸）
cv2.namedWindow("Hands", cv2.WINDOW_NORMAL)      # 可调整大小的显示窗口
cv2.resizeWindow("Hands", 1280, 720)             # 初始窗口大小（显示端）

# --------------------------- 3) 工具函数：坐标镜像换算 ------------------------------------
def mirror_x(x_px: int, width_px: int, enable_mirror: bool) -> int:
    """把原始帧中的x像素坐标，转换为镜像显示帧中的x坐标。
       - x_px: 原始帧中的x坐标（像素）
       - width_px: 图像宽度（像素）
       - enable_mirror: 是否做水平镜像显示（SELFIE_VIEW）
       返回：在显示帧（可能镜像）中的x坐标（像素）
    """
    if not enable_mirror:           # 如果不镜像，坐标不变
        return x_px
    return (width_px - 1) - x_px    # 镜像：x' = W - 1 - x

# --------------------------- 4) 主循环：采集→识别→绘制→显示 -------------------------------
prev_time = time.time()             # 记录上一帧时间（用于FPS计算）

while True:
    ret, frame_bgr = cap.read()     # 从摄像头读取一帧（BGR格式）
    if not ret:                     # 读取失败（设备被占用/断开）则退出
        break

    h, w = frame_bgr.shape[:2]      # 获取当前帧的高和宽（像素）

    # —— 识别阶段：必须用“未镜像的原始帧”输入模型，以保证左右手判断正确 ——
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # BGR→RGB
    results = hands.process(frame_rgb)                      # 执行手部关键点检测/跟踪  results是个对象

    # —— 绘制阶段（图形/骨架）：先画到原始帧上（这些线条镜像也OK，不会“倒字”）——
    if results.multi_hand_landmarks:                        # 如果检测到至少一只手  列表
        for hand_lms in results.multi_hand_landmarks:       # 遍历每只手的关键点  单只手的 21 个关键点  列表
            mp_draw.draw_landmarks(                     # 在原始帧上绘制手部关键点和骨架
                frame_bgr,                              # 参数1：要绘制在上面的目标图像
                hand_lms,                               # 参数2：检测到的手部关键点数据
                mp_hands.HAND_CONNECTIONS,              # 参数3：关键点之间的连接关系定义
                mp_style.get_default_hand_landmarks_style(),  # 参数4：关键点的绘制样式
                mp_style.get_default_hand_connections_style() # 参数5：连接线的绘制样式
            )

    # —— 显示前：根据需求对整帧做水平镜像（得到用于显示的帧）——
    display = cv2.flip(frame_bgr, 1) if SELFIE_VIEW else frame_bgr

    # —— 文字阶段：所有文字（FPS、左右手标签）一律在“显示帧”上绘制，保证文字正向 ——
    # 左右手标签需要读取 handedness(每只手的 左右手分类)，并进行“镜像语义纠偏”（可选）
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handed in zip(results.multi_hand_landmarks,results.multi_handedness):
            # 取手腕（索引0）的原始像素坐标（在未镜像帧中的位置）
            wrist = hand_lms.landmark[0]                   # 手腕关键点（归一化坐标）
            cx_raw = int(wrist.x * w)                      # 原始x像素
            cy_raw = int(wrist.y * h)                      # 原始y像素

            # 将原始x坐标换算到“显示帧”的x坐标（如果镜像）
            cx_disp = mirror_x(cx_raw, w, SELFIE_VIEW)     # 显示帧中的x
            cy_disp = cy_raw                                # y坐标不受水平镜像影响

            # 原始模型输出的左右手标签（未镜像语义）
            #handed.classification 是一个列表，存放手的分类结果
            #虽然列表里通常只有 1 个元素，但依然要用 [0] 取出。
            raw_label = handed.classification[0].label     # "Left" 或 "Right"
            score     = handed.classification[0].score     # 置信度（0~1）

            # 镜像显示时，是否交换标签文本：使得“看起来符合镜像直觉”
            if SELFIE_VIEW and SWAP_LABEL_FOR_SELFIE:
                label = "Left" if raw_label == "Right" else "Right"
            else:
                label = raw_label

            # 拼接显示文本（左右手 + 置信度）
            tag_text = f"{label} ({score:.2f})"

            # 在“显示帧”（已镜像）上绘制标签文字（此时文字不会被镜像）
            cv2.putText(display, tag_text, (cx_disp + 10, cy_disp - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # 计算FPS（基于时间差）
    now = time.time()                                   # 当前时间
    fps = 1.0 / max(now - prev_time, 1e-6)             # FPS = 1/Δt（防止除零）
    prev_time = now                                     # 更新时间戳

    # 在“显示帧”上绘制FPS（保证文字正向）
    cv2.putText(display, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示最终结果（镜像+正向文字）
    cv2.imshow("Hands", display)                        # 把显示帧画到窗口
    if cv2.waitKey(1) & 0xFF == ord('q'):              # 监听键盘：按 q 退出
        break

# --------------------------- 5) 资源清理 ---------------------------------------------------
cap.release()                      # 释放摄像头资源
hands.close()                      # 关闭 MediaPipe Hands
cv2.destroyAllWindows()            # 销毁所有OpenCV窗口
