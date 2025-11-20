import cv2
import mediapipe as mp
import numpy as np

# ===================== helpers =====================
def lm_xy(landmarks, idx, shape):
    h, w = shape[:2]
    pt = landmarks[idx]
    return (int(pt.x * w + 0.5), int(pt.y * h + 0.5))

def dist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def clamp_rect(x0, y0, x1, y1, w, h):
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
    if x1 <= x0: x1 = min(w-1, x0+1)
    if y1 <= y0: y1 = min(h-1, y0+1)
    return x0, y0, x1, y1

def put_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

def affine_scale_masked(img, center, axes_xy, sx=1.0, sy=1.0, feather=0.25, angle_deg=0.0, ellipse=True):
    """
    在以 center 为中心、axes_xy 为半轴的区域内做各向异性缩放（warpAffine），并羽化融合。
    """
    h, w = img.shape[:2]
    cx, cy = center
    M = np.array([[sx, 0, (1 - sx) * cx],
                  [0, sy, (1 - sy) * cy]], dtype=np.float32)
    if abs(angle_deg) > 1e-3:
        R = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        M = (np.vstack([M, [0,0,1]]) @ np.vstack([R, [0,0,1]]))[0:2,:]

    warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask = np.zeros((h, w), np.float32)
    ax, ay = int(axes_xy[0]), int(axes_xy[1])
    if ellipse:
        cv2.ellipse(mask, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1, -1)
    else:
        cv2.rectangle(mask, (cx-ax, cy-ay), (cx+ax, cy+ay), 1, -1)
    if feather > 0:
        k = int(max(3, (ax + ay) * 0.5 * feather) // 2 * 2 + 1)  # odd
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    mask3 = mask[:, :, None]
    out = (warped * mask3 + img * (1 - mask3)).astype(np.uint8)
    return out

# ---- FaceMesh 人脸轮廓索引（官方常用 FACE_OVAL）----
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,
             378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,
             162,21,54,103,67,109]

def lms_to_points(landmarks, shape, idx_list):
    h, w = shape[:2]
    pts = []
    for i in idx_list:
        lm = landmarks[i]
        pts.append([int(lm.x*w + 0.5), int(lm.y*h + 0.5)])
    return np.asarray(pts, dtype=np.int32)

def soft_face_mask(shape, poly_pts, shrink=0.90, blur_ratio=0.02):
    """
    根据人脸轮廓点生成柔和掩膜：
    - shrink: 掩膜面积缩小一点，避免把发际、耳朵拉扯进来
    - blur_ratio: 掩膜羽化强度（相对于短边）
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    c = np.mean(poly_pts, axis=0, keepdims=True)
    shrink_pts = (c + (poly_pts - c) * shrink).astype(np.int32)
    cv2.fillPoly(mask, [shrink_pts], 255)
    k = max(3, int(min(h, w) * blur_ratio) | 1)  # 奇数核
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask

def barrel_remap_patch(patch, k1=-0.28, k2=0.06):
    """
    对小 patch 做径向畸变（remap），模拟哈哈镜鼓肚子效果。
    """
    ph, pw = patch.shape[:2]
    xs = np.linspace(-1, 1, pw, dtype=np.float32)
    ys = np.linspace(-1, 1, ph, dtype=np.float32)
    mx, my = np.meshgrid(xs, ys)
    r2 = mx*mx + my*my
    factor = 1 + k1*r2 + k2*(r2**2)
    src_x = ((mx * factor) + 1) * 0.5 * (pw - 1)
    src_y = ((my * factor) + 1) * 0.5 * (ph - 1)
    return cv2.remap(patch, src_x, src_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

# ===================== MediaPipe init =====================
mp_faces = mp.solutions.face_mesh
faces = mp_faces.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,             # 只追踪一张脸，提速很明显
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== video =====================
cap = cv2.VideoCapture(0)
# 为了流畅，默认降到 960×540；需要更清晰你可以再升回 1280×720 或 1920×1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("distorting mirror", cv2.WINDOW_NORMAL)
cv2.resizeWindow("distorting mirror", 1280, 720)

CELL_W, CELL_H = 640, 360  # 四宫格每格尺寸

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = faces.process(frame_rgb)

    H, W = frame_bgr.shape[:2]

    # 四个效果画面
    eff1 = frame_bgr.copy()  # 嘴巴局部放大
    eff2 = frame_bgr.copy()  # 眼球左右拉长
    eff3 = frame_bgr.copy()  # 鼻子上下拉长
    eff4 = frame_bgr.copy()  # 整脸上下左右非线性（仅脸部 + 无痕融合）

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            # 眼睛：左(33,133,159,145) 右(362,263,386,374)
            L_eye_out = lm_xy(face_lms.landmark, 33, frame_bgr.shape)
            L_eye_in  = lm_xy(face_lms.landmark, 133, frame_bgr.shape)
            L_eye_top = lm_xy(face_lms.landmark, 159, frame_bgr.shape)
            L_eye_bot = lm_xy(face_lms.landmark, 145, frame_bgr.shape)
            R_eye_in  = lm_xy(face_lms.landmark, 362, frame_bgr.shape)
            R_eye_out = lm_xy(face_lms.landmark, 263, frame_bgr.shape)
            R_eye_top = lm_xy(face_lms.landmark, 386, frame_bgr.shape)
            R_eye_bot = lm_xy(face_lms.landmark, 374, frame_bgr.shape)

            # 嘴：左右角(61,291)，中线上下(13,14)
            mouth_L   = lm_xy(face_lms.landmark, 61,  frame_bgr.shape)
            mouth_R   = lm_xy(face_lms.landmark, 291, frame_bgr.shape)
            mouth_up  = lm_xy(face_lms.landmark, 13,  frame_bgr.shape)
            mouth_dn  = lm_xy(face_lms.landmark, 14,  frame_bgr.shape)

            # 鼻尖（1）
            nose_tip  = lm_xy(face_lms.landmark, 1,   frame_bgr.shape)

            # 人脸轮廓点与柔和掩膜
            face_pts  = lms_to_points(face_lms.landmark, frame_bgr.shape, FACE_OVAL)
            mask_soft = soft_face_mask(frame_bgr.shape, face_pts, shrink=0.90, blur_ratio=0.02)

            # 面部外接框（用于限制计算范围）
            xs = face_pts[:,0]; ys = face_pts[:,1]
            x0, y0, x1, y1 = clamp_rect(xs.min(), ys.min(), xs.max(), ys.max(), W, H)
            face_rect = (x0, y0, x1, y1)

            # ---------- 1) 嘴巴局部放大（更贴近嘟嘴） ----------
            mouth_cx = (mouth_L[0] + mouth_R[0]) // 2
            mouth_cy = (mouth_up[1] + mouth_dn[1]) // 2
            mouth_w  = dist(mouth_L, mouth_R)
            mouth_h  = dist(mouth_up, mouth_dn)
            ax = int(mouth_w * 0.55)
            ay = int(max(10, mouth_h * 0.9))
            eff1 = affine_scale_masked(eff1, (mouth_cx, mouth_cy), (ax, ay),
                                       sx=1.35, sy=1.35, feather=0.35)

            # ---------- 2) 眼球左右拉长（sx>1, sy≈1） ----------
            L_cx = int((L_eye_out[0] + L_eye_in[0]) * 0.5)
            L_cy = int((L_eye_top[1] + L_eye_bot[1]) * 0.5)
            L_w  = dist(L_eye_out, L_eye_in)
            L_h  = dist(L_eye_top, L_eye_bot)
            R_cx = int((R_eye_out[0] + R_eye_in[0]) * 0.5)
            R_cy = int((R_eye_top[1] + R_eye_bot[1]) * 0.5)
            R_w  = dist(R_eye_out, R_eye_in)
            R_h  = dist(R_eye_top, R_eye_bot)
            lax = int(L_w * 0.75); lay = int(max(6, L_h * 0.95))
            rax = int(R_w * 0.75); ray = int(max(6, R_h * 0.95))
            eff2 = affine_scale_masked(eff2, (L_cx, L_cy), (lax, lay),
                                       sx=1.55, sy=1.25, feather=0.35)
            eff2 = affine_scale_masked(eff2, (R_cx, R_cy), (rax, ray),
                           sx=1.55, sy=1.25, feather=0.35)


# ---------- 3) 鼻子上下拉长（sx≈1, sy>1） ----------
            eye_dist = dist((L_cx, L_cy), (R_cx, R_cy))
            nax = int(eye_dist * 0.18)
            nay = int(eye_dist * 0.28)
            eff3 = affine_scale_masked(eff3, nose_tip, (nax, nay),
                                       sx=1.70, sy=1.35, feather=0.35)


            # ---------- 4) 整脸非线性变形（仅脸部 + Poisson 无痕融合） ----------
            # 取脸部 bbox 的子图与子掩膜
            x, y, w_rect, h_rect = cv2.boundingRect(face_pts)
            x = max(0, x); y = max(0, y)
            x2 = min(W, x + w_rect); y2 = min(H, y + h_rect)
            if x2 - x > 5 and y2 - y > 5:
                face_patch = eff4[y:y2, x:x2].copy()
                mask_patch = mask_soft[y:y2, x:x2].copy()
                # 做畸变
                warped_patch = barrel_remap_patch(face_patch, k1=-0.28, k2=0.06)
                # seamlessClone 要求 mask 是单通道 0/255，中心点在目标坐标系
                center = (x + (x2 - x)//2, y + (y2 - y)//2)
                eff4 = cv2.seamlessClone(
                    warped_patch,        # src
                    eff4,                # dst
                    mask_patch,          # mask
                    center,              # center
                    cv2.NORMAL_CLONE     # 或 cv2.MIXED_CLONE
                )

    else:
        # 兜底：没检测到人脸也给出轻量的中心变形，避免空画面
        H, W = frame_bgr.shape[:2]
        cx, cy = W//2, H//2
        eff1 = affine_scale_masked(eff1, (cx, int(H*0.65)), (int(W*0.12), int(H*0.06)),
                                   sx=1.55, sy=1.55, feather=0.35)
        eff2 = affine_scale_masked(eff2, (int(W*0.35), int(H*0.38)), (int(W*0.09), int(H*0.05)),
                                   sx=1.35, sy=1.00, feather=0.35)
        eff2 = affine_scale_masked(eff2, (int(W*0.65), int(H*0.38)), (int(W*0.09), int(H*0.05)),
                                   sx=1.35, sy=1.00, feather=0.35)
        eff3 = affine_scale_masked(eff3, (cx, int(H*0.52)), (int(W*0.07), int(H*0.10)),
                                   sx=1.00, sy=1.35, feather=0.35)
        # 大范围轻畸变
        x0, y0, x1, y1 = int(W*0.2), int(H*0.15), int(W*0.8), int(H*0.9)
        face_patch = eff4[y0:y1, x0:x1].copy()
        mask_patch = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.ellipse(mask_patch, ((x1-x0)//2, (y1-y0)//2),
                    ((x1-x0)//2-5, (y1-y0)//2-12), 0, 0, 360, 255, -1)
        k = max(3, int(min(mask_patch.shape[:2]) * 0.06) | 1)
        mask_patch = cv2.GaussianBlur(mask_patch, (k,k), 0)
        warped_patch = barrel_remap_patch(face_patch, k1=-0.25, k2=0.05)
        center = (x0 + (x1-x0)//2, y0 + (y1-y0)//2)
        eff4 = cv2.seamlessClone(warped_patch, eff4, mask_patch, center, cv2.NORMAL_CLONE)

    # 标注 & 拼图
    eff1 = put_label(eff1, "1) Mouth Local Zoom (warpAffine)")
    eff2 = put_label(eff2, "2) Eyes Horizontal Stretch (warpAffine)")
    eff3 = put_label(eff3, "3) Nose Vertical Stretch (warpAffine)")
    eff4 = put_label(eff4, "4) Full-face Barrel (mask+seamlessClone)")

    eff1s = cv2.resize(eff1, (CELL_W, CELL_H))
    eff2s = cv2.resize(eff2, (CELL_W, CELL_H))
    eff3s = cv2.resize(eff3, (CELL_W, CELL_H))
    eff4s = cv2.resize(eff4, (CELL_W, CELL_H))

    top = cv2.hconcat([eff1s, eff2s])
    bot = cv2.hconcat([eff3s, eff4s])
    grid = cv2.vconcat([top, bot])

    cv2.imshow("distorting mirror", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
