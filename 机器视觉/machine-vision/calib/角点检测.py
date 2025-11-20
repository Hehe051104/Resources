import cv2
import glob, os

# ====== 配置 ======
CALIB_DIR = os.path.dirname(__file__)   # 当前代码所在的calib文件夹  # 你的图片文件夹

PATTERN_CANDIDATES = [
    (9, 6), (6, 9),   # 常见 9x6，含互换
    (8, 5), (5, 8),   # 常见 8x5，含互换
    (11, 8), (8, 11), # 你也可以按需再加
]
SAVE_DIR = os.path.join(CALIB_DIR, "debug_out")
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取所有图
images = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jp*")))
print(f"发现 {len(images)} 张图片")
assert len(images) > 0, "没有读到任何图片！"

def try_detect(gray, pattern):
    """尽可能鲁棒地找角点：先试 SB，再试传统接口 + 多 flags + 预处理"""
    w, h = pattern

    # 1) 新版 SB 检测（更鲁棒）
    try:
        flags_sb = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCornersSB(gray, (w, h), flags_sb)
        if ret and corners is not None:
            return True, corners, "SB"
    except Exception:
        pass

    # 2) 传统接口（多 flags）
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, (w, h), flags)
    if ret:
        # 亚像素细化
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        )
        return True, corners, "LEGACY"

    # 3) 预处理后再试：CLAHE 提升对比度 + 轻微模糊抑噪
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (3,3), 0)

    ret, corners = cv2.findChessboardCorners(blur, (w, h), flags)
    if ret:
        corners = cv2.cornerSubPix(
            blur, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        )
        return True, corners, "LEGACY+CLAHE"
    return False, None, "FAIL"

report = []
for fp in images:
    img = cv2.imread(fp)
    if img is None:
        report.append((fp, "IO_FAIL", None, None))
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok = False
    used = None
    used_pat = None
    used_corners = None

    for pat in PATTERN_CANDIDATES:
        ret, corners, tag = try_detect(gray, pat)
        if ret:
            ok = True
            used = tag
            used_pat = pat
            used_corners = corners
            # 可视化并保存
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pat, used_corners, True)
            base = os.path.basename(fp)
            outp = os.path.join(SAVE_DIR, f"{os.path.splitext(base)[0]}_OK_{pat[0]}x{pat[1]}_{tag}.jpg")
            cv2.imwrite(outp, vis)
            break

    if not ok:
        # 保存增强失败的对比图，便于你目测
        base = os.path.basename(fp)
        outp = os.path.join(SAVE_DIR, f"{os.path.splitext(base)[0]}_FAIL.jpg")
        cv2.imwrite(outp, img)
        report.append((fp, "NO_CORNERS", None, None))
    else:
        report.append((fp, "OK", used_pat, used))

# 打印汇总
ok_cnt = sum(1 for r in report if r[1] == "OK")
print(f"\n检测完成：OK {ok_cnt} / {len(report)} 张。调试图已保存到 {SAVE_DIR}")
for r in report:
    print(r)
