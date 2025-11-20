import cv2, os, glob, re
import numpy as np

# ===== 1) 基本配置 =====
CALIB_DIR = os.path.dirname(__file__)     # 你的图片所在目录
OUT_DIR   = os.path.join(CALIB_DIR, "output")
CHECKERBOARD = (8, 5)                     # 角点检测脚本已经确认
SQUARE_SIZE  = 25.0                       # 每格实际尺寸（mm 或 cm；自定义，但前后一致即可）
os.makedirs(OUT_DIR, exist_ok=True)

print("工作目录:", os.getcwd())
print("读图目录:", CALIB_DIR)
print("输出目录:", OUT_DIR)


def nat_key(p):
    m = re.search(r'(\d+)(?=\.\w+$)', os.path.basename(p))
    return int(m.group(1)) if m else p

# ===== 2) 读图并提角点（用SB接口，与你探测保持一致）=====
images = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jp*")), key=nat_key)
if not images:
    raise FileNotFoundError(f"没找到图片：{CALIB_DIR}\\*.jp*")

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints, imgpoints, used_list = [], [], []
flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

for fp in images:
    img = cv2.imread(fp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags_sb)
    if ok:
        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float32))
        used_list.append(fp)
    else:
        print("跳过（未检出角点）：", fp)

print(f"用于标定的有效图片数：{len(used_list)} / {len(images)}")
if len(used_list) < 5:
    raise RuntimeError("可用图片太少（<5），请检查棋盘/光照/角点参数。")

# ===== 3) 标定 =====
h, w = cv2.imread(used_list[0]).shape[:2]
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
print("\n=== 标定完成 ===")
print("相机内参 K:\n", K)
print("畸变系数 dist:\n", dist.ravel())

# ===== 4) 重投影误差 =====
tot_err = 0
for i in range(len(objpoints)):
    proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    tot_err += err
mean_err = tot_err / len(objpoints)
print(f"平均重投影误差：{mean_err:.4f} 像素")

# ===== 5) 试校正 + 导出示例图 =====
test = cv2.imread(used_list[0])
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
und = cv2.undistort(test, K, dist, None, newK)
x, y, w2, h2 = roi
und = und[y:y+h2, x:x+w2]

preview_path = os.path.join(OUT_DIR, "undistorted_preview.jpg")
cv2.imwrite(preview_path, und)
print("已导出示例校正图：", preview_path)

# ===== 6) 保存参数 =====
npz_path = os.path.join(OUT_DIR, "camera_params.npz")
np.savez(npz_path,
         K=K, dist=dist, rvecs=rvecs, tvecs=tvecs, newK=newK,
         checkerboard=CHECKERBOARD, square_size=SQUARE_SIZE,
         used_images=np.array(used_list))
print("参数已保存：", npz_path)
