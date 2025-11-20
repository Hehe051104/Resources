# step1_check_env.py
# pip install opencv-python  （若未安装）
import cv2
from pathlib import Path

print("OpenCV 版本:", cv2.__version__)
haar_dir = Path(cv2.data.haarcascades)
print("Haar 模型目录:", haar_dir)

need = {
    "face": "haarcascade_frontalface_default.xml",
    "eye": "haarcascade_eye.xml",
    "mouth/smile": "haarcascade_smile.xml",
    "nose(optional)": "haarcascade_mcs_nose.xml",  # 没有也先不慌
}

for name, fname in need.items():
    p = haar_dir / fname
    print(f"{name:<14} ->", "OK" if p.exists() else "MISSING", "-", p)


