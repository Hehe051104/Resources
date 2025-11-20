import cv2 as cv
import numpy as np
import glob
import os
from pathlib import Path

def skin_mask(img):
    """通过YCrCb+HSV双阈值提取肤色区域"""
    img = cv.GaussianBlur(img, (5,5), 0)

    # YCrCb阈值
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    mask1 = cv.inRange(ycrcb, (0,135,85), (255,180,135))

    # HSV阈值（两段：橙/棕 + 红端）
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    m2a = cv.inRange(hsv, (0,30,40),   (25,255,255))
    m2b = cv.inRange(hsv, (160,30,40), (179,255,255))
    mask2 = cv.bitwise_or(m2a, m2b)

    # 交集 + 形态学净化
    mask = cv.bitwise_and(mask1, mask2)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
    return mask

def classify_skin(img):
    """根据V通道亮度判断肤色深浅"""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    mean_v = np.mean(v)
    label = "浅色肤色" if mean_v > 120 else "深色肤色"
    return mean_v, label

def put_label(img, text, pos=(12, 36)):
    """在图像上绘制分类结果，不修改原图"""
    out = img.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.8, 2
    (tw, th), _ = cv.getTextSize(text, font, scale, thick)
    x, y = pos
    pad = 8
    # 绘制半透明背景
    overlay = out.copy()
    cv.rectangle(overlay, (x-pad, y-th-pad), (x+tw+pad, y+pad), (0,0,0), -1)
    cv.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    # 白色文字
    cv.putText(out, text, (x, y), font, scale, (255,255,255), thick, cv.LINE_AA)
    return out

def to_three_channels(mask):
    """单通道转三通道，便于拼接展示"""
    return cv.cvtColor(mask, cv.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask

def resize_to_height(img, h):
    """等比缩放到指定高度"""
    if img.shape[0] == h:
        return img
    w = int(img.shape[1] * (h / img.shape[0]))
    return cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

def imsave(path, img):
    """中文路径安全保存函数"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ret, buf = cv.imencode(ext, img)
    if ret:
        buf.tofile(str(path))
    else:
        raise IOError(f"保存失败: {path}")

if __name__ == "__main__":
    # 搜索当前目录下所有图片
    image_paths = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
    if not image_paths:
        print("当前文件夹下没有找到图片！")
        exit()

    out_dir = "结果输出"
    os.makedirs(out_dir, exist_ok=True)

    # 固定窗口复用，不会重复创建
    cv.namedWindow("肤色识别结果", cv.WINDOW_NORMAL)

    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"无法读取: {path}")
            continue

        # 生成肤色掩膜 & 抠图
        mask = skin_mask(img)
        skin = cv.bitwise_and(img, img, mask=mask)

        # 保存结果（用 imencode + tofile，支持中文路径）
        base = os.path.splitext(os.path.basename(path))[0]
        imsave(os.path.join(out_dir, f"{base}_mask.png"), mask)
        imsave(os.path.join(out_dir, f"{base}_skin.png"), skin)

        # 分类
        mean_v, label = classify_skin(img)
        print(f"{path} → 平均亮度(V)={mean_v:.2f} → {label}")

        # 在抠图上绘制分类结果
        preview = put_label(skin, f"{label} | V均值={mean_v:.1f}")

        # 拼接展示：原图 | 掩膜 | 抠图+分类
        h_target = min(img.shape[0], 380)
        panel = cv.hconcat([
            resize_to_height(img, h_target),
            resize_to_height(to_three_channels(mask), h_target),
            resize_to_height(preview, h_target)
        ])

        # 显示结果
        cv.imshow("肤色识别结果", panel)
        key = cv.waitKey(0) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cv.destroyAllWindows()
    print(f"\n处理完成！结果保存在：{out_dir} 文件夹")
