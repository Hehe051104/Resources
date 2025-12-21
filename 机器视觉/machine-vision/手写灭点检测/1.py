import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def add_cn_text(img, text, pos, color=(0, 0, 255), size=40):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("msyh.ttc", size)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", size)
        except:
            font = ImageFont.load_default()

    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_kb(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return None, x1
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k, b

def find_intersect(l1, l2):
    k1, b1 = l1
    k2, b2 = l2

    if k1 is None and k2 is None: return None
    if k1 is None: return (int(b1), int(k2 * b1 + b2))
    if k2 is None: return (int(b2), int(k1 * b2 + b1))

    if abs(k1 - k2) < 1e-4: return None

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return (int(x), int(y))

def process(img_path, save_path):
    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    if img is None: return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=h//4, maxLineGap=30)

    left_lines = []
    right_lines = []
    viz_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            k, b = get_kb(x1, y1, x2, y2)

            if k is None: continue

            angle = math.degrees(math.atan(k))
            if abs(angle) < 10 or abs(angle) > 89:
                continue

            viz_lines.append((x1, y1, x2, y2))

            if k < 0:
                left_lines.append((k, b))
            else:
                right_lines.append((k, b))

    intersections = []
    for l in left_lines:
        for r in right_lines:
            pt = find_intersect(l, r)
            if pt:
                if -w*0.2 <= pt[0] <= w*1.2 and -h*0.2 <= pt[1] <= h*1.2:
                    intersections.append(pt)

    found = True
    vp_x, vp_y = 0, 0

    if not left_lines or not right_lines:
        found = False
    elif len(intersections) < 1:
        found = False
    else:
        pts = np.array(intersections)
        mean_pt = np.mean(pts, axis=0)
        std_dev = np.std(pts, axis=0)

        if np.sum(std_dev) > (w * 0.25):
            found = False
        else:
            vp_x, vp_y = int(mean_pt[0]), int(mean_pt[1])
            if vp_y > h * 0.6:
                found = False

    res = img.copy()
    for (x1, y1, x2, y2) in viz_lines:
        cv2.line(res, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if found:
        cv2.circle(res, (vp_x, vp_y), 15, (0, 0, 255), -1)
        cv2.drawMarker(res, (vp_x, vp_y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)

        sp = (vp_x - 150, vp_y)
        if sp[0] < 0: sp = (10, vp_y)
        cv2.arrowedLine(res, sp, (vp_x - 25, vp_y), (0, 0, 255), 4, tipLength=0.4)
    else:
        res = add_cn_text(res, "大概率不包含灭点", (w//2 - 180, h//2 - 30), (255, 0, 0), 60)

    cv2.imwrite(save_path, res)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

def main():
    src_dir = "img"
    out_dir = "result"
    os.makedirs(out_dir, exist_ok=True)

    imgs_to_show = []
    titles = []

    for i in range(1, 11):
        fname = f"{i}.jpg"
        p_in = os.path.join(src_dir, fname)
        p_out = os.path.join(out_dir, f"result_{fname}")

        if os.path.exists(p_in):
            out = process(p_in, p_out)
            if out is not None:
                imgs_to_show.append(out)
                titles.append(f"Image {i}")
        else:
            print(f"Missing: {p_in}")

    if not imgs_to_show: return

    n = len(imgs_to_show)
    cols = 5
    rows = math.ceil(n / cols)

    plt.figure(figsize=(20, 4 * rows))
    for i, img in enumerate(imgs_to_show):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()