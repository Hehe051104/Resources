import cv2
import numpy as np


def get_gradients(g):
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    return mag, ang


def nms(mag, ang):
    H, W = mag.shape
    out = np.zeros_like(mag)
    ang = ang * 180 / np.pi
    ang[ang < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            a = ang[i, j]
            m = mag[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                p, q = mag[i, j+1], mag[i, j-1]
            elif 22.5 <= a < 67.5:
                p, q = mag[i-1, j+1], mag[i+1, j-1]
            elif 67.5 <= a < 112.5:
                p, q = mag[i-1, j], mag[i+1, j]
            else:  # 112.5~157.5
                p, q = mag[i-1, j-1], mag[i+1, j+1]
            out[i, j] = m if (m >= p and m >= q) else 0
    return out


def double_thresh(img, low=0.05, high=0.15):
    hi = img.max() * high
    lo = hi * low
    strong, weak = 255, 75
    res = np.zeros_like(img, dtype=np.uint8)
    res[img >= hi] = strong
    res[(img <= hi) & (img >= lo)] = weak
    return res, weak, strong


def hysteresis(img, weak, strong):
    H, W = img.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if img[i,j] == weak:
                if strong in img[i-1:i+2, j-1:j+2]:
                    img[i,j] = strong
                else:
                    img[i,j] = 0
    img[img != strong] = 0
    return img


def canny_simple(gray, low=0.05, high=0.15):
    g = cv2.GaussianBlur(gray, (5,5), 1.0)
    mag, ang = get_gradients(g)
    n = nms(mag, ang)
    dt, weak, strong = double_thresh(n, low, high)
    return hysteresis(dt, weak, strong)


if __name__ == '__main__':
    img_path = "1.jpg"
    out_path = "canny_out.png"

    img = cv2.imread(img_path, 0)
    if img is None:
        raise SystemExit(f"读图失败：{img_path}")

    out = canny_simple(img)
    cv2.imwrite(out_path, out)

    # ---- 显示结果 ----
    cv2.imshow("Input", img)
    cv2.imshow("Canny Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("处理完成，输出：", out_path)
