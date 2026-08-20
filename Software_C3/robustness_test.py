# robustness_test.py —— 构建干扰数据集并验证红绿灯识别算法的鲁棒性
# 覆盖题目要求的干扰场景：阳光反光 / 夜间 / 红色车尾灯误检 / 绿色广告牌误检 / 摄像头抖动
# 通过数据增强由基准集生成干扰图，再送入同一识别算法，检查状态是否仍正确。
import os
import cv2
import numpy as np

import traffic_light_detect as det

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = det.DATA_DIR
INTERFERENCE_DIR = os.path.join(BASE, 'interference')
RESULT_DIR = os.path.join(BASE, 'results_interference')


def glare(img, strength=0.9):
    """阳光反光：加白色圆形高光，提高局部亮度并去饱和"""
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy, r = int(w * 0.3), int(h * 0.35), int(min(h, w) * 0.22)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = np.clip(1 - dist / r, 0, 1) ** 2
    mask = mask[:, :, None]
    out = out * (1 - strength * mask) + 255 * strength * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def night(img, factor=0.35):
    """夜间：整体压暗并降低饱和"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.5
    hsv[..., 2] *= factor
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def red_taillight(img):
    """红色车尾灯误检：在画面中加入若干红色发光圆点"""
    out = img.copy()
    h, w = out.shape[:2]
    rng = np.random.RandomState(1)
    for _ in range(3):
        cx, cy = rng.randint(0, w), rng.randint(int(h * 0.6), h)
        cv2.circle(out, (cx, cy), rng.randint(8, 15), (40, 40, 255), -1)
    return out


def green_sign(img):
    """绿色广告牌误检：加入一大块亮绿色区域"""
    out = img.copy()
    h, w = out.shape[:2]
    x0, y0 = int(w * 0.05), int(h * 0.05)
    x1, y1 = int(w * 0.25), int(h * 0.2)
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 0), -1)
    return out


def shake(img, sigma=3):
    """摄像头抖动：高斯模糊模拟运动模糊"""
    return cv2.GaussianBlur(img, (0, 0), sigma)


AUGMENT = {'glare': glare, 'night': night, 'taillight': red_taillight,
           'green_sign': green_sign, 'shake': shake}


def save_image(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or '.jpg', img)
    if ok:
        buf.tofile(path)


def main():
    os.makedirs(INTERFERENCE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])

    # 每种干扰挑若干张基准图生成
    print('生成干扰数据集 & 验证鲁棒性...')
    stats = {}
    for fname in files[:20]:            # 用前20张基准图做演示
        img = det.load_image(os.path.join(DATA_DIR, fname))
        base_state, _, _ = det.classify_state(img)          # 基准真实状态
        for name, fn in AUGMENT.items():
            aug = fn(img)
            # 干扰图可能改变照明，先尝试恢复识别
            st, counts, box = det.classify_state(aug)
            ok = (st == base_state)
            stats.setdefault(name, [0, 0])
            stats[name][1] += 1
            if ok:
                stats[name][0] += 1
            # 保存干扰图与标注结果
            ifn = f'{fname[:-4]}_{name}.jpg'
            save_image(os.path.join(INTERFERENCE_DIR, ifn), aug)
            save_image(os.path.join(RESULT_DIR, ifn[:-4] + '_out.png'),
                       det.annotate(aug, st, box))

    print('\n=== 鲁棒性结果（各干扰下识别与基准一致的比率）===')
    for name, (hit, tot) in stats.items():
        print(f'  {name:12s}: {hit}/{tot}  ({hit/tot*100:.0f}%)')


if __name__ == '__main__':
    main()
