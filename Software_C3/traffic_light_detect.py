# traffic_light_detect.py —— 纯 OpenCV 交通信号灯状态识别（红/黄/绿/熄灭）
# 不使用任何深度学习/机器学习方法，仅用：颜色空间转换(HSV) + 阈值分割 + 形态学/连通域 + 轮廓。
#
# 思路：
#   信号灯每次只点亮一个灯，点亮的灯呈"高亮度+高饱和"的颜色区域；
#   未点亮灯的外壳带较暗底色。因此统计红/黄/绿三色在高亮度(HSV 的 V 值高)区的像素量，
#   超过阈值即判定为该色点亮，均低于阈值则判为熄灭。
import os
import sys
import glob
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# 数据集位置：优先用脚本同级 data/ 目录，否则回退到 temp 下载的数据集
DATA_DIR = os.path.join(BASE, 'data')
if not os.path.isdir(DATA_DIR):
    DATA_DIR = r'F:\资源夹\科研\zzlab (preparing)\pre\2524030232_张锐寒\temp\参考材料及部分模版和数据集\参考材料及部分模版和数据集\Software_C3'

# HSV 阈值（V 高=亮，S 高=颜色饱和）
V_MIN = 170      # 亮度下界，分离"点亮"与"熄灭/外壳"
S_MIN = 80       # 饱和度下界
# 各颜色判决的亮色像素阈值（经数据集统计：点亮灯约 950~1900，未点亮底色约 0~740）
THRESH = {'red': 500, 'green': 500, 'yellow': 800}
# 颜色对应 HSV 色相区间（红色在色环上跨 0 与 180 两侧，需两段）
HUE_RANGES = {
    'red':    [(0, 10), (170, 180)],
    'yellow': [(20, 38)],
    'green':  [(40, 85)],
}
# 中文/英文标注
LABEL = {'red': 'RED', 'yellow': 'YELLOW', 'green': 'GREEN', 'off': 'OFF'}
COLOR_BGR = {'red': (0, 0, 255), 'yellow': (0, 215, 255), 'green': (0, 200, 0), 'off': (180, 180, 180)}


def bright_count(hsv, hue_ranges):
    """统计指定色相区间内"高亮+高饱和"像素数"""
    total = 0
    for lo, hi in hue_ranges:
        m = cv2.inRange(hsv, np.array([lo, S_MIN, V_MIN]), np.array([hi, 255, 255]))
        total += cv2.countNonZero(m)
    return total


def largest_blob(mask):
    """返回最大连通域的 (面积, (x,y,w,h))，用于画点亮的灯框"""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return 0, (0, 0, 0, 0)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return stats[best, cv2.CC_STAT_AREA], tuple(int(v) for v in stats[best, :4])


def classify_state(img):
    """识别信号灯状态，返回 (状态, 各颜色亮像素数, 点亮灯框)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    counts = {c: bright_count(hsv, HUE_RANGES[c]) for c in ('red', 'green', 'yellow')}

    # 决策：点亮灯颜色 -> 红灯/绿灯/黄灯；否则熄灭
    state = 'off'
    for c in ('red', 'green', 'yellow'):
        if counts[c] > THRESH[c]:
            state = c
            break

    # 画点亮灯的框：取该颜色高亮像素的最大连通域
    box = (0, 0, 0, 0)
    if state != 'off':
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for lo, hi in HUE_RANGES[state]:
            mask |= cv2.inRange(hsv, np.array([lo, S_MIN, V_MIN]), np.array([hi, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 去噪
        _, box = largest_blob(mask)
    return state, counts, box


def annotate(img, state, box):
    """在图上标注状态文字与点亮灯框"""
    out = img.copy()
    x, y, w, h = box
    if w > 0 and h > 0:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), 3)
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BGR[state], 2)
    cv2.putText(out, LABEL[state], (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, COLOR_BGR[state], 3, cv2.LINE_AA)
    return out


def load_image(path):
    """兼容含中文路径的图像读取"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def run(img_dir=None, out_dir='results'):
    img_dir = img_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(img_dir, 'RGBlight*.jpg')))
    if not files:
        print('未找到图像，请检查数据集路径：', img_dir)
        return
    print(f'共 {len(files)} 张图像，输出到 {out_dir}/')
    summary = {}
    for f in files:
        img = load_image(f)
        state, counts, box = classify_state(img)
        summary[os.path.basename(f)] = state
        out = annotate(img, state, box)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f).replace('.jpg', '_out.png')), out)
        print(f"  {os.path.basename(f):16s} -> {LABEL[state]:6s}  亮像素(R,Y,G)={counts['red']},{counts['yellow']},{counts['green']}")

    from collections import Counter
    c = Counter(summary.values())
    print('\n状态统计:', dict(c))
    return summary


if __name__ == '__main__':
    # 用法：python traffic_light_detect.py [图像目录]
    run(sys.argv[1] if len(sys.argv) > 1 else None)
