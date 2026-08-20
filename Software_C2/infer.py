# infer.py —— 用训练好的 YOLO 权重对图片推理并保存标注结果
# 用法：python infer.py [权重路径] [图片或目录]
import os
import sys
import glob
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE, 'runs', 'detect', 'weights', 'best.pt')
# 官方给定待检测图
DEFAULT_IMG = r'F:\资源夹\科研\zzlab (preparing)\pre\2524030232_张锐寒\temp\参考材料及部分模版和数据集\参考材料及部分模版和数据集\Software_C2\datas'


def load_image(path):
    import numpy as np
    import cv2
    data = np.fromfile(path, dtype=np.uint8)  # 兼容中文路径
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    weights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    src = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMG
    model = YOLO(weights)
    os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)

    files = glob.glob(os.path.join(src, '*.jpg')) if os.path.isdir(src) else [src]
    for f in files:
        img = load_image(f)
        res = model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        out = res.plot()                       # 带框标注图
        name = os.path.basename(f).replace('.jpg', '_detect.png')
        ok, buf = cv2.imencode('.png', out)    # 兼容中文路径保存
        if ok:
            buf.tofile(os.path.join(BASE, 'results', name))
        print(f'{os.path.basename(f)}: 检测到 {len(res.boxes)} 个目标 -> results/{name}')
