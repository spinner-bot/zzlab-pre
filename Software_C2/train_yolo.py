# train_yolo.py —— 用 ultralytics YOLOv8 训练目标检测模型（社区人员/非社区人员/电动车）
# 前置：pip install ultralytics，并已按 data.yaml 准备好标注好的数据集。
# 用法：python train_yolo.py
import os
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE, 'data.yaml')

if __name__ == '__main__':
    # 从零训练（或换成 yolov8n.pt 做迁移学习更快）
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=DATA_YAML,
        epochs=100,            # 数据量小(68张)，可加大数据增强/提高 epochs
        imgsz=640,
        batch=16,
        device=0,              # 0 = 第一张 GPU(H200)
        lr0=0.01,
        augment=True,
        project=os.path.join(BASE, 'runs'),
        name='detect',
    )
    print('训练完成，权重保存在 runs/detect/weights/best.pt')
