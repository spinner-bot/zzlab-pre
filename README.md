# 智泽实验室 2026 招新考核提交

- **学号**：2524030232
- **姓名**：张锐寒
- **选题方向**：软件类
- **个人简介**：见 `自我介绍.md`（提交时请另导出为 `自我介绍.pdf`）

## 选题说明
软件类规则：A 类（面向对象）必做，其余四类（B/C/D/E）任选两类。本次完成 **A 类必做 + C/D/E 三类选做**，非 A 类预估 **13 分**（达标线 9 分）。

## 目录结构
```
2524030232_张锐寒/
├── README.md              本文档
├── 自我介绍.md            个人简介（提交时导出为 自我介绍.pdf）
├── Base_A/  Markdown 基础
├── Base_B/  Git 基础（learngit 练习）
├── Base_C/  Linux 基础
├── Software_A1/  有理数类 Rational（C++ OOP）【必做】
├── Software_A2/  Shape 图形类体系（C++ OOP）【必做】
├── Software_C1/  OpenCV 基础图像处理 + 人脸检测模糊【1分】
├── Software_C2/  YOLO 目标检测脚手架（数据待标注）【部分】
├── Software_C3/  纯 OpenCV 红绿灯状态识别（含干扰鲁棒性）【5分】
├── Software_D1/  强化学习研究笔记（OPD/多模态）【学习文档】
├── Software_D2/  井字棋 Q-Learning 智能体【3分】
├── Software_E1/  numpy 手写 BP 多层感知机【1分】
└── Software_E2/  手写数字识别 MLP vs CNN（PyTorch）【3分】
```

## 各题完成状态与运行
| 题目 | 状态 | 运行 |
|------|------|------|
| A1 有理数类 | ✅ 完整 | g++ Software_A1/有理数类.cpp |
| A2 图形类 | ✅ 完整 | g++ Software_A2/图形设计.cpp |
| C1 OpenCV | ✅ 完整 | python Software_C1/cv_basics.py |
| C2 YOLO | ⚠️ 脚手架 | 需标注后 python train_yolo.py |
| C3 红绿灯 | ✅ 完整+鲁棒性 | python Software_C3/traffic_light_detect.py |
| D1 RL笔记 | ✅ 学习文档 | Software_D1/notes/ |
| D2 井字棋 | ✅ 完整 | python Software_D2/main.py |
| E1 BP-MLP | ✅ 完整 | python Software_E1/test.py |
| E2 MLP vs CNN | ✅ 完整 | python Software_E2/main.py; cnn.py |

## 说明
- 各题均含**学习文档**（记录学习过程、关键知识点、踩坑记录）。
- E2 训练好的模型权重在 `Software_E2/model/`，数据集 `Software_E2/data/`（官方 MNIST，未随 git 提交，本地已备）。
- C2 因数据集无标注且需 GPU 训练，当前交付可运行脚手架，实际训练待标注后补全。
- B 类（ROS）未选做。
