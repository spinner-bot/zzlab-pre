# top5_acc.py —— 计算已训练模型(mlp.pth/cnn.pth)的 Top-1 与 Top-5 准确率
# 可独立运行，直接调用 utils 加载数据与模型。
#
# Top-k 准确率含义：
#   模型对每个样本输出 10 个类别的概率，取概率最高的前 k 个类别；
#   若真实标签落在前 k 个预测中，则该样本记为命中。
#   - Top-1：常规准确率，只看预测概率最高的那一个类别；
#   - Top-5：看前 5 个预测，容忍模型把正确答案排在第 2~5 名的情况。
#   Top-5 通常高于 Top-1，因为分类器往往“把正确答案排得比较靠前但未必第一”。
import os
import torch
from utils import make_loaders, topk_accuracy, BASE
from main import MLP
from cnn import CNN


def main():
    train_loader, test_loader = make_loaders(batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载两个已训练模型
    models = {}
    for name, Cls, file in [("MLP", MLP, "mlp.pth"), ("CNN", CNN, "cnn.pth")]:
        path = os.path.join(BASE, "model", file)
        if not os.path.exists(path):
            print(f"[警告] 未找到 {path}，请先运行 main.py / cnn.py 训练。")
            continue
        model = Cls().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        models[name] = model
        print(f"已加载 {name} 模型: {path}")

    if not models:
        return

    print("\n================ Top-k 准确率 ================")
    for name, model in models.items():
        res = topk_accuracy(model, test_loader, k=(1, 5))
        print(f"{name}:  Top-1 = {res[1]*100:.2f}%   Top-5 = {res[5]*100:.2f}%")

    print("\n【分析】")
    print("CNN 借助卷积核提取局部特征(边缘、笔画结构)，参数共享+局部感受野使其"),
    print("在同等或更少参数下精度更高、更抗过拟合；MLP 全连接对每个像素独立加权，"),
    print("需大量参数且忽略空间结构。通常 CNN 的 Top-1 与 Top-5 均高于 MLP。")


if __name__ == "__main__":
    main()
