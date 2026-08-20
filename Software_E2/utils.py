# utils.py —— 数据预处理、加载与准确率计算等工具
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_X = os.path.join(BASE, "data", "mnist_x.txt")   # 官方给定数据：784 维像素
DATA_Y = os.path.join(BASE, "data", "mnist_y.txt")   # 官方给定数据：标签 0-9


def load_data(test_size=10000, seed=42):
    """
    读取官方 mnist_x.txt / mnist_y.txt（70000 条）。
    归一化像素到 [0,1]，按 MNIST 惯例前 60000 条训练、后 10000 条测试。
    """
    X = np.loadtxt(DATA_X, dtype=np.float32) / 255.0
    y = np.loadtxt(DATA_Y, dtype=np.int64)
    X_train, y_train = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    return X_train, y_train, X_test, y_test


def make_loaders(batch_size=128, test_size=10000):
    """构建训练/测试 DataLoader"""
    Xtr, ytr, Xte, yte = load_data(test_size=test_size)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def topk_accuracy(model, loader, k=(1, 5)):
    """
    计算 Top-k 准确率。k 为元组，如 k=(1,5) 返回 {1: 0.99, 5: 1.0}。
    Top-k：预测中前 k 个最高概率的类别里包含真实标签即算对。
    """
    device = next(model.parameters()).device
    model.eval()
    topk = max(k)
    corrects = {kk: 0 for kk in k}
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            _, pred = logits.topk(topk, dim=1)          # 取概率最高的 topk 类
            hit = pred.eq(yb.view(-1, 1))               # 每个位置是否命中
            for kk in k:
                corrects[kk] += hit[:, :kk].any(dim=1).sum().item()
            total += yb.size(0)
    return {kk: corrects[kk] / total for kk in k}


def top1_accuracy(model, loader):
    """普通 Top-1 准确率"""
    acc = topk_accuracy(model, loader, k=(1,))[1]
    return acc


def train_model(model, train_loader, test_loader, epochs=5, name="Model",
                lr=1e-3, save_path=None, momentum=0.9):
    """通用训练流程：返回测试集 Top-1 准确率。自动使用 CUDA(如 H200)。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"训练设备: {device}", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
        print(f"{name} epoch {ep}/{epochs}  loss={tot_loss / len(train_loader.dataset):.4f}",
              flush=True)

    acc = top1_accuracy(model, test_loader)
    elapsed = time.time() - t0
    print(f"{name} 测试准确率: {acc * 100:.2f}%  训练耗时: {elapsed:.1f}s", flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"{name} 模型已保存至 {save_path}", flush=True)
    return acc
