import math
import os
import numpy as np


# ============ 通用 numpy 多层感知机（手写反向传播 + Adam，不使用深度学习框架）============
def _fit_mlp(X, Y, hidden=128, max_epochs=6000, lr=0.01,
             beta1=0.9, beta2=0.999, eps=1e-8, tol=2e-5):
    """
    训练 3 层 MLP（输入 -> hidden -> hidden -> 输出），tanh 激活，Adam 优化。
    X: (N, n_in)，Y: (N,)。返回权重元组 (W1,b1,W2,b2,W3,b3)。
    训练到平均绝对误差 < 1e-3 且收敛，或达到最大轮数。
    """
    rng = np.random.RandomState(42)
    Y = Y.reshape(-1, 1)
    n_in = X.shape[1]
    N = X.shape[0]

    # He 初始化
    W1 = rng.randn(n_in, hidden) * np.sqrt(2.0 / n_in); b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, hidden) * np.sqrt(2.0 / hidden); b2 = np.zeros(hidden)
    W3 = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden); b3 = np.zeros(1)

    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]

    prev_err = float('inf')
    for epoch in range(1, max_epochs + 1):
        # 前向传播
        z1 = X @ W1 + b1; a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2; a2 = np.tanh(z2)
        out = a2 @ W3 + b3

        err = out - Y
        avg_err = np.mean(np.abs(err))

        # 反向传播（梯度）
        gW3 = a2.T @ err / N;               gb3 = err.sum(axis=0) / N
        da2 = (err @ W3.T) * (1 - a2 ** 2)
        gW2 = a1.T @ da2 / N;               gb2 = da2.sum(axis=0) / N
        da1 = (da2 @ W2.T) * (1 - a1 ** 2)
        gW1 = X.T @ da1 / N;                gb1 = da1.sum(axis=0) / N
        grads = [gW1, gb1, gW2, gb2, gW3, gb3]

        # Adam 更新
        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** epoch)
            v_hat = v[i] / (1 - beta2 ** epoch)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % 2000 == 0:
            print(f"    epoch {epoch}: 平均绝对误差 = {avg_err:.5f}", flush=True)
        # 收敛判断：误差足够小且不再明显下降
        if avg_err < 1e-3 and abs(prev_err - avg_err) < tol:
            break
        prev_err = avg_err

    return W1, b1, W2, b2, W3, b3


def _predict(x, params):
    """前向传播预测。x: (M, n_in)，返回 (M, 1)"""
    W1, b1, W2, b2, W3, b3 = params
    a1 = np.tanh(x @ W1 + b1)
    a2 = np.tanh(a1 @ W2 + b2)
    return a2 @ W3 + b3


# 一维输入的神经网络实现（拟合 sin(x)，x∈[0,2π]）
class NeuralNetwork1D:
    def __init__(self):
        xs = np.linspace(0, 2 * math.pi, 800)
        ys = np.sin(xs)
        self.params = _fit_mlp(xs.reshape(-1, 1), ys)

    def predict(self, input_x: float) -> float:
        return float(_predict(np.array([[input_x]]), self.params)[0, 0])


# 二维输入的神经网络实现（拟合 sin(x1)*cos(x2)）
class NeuralNetwork2D:
    def __init__(self):
        n = 40
        x1 = np.linspace(0, 2 * math.pi, n)
        x2 = np.linspace(0, 2 * math.pi, n)
        X1, X2 = np.meshgrid(x1, x2)
        xs = np.stack([X1.ravel(), X2.ravel()], axis=1)
        ys = np.sin(X1.ravel()) * np.cos(X2.ravel())
        self.params = _fit_mlp(xs, ys)

    def predict(self, input_x1: float, input_x2: float) -> float:
        return float(_predict(np.array([[input_x1, input_x2]]), self.params)[0, 0])


# 不要改动此类
class Test:
    def __init__(self):
        self.net1 = NeuralNetwork1D()
        self.net2 = NeuralNetwork2D()

    # 按参数个数分派到 1D / 2D 网络（模板原实现同名重载在 Python 中会被后者覆盖，这里修正）
    def output_y(self, *args):
        if len(args) == 1:
            return self.net1.predict(args[0])
        return self.net2.predict(args[0], args[1])

    def testbench(self, num: int):
        sum_error = 0.0
        average_error = 0.0

        if num == 0:
            total = 500
            for i in range(total):
                x = 1.0 * i / total * 2 * math.pi
                y = self.output_y(x)
                sum_error += abs(math.sin(x) - y)
            average_error = sum_error / total
        else:
            total = 20
            for i in range(total):
                for j in range(total):
                    x1 = 1.0 * i / total * 2 * math.pi
                    x2 = 1.0 * j / total * 2 * math.pi
                    y = self.output_y(x1, x2)
                    true_y = math.sin(x1) * math.cos(x2)
                    sum_error += abs(true_y - y)
            average_error = sum_error / (total * total)

        label = "The 2D is " if num else "The 1D is "
        if average_error <= 1e-2:
            print(f"{label}Success! Average: {average_error}")
        else:
            print(f"{label}Failure! Average: {average_error}")


if __name__ == "__main__":
    t = Test()
    t.testbench(0)  # 参数为0或1，参数为0的时候输入1维度，参数为1的时候输入二维
    t.testbench(1)
