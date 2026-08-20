# main.py —— MLP 全连接网络训练代码
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避本机 OpenMP 重复加载(OMP Error #15)
import torch.nn as nn
from utils import make_loaders, train_model, BASE


class MLP(nn.Module):
    """3 层全连接网络：784 -> 256 -> 256 -> 10，ReLU 激活"""
    def __init__(self, n_in=784, n_out=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_out),
        )

    def forward(self, x):
        return self.net(x)


def main(epochs=5, batch_size=128, lr=1e-2):
    train_loader, test_loader = make_loaders(batch_size=batch_size)
    model = MLP()
    acc = train_model(model, train_loader, test_loader, epochs=epochs,
                      name="MLP", lr=lr,
                      save_path=os.path.join(BASE, "model", "mlp.pth"))
    return acc


if __name__ == "__main__":
    # 可通过环境变量调整轮数，如：EPOCHS=8 python main.py
    main(epochs=int(os.environ.get("EPOCHS", 5)))
