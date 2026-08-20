# cnn.py —— CNN 卷积神经网络训练代码
# 将 MLP 的全连接结构改造成卷积结构，利用空间局部性，通常精度更高、收敛更快。
import os
import torch.nn as nn
from utils import make_loaders, train_model, BASE


class CNN(nn.Module):
    """
    简单 LeNet 风格 CNN：
      Conv(1->32) -> ReLU -> MaxPool(2)
      Conv(32->64) -> ReLU -> MaxPool(2)
      Flatten -> FC(64*7*7 -> 128) -> ReLU -> FC(128 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                 # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)   # 784 -> 1x28x28
        x = self.features(x)
        return self.classifier(x)


def main(epochs=5, batch_size=128, lr=1e-2):
    train_loader, test_loader = make_loaders(batch_size=batch_size)
    model = CNN()
    acc = train_model(model, train_loader, test_loader, epochs=epochs,
                      name="CNN", lr=lr,
                      save_path=os.path.join(BASE, "model", "cnn.pth"))
    return acc


if __name__ == "__main__":
    main(epochs=int(os.environ.get("EPOCHS", 5)))
