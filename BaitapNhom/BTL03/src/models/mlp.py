"""
BTL03: Chuyển đổi kiến trúc Multilayer Perceptron (MLP) sang PyTorch
Bao gồm:
1. GaussianMLP: Xấp xỉ hàm mật độ Gaussian 2D (tương đương Func_apxm_MLP.ipynb)
2. MNIST_MLP: Phân loại chữ số viết tay MNIST bằng mạng Dense đa tầng (tương đương MNIST_MLP.ipynb)
"""

import torch
import torch.nn as nn


class GaussianMLP(nn.Module):
    """
    Mạng Perceptron đa tầng xấp xỉ hàm mật độ xác suất Gaussian 2 chiều.
    Kế thừa kiến trúc Keras:
        Input(shape=(2,))
        Dense(units=B, activation='sigmoid')
        Dense(units=1, activation='sigmoid')
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        return out


class MNIST_MLP(nn.Module):
    """
    Mạng Perceptron đa tầng phân loại ảnh chữ số viết tay MNIST 28x28.
    Kế thừa kiến trúc Keras:
        Input(shape=(28, 28)) -> Flatten()
        Dense(512, activation='relu') -> Dropout(0.2)
        Dense(512, activation='relu') -> Dropout(0.2)
        Dense(10) -> Unnormalized logits (CrossEntropyLoss tự tích hợp LogSoftmax)
    """
    def __init__(self, in_features: int = 784, hidden_dim: int = 512, num_classes: int = 10, dropout_rate: float = 0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hỗ trợ cả đầu vào (B, 28, 28) hoặc (B, 1, 28, 28) hoặc (B, 784)
        flat_x = self.flatten(x)
        logits = self.classifier(flat_x)
        return logits
