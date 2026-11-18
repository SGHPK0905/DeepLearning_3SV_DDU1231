"""
BTL03: Chuyển đổi các kiến trúc Convolutional Neural Network (CNN) sang PyTorch
Ánh xạ chính xác 1-1 cấu trúc từng lớp từ các notebook Keras gốc:
1. MNIST_CNN: 3 khối Conv(64 -> 128 -> 256) + MaxPool + Dropout(0.2) + FC(512 -> 512 -> 10)
2. Gender_CNN: 5 khối VGG-style Conv(16 -> 32 -> 64 -> 128 -> 256) + FC(1024 -> 1024 -> 2)
3. Emotion_CNN: 2 khối Conv(64-64 same/valid -> 128-128-218-218 same/valid) + FC(1024 -> 7)
"""

import torch
import torch.nn as nn


class MNIST_CNN(nn.Module):
    """
    Mô hình CNN nhận dạng chữ số MNIST (28x28 mức xám).
    Kế thừa chính xác cấu trúc từ sample/examples/2. CNN + MLP/1_MNIST_CNN.ipynb:
        Block 1: Conv2d(1, 64, 3, same) -> ReLU -> MaxPool(2) -> Dropout(0.2)   [28->14]
        Block 2: Conv2d(64, 128, 3, same) -> ReLU -> MaxPool(2) -> Dropout(0.2) [14->7]
        Block 3: Conv2d(128, 256, 3, same) -> ReLU -> MaxPool(2) -> Dropout(0.2)[7->3]
        Classifier: Flatten(256*3*3=2304) -> Linear(512) -> ReLU -> Dropout(0.2)
                    -> Linear(512) -> ReLU -> Dropout(0.2) -> Linear(10)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        self.features = nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        pool3_out = self.features(x)
        fc1_out = self.fc1(pool3_out)
        fc2_out = self.fc2(fc1_out)
        logits = self.head(fc2_out)
        if return_features:
            gap_pool3 = pool3_out.mean(dim=(2, 3))
            return logits, fc2_out, gap_pool3
        return logits


class Gender_CNN(nn.Module):
    """
    Mô hình CNN phân loại giới tính khuôn mặt (64x64 mức xám).
    Kế thừa chính xác cấu trúc 5 khối VGG từ sample/examples/2. CNN + MLP/2_Human_Gender_CNN.ipynb:
        16(x2) -> Pool -> 32(x3) -> Pool -> 64(x3) -> Pool -> 128(x3) -> Pool -> 256(x3) -> Pool
        64 -> 32 -> 16 -> 8 -> 4 -> 2 (Flatten: 256 * 2 * 2 = 1024)
        FC: 1024 -> 1024 -> 1024 -> 2
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 2: 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 3: 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 4: 8 -> 4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            # Block 5: 4 -> 2
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


class Emotion_CNN(nn.Module):
    """
    Mô hình CNN nhận dạng 7 biểu cảm cảm xúc FER-2013 (48x48 mức xám).
    Kế thừa chính xác cấu trúc từ sample/examples/2. CNN + MLP/3_Emotion_Reg_CNN.ipynb:
        Block 1: Conv(64, same) -> Conv(64, valid) -> Pool(2) -> Dropout(0.25) [48->46->23]
        Block 2: Conv(128, same) -> Conv(128, valid) -> Conv(218, same) -> Conv(218, valid)
                 -> Pool(2) -> Dropout(0.25) [23->21->19->9]
        Classifier: Flatten(218 * 9 * 9 = 17658) -> Linear(1024) -> Dropout(0.5) -> Linear(7)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 48 -> 48 -> 46 -> 23
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            # Block 2: 23 -> 23 -> 21 -> 21 -> 19 -> 9
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(128, 218, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(218, 218, kernel_size=3, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(218 * 9 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))
