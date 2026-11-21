"""
BTL03: Chuyển đổi mô hình phân đoạn ảnh ngữ nghĩa (SegNet Autoencoder) sang PyTorch
Kế thừa chính xác cấu trúc VGG16-style Encoder-Decoder từ sample/examples/5. Others/1_segnet.ipynb.
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_c: int, out_c: int, kernel: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class SegNet(nn.Module):
    """
    Kiến trúc SegNet (5-stage Encoder - Decoder) cho bài toán Semantic Segmentation bàn tay 96x96 -> 7 lớp.
    Đúng 1-1 với 1_segnet.ipynb (hoặc có thể chọn compact=True nếu chạy nhanh trên CPU).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 7, compact: bool = False):
        super().__init__()
        self.compact = compact
        if compact:
            self.encoder = nn.Sequential(
                _conv_bn_relu(in_channels, 32), _conv_bn_relu(32, 32), nn.MaxPool2d(2, 2),
                _conv_bn_relu(32, 64), _conv_bn_relu(64, 64), nn.MaxPool2d(2, 2),
                _conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2, 2),
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(128, 128), _conv_bn_relu(128, 64),
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(64, 64), _conv_bn_relu(64, 32),
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(32, 32),
                nn.Conv2d(32, num_classes, kernel_size=1),
                nn.BatchNorm2d(num_classes)
            )
        else:
            # Đúng 5 tầng Encoder & 5 tầng Decoder như Keras 1_segnet.ipynb
            self.encoder = nn.Sequential(
                # Stage 1: 96 -> 48
                _conv_bn_relu(in_channels, 64), _conv_bn_relu(64, 64), nn.MaxPool2d(2, 2),
                # Stage 2: 48 -> 24
                _conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2, 2),
                # Stage 3: 24 -> 12
                _conv_bn_relu(128, 256), _conv_bn_relu(256, 256), _conv_bn_relu(256, 256), nn.MaxPool2d(2, 2),
                # Stage 4: 12 -> 6
                _conv_bn_relu(256, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
                # Stage 5: 6 -> 3
                _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), nn.MaxPool2d(2, 2),
            )
            self.decoder = nn.Sequential(
                # Stage 5: 3 -> 6
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 512),
                # Stage 4: 6 -> 12
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(512, 512), _conv_bn_relu(512, 512), _conv_bn_relu(512, 256),
                # Stage 3: 12 -> 24
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(256, 256), _conv_bn_relu(256, 256), _conv_bn_relu(256, 128),
                # Stage 2: 24 -> 48
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(128, 128), _conv_bn_relu(128, 64),
                # Stage 1: 48 -> 96
                nn.Upsample(scale_factor=2, mode='nearest'),
                _conv_bn_relu(64, 64),
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.BatchNorm2d(num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.decoder(self.encoder(x))
