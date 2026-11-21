"""
BTL03: Chuyển đổi mô hình tích hợp CNN + LSTM sang PyTorch
Kế thừa chính xác cấu trúc từ sample/examples/3. CNN + LSTM/MNIST_CNN_LSTM.ipynb:
    3 khối TimeDistributed Conv(64 -> 128 -> 256) + MaxPool(2,2) + Dropout(0.2)
    -> TimeDistributed Flatten (256*3*3 = 2304)
    -> LSTM(128, return_sequences=True) -> LSTM(128, return_sequences=False)
    -> Dropout(0.22) -> Linear(1)
"""

import torch
import torch.nn as nn


class MNIST_CNN_LSTM(nn.Module):
    """
    Mạng kết hợp CNN và LSTM dự đoán tổng giá trị chuỗi ảnh chữ số MNIST.
    Sử dụng kỹ thuật Batch Folding (B*T, C, H, W) thay thế TimeDistributed trong Keras.
    """
    def __init__(self, in_channels: int = 1, lstm_hidden: int = 128):
        super().__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28 -> 14
            nn.Dropout2d(p=0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14 -> 7
            nn.Dropout2d(p=0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 7 -> 3
            nn.Dropout2d(p=0.2),

            nn.Flatten()  # 256 * 3 * 3 = 2304
        )

        self.lstm1 = nn.LSTM(
            input_size=256 * 3 * 3,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.22)
        self.regressor = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, T, 1, H, W)

        batch_size, time_steps, c, h, w = x.shape
        x_reshaped = x.reshape(batch_size * time_steps, c, h, w)
        cnn_features = self.cnn_encoder(x_reshaped)  # (B*T, 2304)
        seq_features = cnn_features.view(batch_size, time_steps, -1)

        out1, _ = self.lstm1(seq_features)
        _, (hn2, _) = self.lstm2(out1)
        out = self.dropout(hn2[-1])
        return self.regressor(out).squeeze(-1)
