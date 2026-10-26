# -*- coding: utf-8 -*-
"""
src/model.py
------------
Giai đoạn 2 - Kiến trúc mô hình MLP tổng quát (Thanh Phương phụ trách).

Dùng chung cho cả 3 thực nghiệm A, B, C:
- Số lượng tầng ẩn / số nơ-ron mỗi tầng được cấu hình linh hoạt qua tham số `hidden_layers`.
- Đầu ra: Bài toán hồi quy xấp xỉ hàm mật độ xác suất thực, do đó tầng cuối cùng là một
  `nn.Linear` THUẦN TÚY, TUYỆT ĐỐI KHÔNG GẮN ACTIVATION (không Sigmoid/ReLU/Tanh ở output).
"""

from typing import List
import torch.nn as nn
from src.config import get_activation


class MLPRegressor(nn.Module):
    """
    Mạng Multi-Layer Perceptron (MLP) tổng quát cho bài toán hồi quy xấp xỉ hàm.

    Tham số:
    --------
    input_dim: Số chiều đầu vào (mặc định 2 cho (x1, x2)).
    hidden_layers: Danh sách số nơ-ron mỗi tầng ẩn, ví dụ:
                   - [8]: 1 tầng ẩn, 8 nơ-ron (dùng cho TN A và mốc TN B).
                   - [16, 16]: 2 tầng ẩn, mỗi tầng 16 nơ-ron (dùng cho TN C).
    activation: Tên hàm kích hoạt ("relu", "tanh", "sigmoid", "gelu").
    output_dim: Số chiều đầu ra (mặc định 1 cho y).

    Kiến trúc mạng:
    Linear -> Activation -> Linear -> Activation -> ... -> Linear (Output, không activation).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
        output_dim: int = 1,
    ):
        super().__init__()

        if hidden_layers is None or len(hidden_layers) == 0:
            raise ValueError("hidden_layers phải có ít nhất 1 phần tử (MLP cần ít nhất 1 tầng ẩn).")

        self.input_dim = input_dim
        self.hidden_layers_cfg = list(hidden_layers)
        self.activation_name = activation
        self.output_dim = output_dim

        layers = []
        in_features = input_dim
        for n_units in hidden_layers:
            layers.append(nn.Linear(in_features, n_units))
            layers.append(get_activation(activation))
            in_features = n_units

        # Tầng xuất: Linear thuần túy, không activation
        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_layers={self.hidden_layers_cfg}, "
            f"activation={self.activation_name}, output_dim={self.output_dim}"
        )

