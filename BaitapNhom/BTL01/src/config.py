# -*- coding: utf-8 -*-
"""
src/config.py
--------------
Cấu hình baseline dùng CHUNG cho toàn bộ dự án BTL01:
- Giai đoạn 2 (Framework cốt lõi)
- Giai đoạn 3 (Thực nghiệm A - Trung Hiếu)
- Giai đoạn 4 (Thực nghiệm B - Thanh Phương)
- Giai đoạn 5 (Thực nghiệm C - Cao Sơn)

Đặc điểm đường dẫn:
Sử dụng pathlib.Path tự động giải quyết đường dẫn gốc dự án (PROJECT_ROOT),
không phụ thuộc máy cá nhân, đảm bảo chạy độc lập trên mọi hệ điều hành (Windows, Linux, macOS).
"""

from pathlib import Path
import torch.nn as nn
import torch.optim as optim

# Đường dẫn gốc của thư mục BTL01 (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Các thư mục chức năng chuẩn hóa theo PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Đảm bảo các thư mục luôn sẵn sàng
for _d in (DATA_DIR, LOGS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Cấu hình Baseline chuẩn hóa xuyên suốt nhóm (đã thống nhất batch_size = 32)
BASELINE = {
    # ----- Dữ liệu / Tái lập -----
    "seed": 42,                          # Seed mặc định cho 1 lần chạy đơn lẻ
    "seeds_for_experiments": [0, 1, 2],  # >= 3 seeds để lấy mean +- std theo đúng Rubric
    "batch_size": 32,                    # Cố định xuyên suốt toàn bộ dự án

    # ----- Kiến trúc MLP Baseline (dùng cho Thực nghiệm A & mốc so sánh cho B, C) -----
    "hidden_layers": [8],                # 1 tầng ẩn, 8 nơ-ron
    "activation": "tanh",                # Tanh cho mặt cong xấp xỉ trơn mịn

    # ----- Tối ưu hóa -----
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "epochs": 300,

    # ----- Tiền xử lý -----
    "normalize_input": True,             # Chuẩn hóa x về mean=0, std=1 (fit CHỈ trên TRAIN)

    # ----- Checkpoint -----
    "checkpoint_dir": CHECKPOINTS_DIR,
    "checkpoint_name": "best_model.pt",
}

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


def get_activation(name: str) -> nn.Module:
    """Ánh xạ tên hàm kích hoạt sang nn.Module."""
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Activation '{name}' chưa được hỗ trợ. Chọn trong: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def get_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    """Ánh xạ tên optimizer sang torch.optim tương ứng."""
    key = str(name).lower()
    if key == "adam":
        return optim.Adam(params, lr=lr)
    if key == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Optimizer '{name}' chưa được hỗ trợ. Chọn 'adam' hoặc 'sgd'.")

