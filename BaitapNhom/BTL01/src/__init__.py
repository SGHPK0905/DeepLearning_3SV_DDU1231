# -*- coding: utf-8 -*-
"""
BTL01 - Deep Learning (Function Approximation using MLP)
Trường Đại học Sài Gòn (SGU) - Khoa Công nghệ Thông tin
Học phần: Học sâu (858048) - TS. Đỗ Như Tài hướng dẫn

Gói mã nguồn `src` cung cấp toàn bộ nền tảng:
- `config`: Thiết lập siêu tham số Baseline thống nhất (batch_size=32, lr=1e-3, Tanh, Adam)
- `data`: Công thức hàm mật độ xác suất Gaussian 2D, sinh và đọc dữ liệu độc lập
- `model`: Lớp MLPRegressor tổng quát kế thừa torch.nn.Module với số tầng ẩn động
- `trainer`: Vòng lặp huấn luyện mini-batch, theo dõi validation và cơ chế checkpoint
- `metrics`: Các hàm đo lường sai số hồi quy: MSE, MAE, RMSE, R2
- `preprocessing`: Bộ chuẩn hóa đặc trưng StandardScaler (chống rò rỉ dữ liệu)
- `plotting`: Tiện ích vẽ biểu đồ hồi quy y thật vs y dự đoán
- `visualize`: Trực quan hóa dữ liệu và mặt cong 3D (Ground Truth, Scatter, Best Surface, Residual)
"""

from .config import (
    BASELINE,
    PROJECT_ROOT,
    DATA_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
)
from .data import (
    MU,
    C,
    LOW,
    HIGH,
    ROOT_SEED,
    SPLIT_SIZES,
    gaussian_pdf_numpy,
    gaussian_pdf_torch,
    generate_dataset,
    save_csv,
    load_csv,
    load_torch_csv,
)
from .model import MLPRegressor
from .trainer import (
    train_model,
    evaluate_on_test,
    load_best_checkpoint,
    load_best_from_history,
    TrainHistory,
)
from .metrics import (
    mse,
    mae,
    rmse,
    r2_score,
    evaluate_all,
)
from .preprocessing import (
    Scaler,
    fit_scaler,
)
from .plotting import (
    plot_pred_vs_true,
)

__version__ = "1.0.0"
__author__ = "Nhóm 01 - Đặng Thanh Phương, Phạm Trung Hiếu, Nguyễn Trương Cao Sơn"

__all__ = [
    "BASELINE",
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOGS_DIR",
    "FIGURES_DIR",
    "CHECKPOINTS_DIR",
    "MU",
    "C",
    "LOW",
    "HIGH",
    "ROOT_SEED",
    "SPLIT_SIZES",
    "gaussian_pdf_numpy",
    "gaussian_pdf_torch",
    "generate_dataset",
    "save_csv",
    "load_csv",
    "load_torch_csv",
    "MLPRegressor",
    "train_model",
    "evaluate_on_test",
    "load_best_checkpoint",
    "load_best_from_history",
    "TrainHistory",
    "mse",
    "mae",
    "rmse",
    "r2_score",
    "evaluate_all",
    "Scaler",
    "fit_scaler",
    "plot_pred_vs_true",
]
