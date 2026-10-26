# -*- coding: utf-8 -*-
"""
src/preprocessing.py
---------------------
Giai đoạn 2 - Tiền xử lý và chuẩn hóa dữ liệu (Feature Scaling).

Nguyên tắc quan trọng để tránh rò rỉ dữ liệu (Data Leakage):
Mean và Std CHỈ ĐƯỢC TÍNH (FIT) TRÊN TẬP TRAIN, sau đó áp dụng (transform)
nguyên giá trị đó cho cả tập train, validation và test.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Scaler:
    """Lưu trữ giá trị mean và std tính từ tập train để áp dụng đồng nhất."""

    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        x_scaled = np.asarray(x_scaled, dtype=np.float64)
        return x_scaled * self.std_ + self.mean_


def fit_scaler(x_train: np.ndarray, eps: float = 1e-8) -> Scaler:
    """
    Tính mean và std DUY NHẤT trên tập train.
    `eps` giúp tránh lỗi chia cho 0 nếu phương sai xấp xỉ 0.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    mean_ = x_train.mean(axis=0)
    std_ = x_train.std(axis=0)
    std_ = np.where(std_ < eps, 1.0, std_)
    return Scaler(mean_=mean_, std_=std_)

