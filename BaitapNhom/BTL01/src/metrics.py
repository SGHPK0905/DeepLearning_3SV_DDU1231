# -*- coding: utf-8 -*-
"""
src/metrics.py
---------------
Giai đoạn 2 - Các hàm đo lường sai số cho bài toán hồi quy (Thanh Phương phụ trách).

Dùng chung cho toàn bộ các thực nghiệm A, B, C và đánh giá trên tập test:
- MSE: Mean Squared Error (chỉ số chính bắt buộc)
- MAE: Mean Absolute Error (chỉ số bổ sung)
- RMSE: Root Mean Squared Error (chỉ số bổ sung)
- R^2: Hệ số xác định (đánh giá mức độ giải thích phương sai)
"""

from typing import Dict, Union
import numpy as np
import torch

ArrayLike = Union[torch.Tensor, np.ndarray, list]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.reshape(-1).astype(np.float64)


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error (MSE)."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean((yt - yp) ** 2))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error (MAE)."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Hệ số xác định R^2. Trả về nan nếu phương sai của y_true bằng 0."""
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    """Trả về từ điển đầy đủ các chỉ số đánh giá."""
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

