"""
BTL03: Bộ chỉ số đánh giá toán học và thống kê (Metrics Engine)
Bao gồm:
- Classification: Accuracy, Macro F1, Precision, Recall, Confusion Matrix
- Regression: MSE, RMSE, MAE, R² (Hệ số xác định)
- Segmentation: Mean IoU, Pixel Accuracy
"""

import numpy as np
import torch
from typing import Dict, Tuple


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Tính toán Accuracy, Macro Precision, Macro Recall, Macro F1."""
    acc = float(np.mean(y_true == y_pred))
    
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = (2 * p * r) / max(1e-8, p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return {
        'accuracy': acc,
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'macro_f1': float(np.mean(f1s))
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Tạo ma trận nhầm lẫn kích thước (num_classes, num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán MSE, RMSE, MAE, R2 cho bài toán hồi quy (Gaussian approximation & LSTM)."""
    y_t = y_true.flatten()
    y_p = y_pred.flatten()

    mse = float(np.mean((y_t - y_p) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_t - y_p)))

    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-8, ss_tot)))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compute_segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> Dict[str, float]:
    """Tính Pixel Accuracy và Mean IoU (mIoU) cho bài toán SegNet."""
    y_t = y_true.flatten()
    y_p = y_pred.flatten()

    pixel_acc = float(np.mean(y_t == y_p))
    ious = []

    for c in range(num_classes):
        intersection = np.sum((y_t == c) & (y_p == c))
        union = np.sum((y_t == c) | (y_p == c))
        if union > 0:
            ious.append(intersection / union)

    miou = float(np.mean(ious)) if ious else 0.0
    return {
        'pixel_accuracy': pixel_acc,
        'miou': miou
    }
