"""
BTL02: Metrics Calculation Module for Facial Emotion Recognition
Provides pure NumPy / PyTorch implementations of:
- Multiclass Accuracy
- Confusion Matrix (Count and Normalized)
- Per-class Precision, Recall, Specificity, F1-Score
- Macro and Weighted Average F1
"""

import numpy as np
import pandas as pd
import torch

from .config import CLASSES, NUM_CLASSES


def compute_accuracy(y_true, y_pred):
    """
    Computes standard top-1 accuracy.
    y_true, y_pred: numpy arrays or torch tensors of shape (N,)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    """
    Computes (num_classes, num_classes) confusion matrix where:
    rows = Ground Truth, columns = Predictions.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_report(y_true, y_pred, class_names=CLASSES):
    """
    Computes per-class precision, recall, F1-score, support,
    and macro / weighted averages. Returns pandas DataFrame and summary dict.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    total_samples = np.sum(cm)

    rows = []
    precisions, recalls, f1s, supports = [], [], [], []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(cm[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        rows.append({
            'Class': name,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'Support': int(support)
        })

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weights = np.array(supports) / total_samples if total_samples > 0 else np.ones(len(supports))
    weighted_p = float(np.sum(np.array(precisions) * weights))
    weighted_r = float(np.sum(np.array(recalls) * weights))
    weighted_f1 = float(np.sum(np.array(f1s) * weights))
    accuracy = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

    df_report = pd.DataFrame(rows)

    summary = {
        'accuracy': round(accuracy, 4),
        'macro_precision': round(macro_p, 4),
        'macro_recall': round(macro_r, 4),
        'macro_f1': round(macro_f1, 4),
        'weighted_precision': round(weighted_p, 4),
        'weighted_recall': round(weighted_r, 4),
        'weighted_f1': round(weighted_f1, 4),
        'total_support': int(total_samples),
        'confusion_matrix': cm
    }

    return df_report, summary
