# -*- coding: utf-8 -*-
"""
src/plotting.py
-----------------
Giai đoạn 2 - Tiện ích vẽ biểu đồ cơ bản (Thanh Phương phụ trách).

Cung cấp hàm vẽ scatter đồ thị y thật vs y dự đoán theo đường chéo lý tưởng y = x,
đáp ứng tiêu chí đánh giá trực quan hóa mục 3.2 của đề bài.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np


def plot_pred_vs_true(
    y_true,
    y_pred,
    title: str = "Dự đoán y so với giá trị thật y",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ scatter y_true (trục x) vs y_pred (trục y), kèm đường y = x lý tưởng.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="#1f77b4", label="Dự đoán của mô hình")

    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    margin = (hi - lo) * 0.05
    lo -= margin
    hi += margin

    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = ŷ (Lý tưởng)")

    ax.set_xlabel("y thật (Ground Truth)")
    ax.set_ylabel("ŷ dự đoán (MLP Prediction)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=150)

    return fig, ax

