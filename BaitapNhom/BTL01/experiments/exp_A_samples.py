# -*- coding: utf-8 -*-
"""
experiments/exp_A_samples.py
------------------------------
Giai đoạn 3 - Thực nghiệm A: Khảo sát ảnh hưởng của QUY MÔ DỮ LIỆU HUẤN LUYỆN (Số mẫu A).
Phụ trách chính: Nguyễn Trung Hiếu.

Mục tiêu & Phương pháp:
1. Giữ cố định:
   - Kiến trúc mô hình baseline: MLP 2 tầng ẩn [16, 16], hàm kích hoạt Tanh.
   - Siêu tham số huấn luyện: optimizer Adam, lr = 1e-3, batch_size = 32, epochs = 300.
   - Chuẩn hóa đầu vào fit riêng trên đúng A mẫu train của từng lần chạy.
   - Tập validation (400 mẫu) và tập test độc lập (100 mẫu) giữ cố định xuyên suốt.
2. Biến thiên biến độc lập:
   - Số mẫu train A in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000] (10 mốc).
   - Lấy A mẫu đầu tiên của train_full.csv để tập nhỏ là tập con (subset) thực sự của tập lớn.
3. Đánh giá:
   - Chạy trên 3 random seeds (seeds = [0, 1, 2]) để tính Mean ± Std của MSE, MAE, RMSE.
   - Vẽ đường cong học tập (Learning Curve) theo A.
4. Đầu ra:
   - logs/exp_A_results.csv (kết quả chi tiết từng seed)
   - logs/exp_A_summary.csv (tổng hợp mean ± std)
   - figures/exp_A_learning_curve.png (biểu đồ MSE, MAE, RMSE)
"""

from pathlib import Path
import os
import sys

# Đảm bảo console Windows in đúng ký tự tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thiết lập đường dẫn gốc dự án độc lập môi trường (chạy được trên mọi máy tính)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import BASELINE
from src.data import load_csv, SPLIT_SIZES
from src.model import MLPRegressor
from src.preprocessing import fit_scaler
from src.trainer import train_model, load_best_from_history, evaluate_on_test

A_VALUES = list(range(200, 2001, 200))


def load_data_A(data_dir: Path):
    """Đọc và kiểm tra tính hợp lệ của dữ liệu train_full, val, test."""
    data = {}
    for split in ("train_full", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
        x, y = load_csv(path)
        assert x.shape == (SPLIT_SIZES[split], 2), f"Sai shape x của {split}"
        assert y.shape == (SPLIT_SIZES[split], 1), f"Sai shape y của {split}"
        data[split] = (x, y)
    assert len(data["train_full"][0]) == max(A_VALUES), f"Tập train_full cần {max(A_VALUES)} mẫu"
    assert len(data["test"][0]) == 100, "Tập test phải có đúng 100 mẫu"
    return data


def run_single_A(a_value: int, seed: int, data: dict, device: str = "cpu") -> dict:
    """
    Huấn luyện một mô hình MLP trên A mẫu train với một seed xác định.
    Khôi phục checkpoint có val loss tốt nhất trước khi tính metrics test.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_full, y_full = data["train_full"]
    # Lấy đúng A mẫu đầu tiên để đảm bảo tính phân cấp tập con
    x_train, y_train = x_full[:a_value].copy(), y_full[:a_value].copy()
    x_val, y_val = data["val"][0].copy(), data["val"][1].copy()
    x_test, y_test = data["test"][0].copy(), data["test"][1].copy()

    # Chuẩn hóa đầu vào: fit scaler CHỈ trên A mẫu train để tránh rò rỉ dữ liệu (data leakage)
    if BASELINE["normalize_input"]:
        scaler = fit_scaler(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

    model_A = MLPRegressor(
        input_dim=2,
        hidden_layers=BASELINE["hidden_layers"],
        activation=BASELINE["activation"],
        output_dim=1,
    )

    history_A = train_model(
        model=model_A,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        optimizer_name=BASELINE["optimizer"],
        learning_rate=BASELINE["learning_rate"],
        batch_size=BASELINE["batch_size"],
        epochs=BASELINE["epochs"],
        checkpoint_path=None,
        device=device,
        verbose=False,
    )

    # Đánh giá trên model có validation loss thấp nhất
    model_A = load_best_from_history(model_A, history_A, device=device)
    train_metrics = evaluate_on_test(model_A, x_train, y_train, device=device)
    test_metrics = evaluate_on_test(model_A, x_test, y_test, device=device)

    result = {
        "A": a_value,
        "seed": seed,
        "best_val_loss": history_A.best_val_loss,
        "best_epoch": history_A.best_epoch,
        "training_time_sec": history_A.training_time_sec,
    }
    for metric in ("mse", "mae", "rmse"):
        result[f"train_{metric}"] = train_metrics[metric]
        result[f"test_{metric}"] = test_metrics[metric]

    return result


def summarize_results_A(results: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mean và std qua các random seeds cho từng mốc A."""
    columns = [
        f"{split}_{metric}"
        for split in ("train", "test")
        for metric in ("mse", "mae", "rmse")
    ]
    summary = results.groupby("A")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def plot_results_A(summary: pd.DataFrame, save_path: Path):
    """Vẽ 3 biểu đồ MSE, MAE, RMSE theo số mẫu A kèm thanh sai số mean ± std."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics_info = [
        ("mse", "MSE (Mean Squared Error)"),
        ("mae", "MAE (Mean Absolute Error)"),
        ("rmse", "RMSE (Root Mean Squared Error)"),
    ]

    for ax, (metric, title) in zip(axes, metrics_info):
        for split, color, marker in (("train", "#1f77b4", "o"), ("test", "#ff7f0e", "s")):
            mean_col = f"{split}_{metric}_mean"
            std_col = f"{split}_{metric}_std"
            ax.errorbar(
                summary["A"],
                summary[mean_col],
                yerr=summary[std_col],
                color=color,
                marker=marker,
                capsize=3,
                linewidth=1.8,
                markersize=6,
                label=f"{split.title()} (mean ± std)",
            )
        ax.set_xlabel("A — Số mẫu huấn luyện (Training Samples)", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(A_VALUES)
        ax.tick_params(axis="x", labelsize=8, rotation=25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    fig.suptitle(
        "Thực nghiệm A: Ảnh hưởng của số mẫu huấn luyện A lên sai số xấp xỉ (batch_size=32, 3 seeds)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">> Đã lưu biểu đồ: {save_path}")


def run_experiment_A(project_dir: Path = PROJECT_ROOT, device: str = "cpu"):
    """Thực thi toàn bộ quy trình Thực nghiệm A."""
    project_dir = Path(project_dir)
    assert BASELINE["batch_size"] == 32, "Bắt buộc batch_size=32 theo quy chuẩn dự án."
    seeds = BASELINE["seeds_for_experiments"]
    assert len(seeds) >= 3, "Cần ít nhất 3 random seeds để thống kê mean ± std."

    data_dir = project_dir / "data"
    logs_dir = project_dir / "logs"
    figures_dir = project_dir / "figures"
    for folder in (logs_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM A: QUY MÔ DỮ LIỆU HUẤN LUYỆN")
    print(f"Các mốc A: {A_VALUES}")
    print(f"Seeds: {seeds} | Batch size: {BASELINE['batch_size']} | Thiết bị: {device}")
    print("=" * 70)

    data = load_data_A(data_dir)
    results = []

    for a_val in A_VALUES:
        for seed in seeds:
            res = run_single_A(a_val, seed, data, device=device)
            results.append(res)
        print(f"  [Hoàn thành] A = {a_val:4d} (3 seeds)")

    df_results = pd.DataFrame(results)
    df_summary = summarize_results_A(df_results)

    # Lưu kết quả
    path_res = logs_dir / "exp_A_results.csv"
    path_sum = logs_dir / "exp_A_summary.csv"
    path_fig = figures_dir / "exp_A_learning_curve.png"

    df_results.to_csv(path_res, index=False)
    df_summary.to_csv(path_sum, index=False)
    plot_results_A(df_summary, path_fig)

    print("\n" + "=" * 70)
    print("TỔNG HỢP KẾT QUẢ THỰC NGHIỆM A (TEST MSE MEAN ± STD):")
    for _, row in df_summary.iterrows():
        a_val = int(row["A"])
        test_mse_m = row["test_mse_mean"]
        test_mse_s = row["test_mse_std"]
        print(f"  A = {a_val:4d} | Test MSE = {test_mse_m:.6f} ± {test_mse_s:.6f}")
    print("=" * 70)

    return df_results, df_summary


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment_A(PROJECT_ROOT, device=device)


if __name__ == "__main__":
    main()

