# -*- coding: utf-8 -*-
"""
experiments/exp_B_width.py
----------------------------
Giai đoạn 4 - Thực nghiệm B: Khảo sát ảnh hưởng của ĐỘ RỘNG mạng (số nơ-ron B trong 1 tầng ẩn).

Mục tiêu:
- Giữ cố định: Số tầng ẩn = 1, cấu hình baseline (optimizer, lr, batch_size=32, epochs=300).
- Biến thiên: B in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64].
- Chạy lặp lại trên 3 random seeds (seeds = [0, 1, 2]) để lấy giá trị trung bình +- độ lệch chuẩn.
- Ghi nhận Train MSE/MAE, Test MSE/MAE, thời gian chạy và vẽ biểu đồ phân tích Underfitting vs Overfitting.
"""

from pathlib import Path
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Tự động thêm thư mục gốc BTL01 vào sys.path để import các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import BASELINE, DATA_DIR, LOGS_DIR, FIGURES_DIR
from src.data import load_csv
from src.model import MLPRegressor
from src.preprocessing import fit_scaler
from src.trainer import train_model, evaluate_on_test, load_best_from_history
from src.plotting import plot_pred_vs_true

# Danh sách số nơ-ron B cần khảo sát
B_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]


def load_all_data():
    """Đọc 3 tập train_full, val, test từ thư mục data."""
    train_path = DATA_DIR / "train_full.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    for p in (train_path, val_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy file: {p}. Hãy chạy 'python src/data.py' để sinh dữ liệu trước!"
            )

    x_train, y_train = load_csv(train_path)
    x_val, y_val = load_csv(val_path)
    x_test, y_test = load_csv(test_path)
    return x_train, y_train, x_val, y_val, x_test, y_test


def run_single_config(b_value: int, seed: int, data: dict, device: str = "cpu"):
    """Huấn luyện 1 cấu hình cụ thể (B, seed) và đánh giá trên tập test bằng model tốt nhất."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Chuẩn hóa đầu vào (fit CHỈ trên train set để chống rò rỉ dữ liệu)
    if BASELINE["normalize_input"]:
        scaler = fit_scaler(x_train)
        x_train_in = scaler.transform(x_train)
        x_val_in = scaler.transform(x_val)
        x_test_in = scaler.transform(x_test)
    else:
        x_train_in, x_val_in, x_test_in = x_train, x_val, x_test

    model = MLPRegressor(
        input_dim=x_train.shape[1],
        hidden_layers=[b_value],
        activation=BASELINE["activation"],
        output_dim=1,
    )

    history = train_model(
        model,
        x_train_in,
        y_train,
        x_val_in,
        y_val,
        optimizer_name=BASELINE["optimizer"],
        learning_rate=BASELINE["learning_rate"],
        batch_size=BASELINE["batch_size"],
        epochs=BASELINE["epochs"],
        checkpoint_path=None,
        device=device,
        verbose=False,
    )

    # Đánh giá bằng trọng số của epoch tốt nhất theo validation loss
    model = load_best_from_history(model, history, device=device)

    train_metrics = evaluate_on_test(model, x_train_in, y_train, device=device)
    test_metrics = evaluate_on_test(model, x_test_in, y_test, device=device)

    model.eval()
    with torch.no_grad():
        x_test_t = torch.as_tensor(x_test_in, dtype=torch.float32).to(device)
        y_test_pred = model(x_test_t).cpu().numpy().reshape(-1)

    record = {
        "B": b_value,
        "seed": seed,
        "train_mse": train_metrics["mse"],
        "train_mae": train_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "best_val_loss": history.best_val_loss,
        "best_epoch": history.best_epoch,
        "training_time_sec": history.training_time_sec,
    }
    return record, y_test_pred


def plot_width_curve(summary: pd.DataFrame, save_path: Path):
    """Vẽ biểu đồ sai số theo độ rộng B với thanh lỗi sai số chuẩn (errorbar)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["B"],
        summary["train_mse_mean"],
        yerr=summary["train_mse_std"],
        marker="o",
        capsize=4,
        color="#2ca02c",
        label="Train MSE (Mean ± Std)",
        linestyle="-",
    )
    ax.errorbar(
        summary["B"],
        summary["test_mse_mean"],
        yerr=summary["test_mse_std"],
        marker="s",
        capsize=4,
        color="#d62728",
        label="Test MSE (Mean ± Std)",
        linestyle="--",
    )

    ax.set_xlabel("Số nơ-ron tầng ẩn B (Độ rộng mạng)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sai số toàn phương trung bình (MSE)", fontsize=11, fontweight="bold")
    ax.set_title("Thực nghiệm B: Ảnh hưởng của Độ rộng tầng ẩn B đến Sai số xấp xỉ", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Thực nghiệm B] Đã lưu biểu đồ: {save_path.name}")


def run_experiment_B(device: str = "cpu"):
    """Chạy toàn bộ thực nghiệm B qua tất cả các mốc B và các random seeds."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_all_data()
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    results = []
    predictions_cache = {}

    print("=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM B: KHẢO SÁT ĐỘ RỘNG MẠNG (B)")
    print(f"Các mốc B: {B_VALUES}")
    print(f"Seeds: {BASELINE['seeds_for_experiments']} | Batch size: {BASELINE['batch_size']}")
    print("=" * 70)

    for b_value in B_VALUES:
        for seed in BASELINE["seeds_for_experiments"]:
            print(f">> Đang chạy B = {b_value:2d} | Seed = {seed} ...", end="\r")
            record, y_test_pred = run_single_config(b_value, seed, data, device=device)
            results.append(record)
            predictions_cache[(b_value, seed)] = y_test_pred
        print(f">> Hoàn thành B = {b_value:2d} ({len(BASELINE['seeds_for_experiments'])} seeds)")

    df = pd.DataFrame(results)
    detail_csv_path = LOGS_DIR / "exp_B_results.csv"
    df.to_csv(detail_csv_path, index=False)
    print(f"\n[Thực nghiệm B] Đã lưu log chi tiết từng lần chạy: {detail_csv_path.name}")

    # Tính bảng tổng hợp Mean +- Std
    summary = (
        df.groupby("B")
        .agg(
            train_mse_mean=("train_mse", "mean"),
            train_mse_std=("train_mse", "std"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            train_mae_mean=("train_mae", "mean"),
            test_mae_mean=("test_mae", "mean"),
            time_sec_mean=("training_time_sec", "mean"),
        )
        .reset_index()
    )
    summary_csv_path = LOGS_DIR / "exp_B_summary.csv"
    summary.to_csv(summary_csv_path, index=False)
    print(f"[Thực nghiệm B] Đã lưu bảng tổng hợp thống kê: {summary_csv_path.name}")

    # Vẽ biểu đồ sai số theo B
    curve_fig_path = FIGURES_DIR / "exp_B_width_curve.png"
    plot_width_curve(summary, curve_fig_path)

    # Vẽ đồ thị y thật vs y dự đoán cho cấu hình B tốt nhất
    best_b = int(summary.loc[summary["test_mse_mean"].idxmin(), "B"])
    best_seed = BASELINE["seeds_for_experiments"][0]
    pred_fig_path = FIGURES_DIR / "exp_B_best_pred_vs_true.png"
    plot_pred_vs_true(
        data["y_test"],
        predictions_cache[(best_b, best_seed)],
        title=f"Thực nghiệm B: y thật vs ŷ dự đoán (Cấu hình tốt nhất B = {best_b}, Seed = {best_seed})",
        save_path=pred_fig_path,
    )
    print(f"[Thực nghiệm B] Đã lưu biểu đồ y-ŷ cho cấu hình tốt nhất: {pred_fig_path.name}")
    print("\nBẢNG KẾT QUẢ TỔNG HỢP THỰC NGHIỆM B (Mean ± Std):")
    print(summary.to_string(index=False))

    return df, summary


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment_B(device=device)
