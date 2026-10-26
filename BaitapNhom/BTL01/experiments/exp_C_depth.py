# -*- coding: utf-8 -*-
"""
experiments/exp_C_depth.py
----------------------------
Thực nghiệm C: Khảo sát ảnh hưởng của ĐỘ SÂU mạng (Số tầng ẩn C).

Mục tiêu & Phương pháp:
1. Giữ cố định:
   - Số nơ-ron mỗi tầng ẩn: ví dụ 32 nơ-ron/tầng (đảm bảo mỗi tầng đủ năng lực biểu diễn).
   - Hàm kích hoạt: Tanh (theo BASELINE).
   - Optimizer: Adam (lr = 1e-3, batch_size = 32, epochs = 300).
   - Chuẩn hóa đầu vào fit trên train set.
2. Biến thiên biến độc lập:
   - Số tầng ẩn C in [1, 2, 3, 4, 5, 6] (vượt mức tối thiểu 4 tầng của đề bài).
3. Đánh giá khoa học:
   - Chạy trên 3 random seeds (0, 1, 2) để tính Mean +- Std.
   - Đo thời gian huấn luyện (Training Time tính bằng giây).
   - Theo dõi tốc độ hội tụ của Loss theo từng epoch (Loss Convergence).
   - Phân tích bản chất: Tại sao mạng quá sâu (C >= 5) có thể bị chậm và hiện tượng tiêu biến gradient (Vanishing Gradient).
4. Xuất kết quả:
   - logs/exp_C_results.csv (chi tiết từng seed)
   - logs/exp_C_summary.csv (bảng tổng hợp báo cáo)
   - figures/exp_C_depth_error_and_time.png (biểu đồ sai số và thời gian)
   - figures/exp_C_loss_convergence.png (đường cong hội tụ loss)
   - figures/exp_C_best_surface_3d.png (so sánh 3D mặt cong của Best Model vs Ground Truth)
"""

from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import BASELINE, DATA_DIR, LOGS_DIR, FIGURES_DIR
from src.data import load_csv
from src.model import MLPRegressor
from src.preprocessing import fit_scaler
from src.trainer import train_model, evaluate_on_test, load_best_from_history
from src.visualize import plot_compare_surfaces_3d, plot_error_surface_3d

# Danh sách số tầng ẩn C cần khảo sát
C_VALUES = [1, 2, 3, 4, 5, 6]
UNITS_PER_LAYER = 32  # Cố định số nơ-ron mỗi tầng ẩn


def load_all_data():
    """Đọc dữ liệu train_full, val, test từ thư mục data."""
    train_path = DATA_DIR / "train_full.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    for p in (train_path, val_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy file: {p}. Hãy chạy src/data.py trước!")

    x_train, y_train = load_csv(train_path)
    x_val, y_val = load_csv(val_path)
    x_test, y_test = load_csv(test_path)
    return x_train, y_train, x_val, y_val, x_test, y_test


def run_single_depth(c_layers: int, seed: int, data: dict, device: str = "cpu"):
    """
    Huấn luyện 1 cấu hình cụ thể:
    - c_layers: số tầng ẩn
    - Mỗi tầng có UNITS_PER_LAYER nơ-ron
    - seed: random seed kiểm thử
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Chuẩn hóa x fit trên train
    if BASELINE["normalize_input"]:
        scaler = fit_scaler(x_train)
        x_train_in = scaler.transform(x_train)
        x_val_in = scaler.transform(x_val)
        x_test_in = scaler.transform(x_test)
    else:
        scaler = None
        x_train_in, x_val_in, x_test_in = x_train, x_val, x_test

    # Danh sách số nơ-ron cho C tầng ẩn, ví dụ C=3 -> [32, 32, 32]
    hidden_cfg = [UNITS_PER_LAYER] * c_layers

    model = MLPRegressor(
        input_dim=x_train.shape[1],
        hidden_layers=hidden_cfg,
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

    # Nạp lại trọng số tốt nhất theo validation loss
    best_model = load_best_from_history(model, history, device=device)

    train_metrics = evaluate_on_test(best_model, x_train_in, y_train, device=device)
    test_metrics = evaluate_on_test(best_model, x_test_in, y_test, device=device)

    record = {
        "C": c_layers,
        "seed": seed,
        "hidden_layers": str(hidden_cfg),
        "train_mse": train_metrics["mse"],
        "train_mae": train_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "best_val_loss": history.best_val_loss,
        "best_epoch": history.best_epoch,
        "training_time_sec": history.training_time_sec,
    }

    return record, history, best_model, scaler


def plot_depth_performance_and_time(summary: pd.DataFrame, save_path: Path):
    """
    Vẽ biểu đồ gồm 2 trục / 2 khung hình:
    - Khung 1: Test MSE & Train MSE theo số tầng ẩn C (kèm thanh sai số std).
    - Khung 2: Thời gian huấn luyện (Training Time - giây) theo số tầng ẩn C.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Đồ thị 1: Sai số theo độ sâu C
    ax1.errorbar(
        summary["C"],
        summary["train_mse_mean"],
        yerr=summary["train_mse_std"],
        marker="o",
        capsize=4,
        color="#2ca02c",
        label="Train MSE (Mean ± Std)",
        linestyle="-",
    )
    ax1.errorbar(
        summary["C"],
        summary["test_mse_mean"],
        yerr=summary["test_mse_std"],
        marker="s",
        capsize=4,
        color="#d62728",
        label="Test MSE (Mean ± Std)",
        linestyle="--",
    )
    ax1.set_xlabel("Số tầng ẩn $C$ (Độ sâu mạng)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sai số (MSE)", fontsize=11, fontweight="bold")
    ax1.set_title("Sai số xấp xỉ theo Độ sâu mạng $C$", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Đồ thị 2: Thời gian huấn luyện theo C
    ax2.plot(
        summary["C"],
        summary["time_sec_mean"],
        marker="^",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        label="Thời gian train trung bình (s)",
    )
    ax2.set_xlabel("Số tầng ẩn $C$ (Độ sâu mạng)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Thời gian huấn luyện (giây)", fontsize=11, fontweight="bold")
    ax2.set_title("Thời gian huấn luyện theo Độ sâu mạng $C$", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Thực nghiệm C] Đã lưu biểu đồ sai số và thời gian: {save_path.name}")


def plot_loss_convergence(loss_histories: dict, save_path: Path):
    """
    Vẽ đường cong suy giảm hàm mất mát qua 300 epochs của các độ sâu C khác nhau (Seed = 0).
    Thể hiện tốc độ hội tụ (Loss Convergence) theo yêu cầu của thầy.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (c_val, hist) in enumerate(loss_histories.items()):
        color = colors[idx % len(colors)]
        ax.plot(hist.train_loss, label=f"$C = {c_val}$ tầng", color=color, alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Train Loss (MSE)", fontsize=11, fontweight="bold")
    ax.set_yscale("log")  # Thang log để nhìn rõ chênh lệch tốc độ hội tụ ở các epoch cuối
    ax.set_title("Tốc độ hội tụ của Loss theo từng Epoch giữa các độ sâu $C$ (Thang Log)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Thực nghiệm C] Đã lưu biểu đồ hội tụ loss: {save_path.name}")


def run_experiment_C(device: str = "cpu"):
    """Hàm điều phối toàn bộ Thực nghiệm C."""
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
    loss_histories = {}
    best_overall_model = None
    best_overall_scaler = None
    best_overall_mse = float("inf")
    best_overall_c = 1

    print("=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM C: KHẢO SÁT ĐỘ SÂU MẠNG (C)")
    print(f"Các mốc số tầng ẩn C: {C_VALUES}")
    print(f"Số nơ-ron mỗi tầng: {UNITS_PER_LAYER} | Batch size: {BASELINE['batch_size']}")
    print(f"Seeds: {BASELINE['seeds_for_experiments']}")
    print("=" * 70)

    for c_val in C_VALUES:
        for seed in BASELINE["seeds_for_experiments"]:
            print(f">> Đang chạy C = {c_val} tầng | Seed = {seed} ...", end="\r")
            record, history, model, scaler = run_single_depth(c_val, seed, data, device=device)
            results.append(record)

            # Lưu lại history của seed đầu tiên để vẽ biểu đồ so sánh hội tụ
            if seed == BASELINE["seeds_for_experiments"][0]:
                loss_histories[c_val] = history

            # Theo dõi mô hình có Test MSE thấp nhất toàn bộ thực nghiệm C
            if record["test_mse"] < best_overall_mse:
                best_overall_mse = record["test_mse"]
                best_overall_c = c_val
                best_overall_model = model
                best_overall_scaler = scaler

        print(f">> Hoàn thành C = {c_val} tầng ({len(BASELINE['seeds_for_experiments'])} seeds)")

    df = pd.DataFrame(results)
    detail_csv = LOGS_DIR / "exp_C_results.csv"
    print(f"\n[DEBUG] Đang lưu log chi tiết vào: {detail_csv.resolve()}")
    df.to_csv(str(detail_csv), index=False)
    print(f"[Thực nghiệm C] Đã lưu log chi tiết từng lần chạy: {detail_csv.name}")

    summary = (
        df.groupby("C")
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
    summary_csv = LOGS_DIR / "exp_C_summary.csv"
    print(f"[DEBUG] Đang lưu summary vào: {summary_csv.resolve()}")
    summary.to_csv(str(summary_csv), index=False)
    print(f"[Thực nghiệm C] Đã lưu bảng tổng hợp thống kê: {summary_csv.name}")

    # 1. Vẽ biểu đồ sai số và thời gian huấn luyện
    depth_fig = FIGURES_DIR / "exp_C_depth_error_and_time.png"
    plot_depth_performance_and_time(summary, depth_fig)

    # 2. Vẽ biểu đồ tốc độ hội tụ loss
    conv_fig = FIGURES_DIR / "exp_C_loss_convergence.png"
    plot_loss_convergence(loss_histories, conv_fig)

    # 3. Vẽ biểu đồ 3D so sánh Bề mặt thật vs Mô hình tốt nhất (Module của Cao Sơn)
    if best_overall_model is not None:
        surface_fig = FIGURES_DIR / "exp_C_best_surface_3d.png"
        plot_compare_surfaces_3d(
            best_overall_model,
            scaler=best_overall_scaler,
            device=device,
            title_suffix=f"(Best Model: C={best_overall_c} tầng, 32 units/tầng)",
            save_path=surface_fig,
        )

        error_fig = FIGURES_DIR / "exp_C_error_surface_3d.png"
        plot_error_surface_3d(
            best_overall_model,
            scaler=best_overall_scaler,
            device=device,
            save_path=error_fig,
        )

    print("\nBẢNG KẾT QUẢ TỔNG HỢP THỰC NGHIỆM C (Mean ± Std):")
    print(summary.to_string(index=False))

    return df, summary


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment_C(device=device)