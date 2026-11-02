# -*- coding: utf-8 -*-
"""
main.py
-------
Chương trình thực thi Bài tập lớn 01 - Học sâu.
Đề tài: Xấp xỉ hàm Gaussian 2D bằng mạng nơ-ron nhân tạo MLP trong PyTorch.
Nhóm 01: Đặng Thanh Phương (3123580036), Phạm Trung Hiếu (3123580014), Nguyễn Trương Cao Sơn (3123580040).
Giảng viên hướng dẫn: TS. Đỗ Như Tài - Trường Đại học Sài Gòn (SGU).

Cách sử dụng:
    1. Huấn luyện mô hình tối ưu (Champion Model C=3) và lưu checkpoint:
       python main.py

    2. Sinh lại và kiểm định bộ dữ liệu:
       python main.py --prepare-data

    3. Dựng lại toàn bộ biểu đồ trực quan hóa 3D:
       python main.py --visualize-3d

    4. Chạy lại một hoặc toàn bộ các thực nghiệm (A, B, C):
       python main.py --run-exp A
       python main.py --run-exp B
       python main.py --run-exp C
       python main.py --run-exp all

    5. Tải và đánh giá lại checkpoint đã lưu trên tập test:
       python main.py --eval-checkpoint
"""

import argparse
from pathlib import Path
import os
import sys
import time

# Đảm bảo console in đúng tiếng Việt trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đảm bảo import được module trong src/ độc lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import (
    BASELINE,
    PROJECT_ROOT,
    DATA_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
)
from src.data import (
    MU,
    C,
    SPLIT_SIZES,
    generate_dataset,
    save_csv,
    load_csv,
    gaussian_pdf_numpy,
)
from src.model import MLPRegressor
from src.preprocessing import fit_scaler
from src.trainer import (
    train_model,
    load_best_from_history,
    evaluate_on_test,
)
from src.metrics import evaluate_all
from src.plotting import plot_pred_vs_true
from src.visualize import (
    plot_scatter_3d,
    plot_true_surface_3d,
    plot_compare_surfaces_3d,
    plot_error_surface_3d,
)


def ensure_directories():
    """Tự động khởi tạo các thư mục đầu ra nếu chưa tồn tại."""
    for folder in (DATA_DIR, LOGS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def prepare_datasets(force: bool = False):
    """
    Sinh và kiểm định tính hợp lệ của 3 tập dữ liệu (train_full, val, test).
    Nếu file đã tồn tại và force=False, bỏ qua để bảo đảm tính nhất quán.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1: KIỂM TRA & SINH BỘ DỮ LIỆU GAUSSIAN 2D")
    print("=" * 70)
    ensure_directories()

    splits = ["train_full", "val", "test"]
    all_exist = all((DATA_DIR / f"{s}.csv").exists() for s in splits)

    if all_exist and not force:
        print(">> Toàn bộ file dữ liệu đã tồn tại trong data/:")
        for s in splits:
            path = DATA_DIR / f"{s}.csv"
            x, y = load_csv(path)
            print(f"   - {s}.csv: {len(x)} mẫu | X shape: {x.shape} | y shape: {y.shape}")
        return

    print(">> Đang sinh mới dữ liệu theo quy chuẩn độc lập (SeedSequence/PCG64)...")
    for split_name in splits:
        n = SPLIT_SIZES[split_name]
        x, y = generate_dataset(n_samples=n, split=split_name)
        out_path = DATA_DIR / f"{split_name}.csv"
        save_csv(out_path, x, y)
        print(f"   [OK] Đã lưu {split_name}.csv: {n} mẫu -> {out_path}")


def run_3d_visualization():
    """Tạo các hình ảnh trực quan hóa không gian 3D chuẩn khoa học."""
    print("\n" + "=" * 70)
    print("BƯỚC 2: TRỰC QUAN HÓA DỮ LIỆU VÀ BỀ MẶT HÀM THẬT 3D")
    print("=" * 70)
    ensure_directories()

    # 1. Scatter 3D 2000 mẫu train
    train_path = DATA_DIR / "train_full.csv"
    if not train_path.exists():
        prepare_datasets()
    x_train, y_train = load_csv(train_path)

    fig_scatter = FIGURES_DIR / "scatter_3d_train_samples.png"
    print(f">> Vẽ đám mây điểm 3D tập huấn luyện -> {fig_scatter.name} ...")
    plot_scatter_3d(x_train, y_train, save_path=fig_scatter, sample_limit=2000)

    # 2. Surface 3D giải tích lý thuyết
    fig_surface = FIGURES_DIR / "surface_3d_ground_truth.png"
    print(f">> Vẽ mặt cong giải tích hàm lý thuyết (Ground Truth) -> {fig_surface.name} ...")
    plot_true_surface_3d(save_path=fig_surface)

    print(">> Hoàn thành trực quan hóa 3D ban đầu!")


def train_champion_model(device: str = "cpu"):
    """
    Huấn luyện mô hình Champion được lựa chọn từ Thực nghiệm C:
    - Kiến trúc: 3 tầng ẩn [32, 32, 32] (2.241 tham số)
    - Hàm kích hoạt: Tanh
    - Tối ưu: Adam (lr = 1e-3, batch_size = 32, epochs = 300)
    - Lưu trọng số tối ưu vào checkpoints/best_model.pt
    - Đánh giá trên tập test (100 mẫu) và xuất biểu đồ so sánh.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH TỐI ƯU (CHAMPION MODEL: C=3 [32, 32, 32])")
    print("=" * 70)
    ensure_directories()

    # Nạp dữ liệu
    x_train, y_train = load_csv(DATA_DIR / "train_full.csv")
    x_val, y_val = load_csv(DATA_DIR / "val.csv")
    x_test, y_test = load_csv(DATA_DIR / "test.csv")

    # Chuẩn hóa đặc trưng: chỉ fit trên train
    scaler = fit_scaler(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Khởi tạo mô hình Champion C=3
    champion_architecture = [32, 32, 32]
    model = MLPRegressor(
        input_dim=2,
        hidden_layers=champion_architecture,
        activation=BASELINE["activation"],
        output_dim=1,
    )

    checkpoint_file = CHECKPOINTS_DIR / "best_model.pt"

    print(f">> Kiến trúc mạng: 2 -> {champion_architecture} -> 1")
    print(f">> Tổng số tham số: {model.count_parameters():,}")
    print(f">> Thiết bị huấn luyện: {device}")
    print(f">> Số epoch tối đa: {BASELINE['epochs']} | Batch size: {BASELINE['batch_size']}")
    print(f">> Đang huấn luyện và theo dõi validation...")

    t0 = time.time()
    history = train_model(
        model=model,
        x_train=x_train_scaled,
        y_train=y_train,
        x_val=x_val_scaled,
        y_val=y_val,
        optimizer_name=BASELINE["optimizer"],
        learning_rate=BASELINE["learning_rate"],
        batch_size=BASELINE["batch_size"],
        epochs=BASELINE["epochs"],
        checkpoint_path=checkpoint_file,
        device=device,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Phục hồi trọng số tốt nhất theo val loss
    model = load_best_from_history(model, history, device=device)

    # Đánh giá trên tập test độc lập
    test_metrics = evaluate_on_test(model, x_test_scaled, y_test, device=device)
    train_metrics = evaluate_on_test(model, x_train_scaled, y_train, device=device)

    # Dự đoán trên test để vẽ biểu đồ so sánh
    model.eval()
    with torch.no_grad():
        t_in = torch.tensor(x_test_scaled, dtype=torch.float32, device=device)
        y_test_pred = model(t_in).cpu().numpy()

    fig_pred = FIGURES_DIR / "champion_pred_vs_true.png"
    plot_pred_vs_true(
        y_true=y_test,
        y_pred=y_test_pred,
        title="Mô hình tối ưu MLP [32, 32, 32] trên Tập Test (100 mẫu)",
        save_path=fig_pred,
    )

    # Trực quan hóa bề mặt 3D so sánh và sai số
    fig_best_surf = FIGURES_DIR / "exp_C_best_surface_3d.png"
    plot_compare_surfaces_3d(
        model=model,
        scaler=scaler,
        device=device,
        title_suffix="(C=3, [32, 32, 32])",
        save_path=fig_best_surf,
    )
    fig_err_surf = FIGURES_DIR / "exp_C_error_surface_3d.png"
    plot_error_surface_3d(
        model=model,
        scaler=scaler,
        device=device,
        title_suffix="(C=3, [32, 32, 32])",
        save_path=fig_err_surf,
    )

    print("\n" + "-" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TỐI ƯU (CHAMPION MODEL):")
    print("-" * 70)
    print(f"  • Thời gian huấn luyện:  {elapsed:.2f} giây")
    print(f"  • Best Validation Epoch:  {history.best_epoch} / {BASELINE['epochs']}")
    print(f"  • Best Val Loss (MSE):    {history.best_val_loss:.8e}")
    print(f"  • Train MSE:              {train_metrics['mse']:.8e}")
    print(f"  • Test MSE:               {test_metrics['mse']:.8e}")
    print(f"  • Test MAE:               {test_metrics['mae']:.6f}")
    print(f"  • Test RMSE:              {test_metrics['rmse']:.6f}")
    print(f"  • Hệ số xác định R²:      {test_metrics['r2']:.6f}")
    print(f"  • File trọng số đã lưu:   {checkpoint_file}")
    print(f"  • Biểu đồ dự đoán đã lưu: {fig_pred}")
    print("-" * 70)

    return model, test_metrics


def evaluate_existing_checkpoint(checkpoint_path: Optional[str] = None, device: str = "cpu"):
    """Tải và đánh giá lại checkpoint đã lưu trong checkpoints/best_model.pt (hoặc đường dẫn chỉ định)."""
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ CHECKPOINT MÔ HÌNH ĐÃ LƯU")
    print("=" * 70)

    if checkpoint_path and checkpoint_path != "default":
        checkpoint_file = Path(checkpoint_path)
    else:
        checkpoint_file = CHECKPOINTS_DIR / "best_model.pt"

    if not checkpoint_file.exists():
        print(f">> Chưa có file {checkpoint_file}. Đang tiến hành huấn luyện mới...")
        train_champion_model(device=device)
        return

    # Tải trọng số
    ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
    hidden_layers = ckpt.get("hidden_layers", [32, 32, 32])
    activation = ckpt.get("activation", "tanh")

    model = MLPRegressor(input_dim=2, hidden_layers=hidden_layers, activation=activation, output_dim=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Nạp dữ liệu test
    x_train, _ = load_csv(DATA_DIR / "train_full.csv")
    x_test, y_test = load_csv(DATA_DIR / "test.csv")
    scaler = fit_scaler(x_train)
    x_test_scaled = scaler.transform(x_test)

    metrics = evaluate_on_test(model, x_test_scaled, y_test, device=device)

    print(f">> Tải thành công checkpoint: {checkpoint_file}")
    print(f">> Kiến trúc: 2 -> {hidden_layers} -> 1")
    print(f">> Test MSE:  {metrics['mse']:.8e}")
    print(f">> Test MAE:  {metrics['mae']:.6f}")
    print(f">> Test RMSE: {metrics['rmse']:.6f}")
    print(f">> Test R²:   {metrics['r2']:.6f}")


def execute_experiments(target: str = "all", device: str = "cpu"):
    """Điều phối chạy các thực nghiệm A, B, C."""
    target = target.lower()

    if target in ("a", "all"):
        print("\n" + "=" * 70)
        print("THỰC THI THỰC NGHIỆM A: QUY MÔ DỮ LIỆU HUẤN LUYỆN")
        print("=" * 70)
        from experiments.exp_A_samples import run_experiment_A
        run_experiment_A(project_dir=PROJECT_ROOT, device=device)

    if target in ("b", "all"):
        print("\n" + "=" * 70)
        print("THỰC THI THỰC NGHIỆM B: ĐỘ RỘNG TẦNG ẨN")
        print("=" * 70)
        from experiments.exp_B_width import main as run_B
        run_B(device=device)

    if target in ("c", "all"):
        print("\n" + "=" * 70)
        print("THỰC THI THỰC NGHIỆM C: ĐỘ SÂU MẠNG VÀ TRỰC QUAN HÓA 3D")
        print("=" * 70)
        from experiments.exp_C_depth import main as run_C
        run_C(device=device)


def main():
    parser = argparse.ArgumentParser(
        description="BTL01 - Xấp xỉ hàm Gaussian 2D bằng mạng MLP trong PyTorch (SGU)"
    )
    parser.add_argument("--prepare-data", action="store_true", help="Sinh và kiểm tra 3 tập dữ liệu")
    parser.add_argument("--visualize-3d", action="store_true", help="Vẽ lại các đồ thị trực quan 3D")
    parser.add_argument("--run-exp", choices=["A", "B", "C", "all"], help="Chạy thực nghiệm A, B, C hoặc all")
    parser.add_argument(
        "--eval-checkpoint",
        nargs="?",
        const="default",
        metavar="PATH",
        help="Kiểm tra đánh giá checkpoint (.pt), mặc định checkpoints/best_model.pt",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy (cpu/cuda)")

    args = parser.parse_args()

    ensure_directories()

    # Nếu truyền cờ cụ thể
    if args.prepare_data:
        prepare_datasets(force=True)
        return

    if args.visualize_3d:
        run_3d_visualization()
        return

    if args.run_exp:
        execute_experiments(target=args.run_exp, device=args.device)
        return

    if args.eval_checkpoint is not None:
        evaluate_existing_checkpoint(checkpoint_path=args.eval_checkpoint, device=args.device)
        return

    # Chạy mặc định: Đảm bảo dữ liệu -> Huấn luyện mô hình tối ưu -> Lưu checkpoint -> Báo cáo kết quả
    prepare_datasets(force=False)
    train_champion_model(device=args.device)


if __name__ == "__main__":
    main()
