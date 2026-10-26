# -*- coding: utf-8 -*-
"""
src/visualize.py
----------------
Module Trực quan hóa Dữ liệu và Mô hình 3D.

Cung cấp các công cụ trực quan hóa chuyên nghiệp đáp ứng Rubric:
1. plot_scatter_3d: Vẽ đám mây điểm dữ liệu phân bố trong không gian 3 chiều (x1, x2, y).
2. plot_true_surface_3d: Vẽ bề mặt cong liên tục lý thuyết của hàm Gaussian 2D (Ground Truth).
3. plot_compare_surfaces_3d: Vẽ song song 2 bề mặt 3D (Hàm thật vs Mô hình MLP dự đoán)
   để so sánh trực quan mức độ xấp xỉ hình học.
4. plot_error_surface_3d: Vẽ bề mặt sai số tuyệt đối |y_that - y_dudoan| để thấy rõ vùng nào mô hình học tốt,
   vùng nào sai số cao (thường ở biên).
5. plot_loss_curves: Vẽ đường cong suy giảm hàm mất mát (Train Loss vs Val Loss) qua các epoch.
"""

from pathlib import Path
import sys
from typing import Optional, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import torch

# Tự động nhận diện thư mục gốc BTL01
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"

from src.data import gaussian_pdf_numpy, LOW, HIGH


def plot_scatter_3d(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Biểu đồ phân bố dữ liệu trong không gian 3D",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ đám mây điểm (Scatter 3D) của tập dữ liệu.
    - Trục X: x1 trong khoảng [-2, 2]
    - Trục Y: x2 trong khoảng [-4, 4]
    - Trục Z: y = f(x1, x2) giá trị mật độ xác suất
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x[:, 0],
        x[:, 1],
        y,
        c=y,
        cmap="viridis",
        s=18,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.2,
    )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label("Mật độ xác suất (Gaussian Value)", fontsize=10)

    ax.set_xlabel("$X_1$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("$X_2$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_zlabel("$y = f(X_1, X_2)$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.view_init(elev=28, azim=-120)  # Góc nhìn tối ưu để thấy rõ đỉnh quả đồi elip

    fig.tight_layout()
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=200, bbox_inches="tight")
        print(f"[Visualize] Đã lưu biểu đồ: {save_p.name}")

    plt.close(fig)
    return fig, ax


def plot_true_surface_3d(
    grid_points: int = 100,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ bề mặt cong 3 chiều lý thuyết của hàm Gaussian 2D (Ground Truth Surface).
    Tạo lưới dày 100x100 điểm để bề mặt cong hoàn toàn trơn mịn.
    """
    x1_lin = np.linspace(LOW[0], HIGH[0], grid_points)
    x2_lin = np.linspace(LOW[1], HIGH[1], grid_points)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    # Ghép lưới thành tensor shape (N, 2) để tính y theo đúng công thức
    grid_coords = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    Y_grid = gaussian_pdf_numpy(grid_coords).reshape(grid_points, grid_points)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X1,
        X2,
        Y_grid,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label("Mật độ $f(X_1, X_2)$", fontsize=10)

    ax.set_xlabel("$X_1$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("$X_2$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_zlabel("Mật độ xác suất", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Bề mặt hàm mật độ Gaussian 2 chiều lý thuyết (Ground Truth)", fontsize=12, fontweight="bold", pad=15)
    ax.view_init(elev=28, azim=-120)

    fig.tight_layout()
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=200, bbox_inches="tight")
        print(f"[Visualize] Đã lưu bề mặt lý thuyết: {save_p.name}")

    plt.close(fig)
    return fig, ax


def plot_compare_surfaces_3d(
    model: torch.nn.Module,
    scaler=None,
    device: str = "cpu",
    grid_points: int = 80,
    title_suffix: str = "",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ song song 2 bề mặt 3D cạnh nhau trên cùng 1 hình (Side-by-Side):
    - Bên trái: Bề mặt thực tế lý thuyết (Ground Truth)
    - Bên phải: Bề mặt do mạng nơ-ron MLP xấp xỉ dự đoán
    Đây là hình ảnh quan trọng nhất để chứng minh mô hình học thành công hình dạng quả đồi Gaussian.
    """
    model.eval()
    x1_lin = np.linspace(LOW[0], HIGH[0], grid_points)
    x2_lin = np.linspace(LOW[1], HIGH[1], grid_points)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    # 1. Tính giá trị thật lý thuyết
    grid_coords = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    Y_true = gaussian_pdf_numpy(grid_coords).reshape(grid_points, grid_points)

    # 2. Cho mô hình MLP dự đoán trên cùng lưới tọa độ
    if scaler is not None:
        grid_input = scaler.transform(grid_coords)
    else:
        grid_input = grid_coords

    with torch.no_grad():
        input_t = torch.as_tensor(grid_input, dtype=torch.float32).to(device)
        Y_pred = model(input_t).cpu().numpy().reshape(grid_points, grid_points)

    # Đồng bộ thang đo Z để so sánh công bằng
    z_min = 0.0
    z_max = max(float(Y_true.max()), float(Y_pred.max())) * 1.05

    fig = plt.figure(figsize=(16, 7))

    # --- Subplot 1: Bề mặt thật ---
    ax1 = fig.add_subplot(121, projection="3d")
    surf1 = ax1.plot_surface(
        X1, X2, Y_true, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.9
    )
    ax1.set_title("Hàm thật (Ground Truth Gaussian 2D)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("$X_1$", fontsize=10, fontweight="bold", labelpad=6)
    ax1.set_ylabel("$X_2$", fontsize=10, fontweight="bold", labelpad=6)
    ax1.set_zlabel("$y$", fontsize=10, fontweight="bold", labelpad=6)
    ax1.set_zlim(z_min, z_max)
    ax1.view_init(elev=28, azim=-120)

    # --- Subplot 2: Bề mặt do MLP xấp xỉ ---
    ax2 = fig.add_subplot(122, projection="3d")
    surf2 = ax2.plot_surface(
        X1, X2, Y_pred, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.9
    )
    ax2.set_title(f"Mô hình MLP xấp xỉ {title_suffix}", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("$X_1$", fontsize=10, fontweight="bold", labelpad=6)
    ax2.set_ylabel("$X_2$", fontsize=10, fontweight="bold", labelpad=6)
    ax2.set_zlabel(r"$\hat{y}$ (Dự đoán)", fontsize=10, fontweight="bold", labelpad=6)
    ax2.set_zlim(z_min, z_max)
    ax2.view_init(elev=28, azim=-120)

    fig.tight_layout()
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=200, bbox_inches="tight")
        print(f"[Visualize] Đã lưu biểu đồ so sánh 2 bề mặt: {save_p.name}")

    plt.close(fig)
    return fig, (ax1, ax2)


def plot_error_surface_3d(
    model: torch.nn.Module,
    scaler=None,
    device: str = "cpu",
    grid_points: int = 80,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Vẽ bề mặt sai số tuyệt đối Error = |y_that - y_dudoan|.
    Giúp phân tích chi tiết: mô hình xấp xỉ tốt ở đỉnh nhưng thường có sai số ở rìa/biên (phục vụ viết Báo cáo).
    """
    model.eval()
    x1_lin = np.linspace(LOW[0], HIGH[0], grid_points)
    x2_lin = np.linspace(LOW[1], HIGH[1], grid_points)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)

    grid_coords = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    Y_true = gaussian_pdf_numpy(grid_coords).reshape(grid_points, grid_points)

    if scaler is not None:
        grid_input = scaler.transform(grid_coords)
    else:
        grid_input = grid_coords

    with torch.no_grad():
        input_t = torch.as_tensor(grid_input, dtype=torch.float32).to(device)
        Y_pred = model(input_t).cpu().numpy().reshape(grid_points, grid_points)

    error_grid = np.abs(Y_true - Y_pred)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X1, X2, error_grid, cmap="magma", linewidth=0, antialiased=True, alpha=0.9
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label(r"Sai số tuyệt đối $|y - \hat{y}|$", fontsize=10)

    ax.set_xlabel("$X_1$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("$X_2$", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_zlabel("Sai số tuyệt đối", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Bề mặt sai số tuyệt đối của mô hình MLP", fontsize=12, fontweight="bold", pad=15)
    ax.view_init(elev=28, azim=-120)

    fig.tight_layout()
    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=200, bbox_inches="tight")
        print(f"[Visualize] Đã lưu biểu đồ bề mặt sai số: {save_p.name}")

    plt.close(fig)
    return fig, ax


if __name__ == "__main__":
    from src.data import load_csv
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> [Demo Visualize] Đang vẽ bề mặt lý thuyết Ground Truth...")
    plot_true_surface_3d(save_path=FIGURES_DIR / "surface_3d_ground_truth.png")

    print(">>> [Demo Visualize] Đang vẽ đám mây điểm Scatter 3D tập train...")
    x_train, y_train = load_csv(DATA_DIR / "train_full.csv")
    plot_scatter_3d(x_train[:500], y_train[:500], title="Phân bố 500 mẫu dữ liệu huấn luyện (Scatter 3D)", save_path=FIGURES_DIR / "scatter_3d_train_samples.png")

    print(">>> Hoàn thành chạy thử nghiệm src/visualize.py!")