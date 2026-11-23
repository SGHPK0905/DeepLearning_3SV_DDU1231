"""
BTL03: Module trực quan hóa đồ họa học thuật chất lượng cao (Visualization Engine)
Xuất đồ thị độ phân giải cao (DPI=300):
1. Đường cong hội tụ Loss & Accuracy (Learning Curves)
2. Bề mặt 3D xấp xỉ hàm Gaussian (Ground Truth vs PyTorch Prediction)
3. Ma trận nhầm lẫn (Confusion Matrix)
4. Phân đoạn bàn tay SegNet dùng bảng màu class_to_rgb chuẩn 1_segnet.ipynb
5. Heatmap & Bounding Boxes phát hiện xe ô tô trên test.jpg bằng locate_cars()
6. Biểu đồ so sánh hiệu năng thực nghiệm các mô hình PyTorch
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import torch
from PIL import Image

from .config import FIGURES_DIR, SAMPLE_DIR, DEVICE

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


def segnet_class_to_rgb(mask_hw: np.ndarray) -> np.ndarray:
    """Chuyển mặt nạ lớp 0..6 sang ảnh RGB theo đúng bảng màu trong 1_segnet.ipynb."""
    h, w = mask_hw.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    palette = {
        0: [0.0, 0.0, 0.0],
        1: [0.0, 0.0, 0.5],
        2: [0.0, 0.5, 0.0],
        3: [0.5, 0.0, 0.0],
        4: [0.5, 0.0, 0.5],
        5: [0.5, 0.5, 0.0],
        6: [1.0, 1.0, 1.0],
    }
    for cls_id, color in palette.items():
        rgb[mask_hw == cls_id] = color
    return rgb


def plot_learning_curves(
    history: Dict[str, list],
    model_name: str,
    save_name: str,
    is_regression: bool = False
) -> Path:
    """Vẽ biểu đồ đường cong hàm mất mát và độ chính xác qua các epoch."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    line1 = ax1.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss')
    line2 = ax1.plot(epochs, history['val_loss'], 'r--', lw=2, label='Val Loss')
    ax1.set_xlabel('Epochs', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', color='#1e293b', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    if not is_regression and 'train_acc' in history and len(history['train_acc']) > 0:
        ax2 = ax1.twinx()
        line3 = ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'g-.', lw=1.5, label='Train Acc (%)')
        line4 = ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'm:', lw=2, label='Val Acc (%)')
        ax2.set_ylabel('Accuracy (%)', color='#047857', fontsize=11, fontweight='bold')
        lines += line3 + line4
        labels += [l.get_label() for l in line3 + line4]

    plt.title(f'Đường cong huấn luyện PyTorch — {model_name}', fontsize=12, fontweight='bold', pad=12)
    ax1.legend(lines, labels, loc='best', framealpha=0.9)
    plt.tight_layout()

    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_gaussian_3d_comparison(
    model: torch.nn.Module,
    save_name: str = "gaussian_3d_comparison.png",
    device: torch.device = DEVICE
) -> Path:
    """Vẽ bề mặt 3D so sánh hàm Gaussian lý thuyết và mô hình PyTorch GaussianMLP."""
    from scipy.stats import multivariate_normal

    x1, x2 = np.mgrid[-3.0:3.0:0.1, -3.0:3.0:0.1]
    grid_pts = np.column_stack([x1.flat, x2.flat]).astype(np.float32)

    mu = np.array([0.0, 0.0])
    cov = np.diag([1.0, 1.0])
    z_true = multivariate_normal.pdf(grid_pts, mean=mu, cov=cov).reshape(x1.shape)

    model.eval()
    with torch.no_grad():
        tensor_pts = torch.tensor(grid_pts, dtype=torch.float32).to(device)
        z_pred = model(tensor_pts).cpu().numpy().reshape(x1.shape)

    z_err = np.abs(z_true - z_pred)

    fig = plt.figure(figsize=(15, 4.8), dpi=300)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(x1, x2, z_true, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('(A) Ground Truth Gaussian 2D', fontsize=10, fontweight='bold')
    ax1.set_zlim(0, 0.2)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(x1, x2, z_pred, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.set_title('(B) PyTorch GaussianMLP', fontsize=10, fontweight='bold')
    ax2.set_zlim(0, 0.2)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(x1, x2, z_err, cmap='coolwarm', edgecolor='none', alpha=0.9)
    ax3.set_title('(C) Sai số tuyệt đối |y - ŷ|', fontsize=10, fontweight='bold')

    plt.tight_layout()
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    save_name: str = "confusion_matrix.png"
) -> Path:
    """Trực quan hóa ma trận nhầm lẫn chuẩn xuất bản."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='Thực tế (Ground Truth)',
        xlabel='Dự đoán (Predicted)'
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_segnet_results(
    model: torch.nn.Module,
    loader,
    save_name: str = "segnet_prediction_sample.png",
    device: torch.device = DEVICE
) -> Path:
    """Trực quan hóa kết quả phân đoạn bàn tay mức sâu SegNet với bảng màu RGB chuẩn."""
    model.eval()
    images, masks = next(iter(loader))
    with torch.no_grad():
        preds = model(images.to(device))
        pred_masks = torch.argmax(preds, dim=1).cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=300)
    for idx in range(2):
        axes[idx, 0].imshow(images[idx, 0].numpy(), cmap='gray')
        axes[idx, 0].set_title(f'Sample {idx+1}: Depth Image (96x96)')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(segnet_class_to_rgb(masks[idx].numpy()))
        axes[idx, 1].set_title(f'Sample {idx+1}: Ground Truth Mask')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(segnet_class_to_rgb(pred_masks[idx]))
        axes[idx, 2].set_title(f'Sample {idx+1}: PyTorch SegNet')
        axes[idx, 2].axis('off')

    plt.suptitle('Phân đoạn ngữ nghĩa ảnh bàn tay Kinect bằng PyTorch SegNet', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_car_detection_sample(
    model,
    save_name: str = "car_detection_sample.png",
    threshold: float = 0.85
) -> Path:
    """Trực quan hóa Heatmap, vùng phát hiện và bounding box trên test.jpg."""
    test_img_path = SAMPLE_DIR / "examples" / "5. Others" / "test.jpg"
    raw_img = np.array(Image.open(test_img_path).convert('RGB'), dtype=np.uint8)

    hm, sel_hm, annotated = model.locate_cars(raw_img, threshold=threshold)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    axes[0].imshow(hm, cmap='hot')
    axes[0].set_title('(A) Heatmap FCN (tanh [-1, 1])', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sel_hm, cmap='gray')
    axes[1].set_title(f'(B) Vùng ngưỡng (hm > {threshold})', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(annotated)
    axes[2].set_title('(C) Bounding Boxes trên test.jpg', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_pytorch_benchmark_summary(
    benchmark_data: List[Dict],
    save_name: str = "pytorch_benchmark_summary.png"
) -> Path:
    """Vẽ biểu đồ tóm tắt hiệu năng thực đo các mô hình PyTorch đã chuyển đổi."""
    models = [item['model'] for item in benchmark_data]
    scores = [item['score_pct'] for item in benchmark_data]
    times = [item['train_time_s'] for item in benchmark_data]

    x = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    bars1 = ax1.bar(x, scores, color='#2563eb', alpha=0.9)
    ax1.set_ylabel('Chỉ số hiệu năng (%)', fontsize=10, fontweight='bold')
    ax1.set_title('Hiệu năng thực đo các mô hình PyTorch (Test / Val)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=8.5)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
    for rect in bars1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    bars2 = ax2.bar(x, times, color='#10b981', alpha=0.9)
    ax2.set_ylabel('Thời gian huấn luyện (giây)', fontsize=10, fontweight='bold')
    ax2.set_title('Thời gian huấn luyện thực đo trên thiết bị hiện tại', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    for rect in bars2:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = FIGURES_DIR / save_name
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
