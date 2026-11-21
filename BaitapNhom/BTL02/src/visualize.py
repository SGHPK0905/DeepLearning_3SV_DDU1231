"""
BTL02: High-Quality Visualization Module for Facial Emotion Recognition (FER2013)
Produces all academic figures required by instructor:
1. Dataset Distribution Histogram (Train / Valid / Test across 7 classes)
2. Training and Validating Image Sample Grids (4x8 grid = 32 images)
3. Learning Curves (Loss & Accuracy across epochs)
4. Confusion Matrix (Normalized heatmap with percentages)
5. True Prediction Cases (High-confidence correct samples)
6. Wrong Prediction Cases (Misclassified samples with true/pred labels & confidence)
7. Final Convolution Filters Visualization (Required by slide 5)
8. Feature Maps / Activations Visualization (Visualizing feature extraction across blocks)
9. Experiment Comparison Charts (A/B/C/D series)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive headless backend
import matplotlib.pyplot as plt
import torch

from .config import CLASSES, CLASSES_VI, FIGURES_DIR


# Styling setup
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def plot_dataset_distribution(y_train, y_val, y_test, class_names=CLASSES, save_name="dataset_distribution_histogram.png"):
    """
    Plots grouped bar chart of class counts across Train, Validation, and Test sets.
    """
    save_path = FIGURES_DIR / save_name
    num_classes = len(class_names)
    indices = np.arange(num_classes)
    bar_width = 0.25

    train_counts = [np.sum(y_train == i) for i in range(num_classes)]
    val_counts = [np.sum(y_val == i) for i in range(num_classes)]
    test_counts = [np.sum(y_test == i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(indices - bar_width, train_counts, bar_width, label=f'Training ({len(y_train):,})', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(indices, val_counts, bar_width, label=f'PublicTest / Validation ({len(y_val):,})', color='#ff7f0e', alpha=0.9)
    rects3 = ax.bar(indices + bar_width, test_counts, bar_width, label=f'PrivateTest / Test ({len(y_test):,})', color='#2ca02c', alpha=0.9)

    ax.set_title("FER-2013 Dataset Distribution by Emotion Class", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Emotion Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Images", fontsize=11, labelpad=8)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{c}\n({v})" for c, v in zip(class_names, CLASSES_VI)], fontsize=9)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Value labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            if h > 0:
                ax.annotate(f"{h}",
                            xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_image_samples(images, labels, class_names=CLASSES, rows=4, cols=8, title="Sample Images", save_name="trainining_images.png"):
    """
    Plots a rows x cols grid of face images with emotion labels (matching demo_report format).
    """
    save_path = FIGURES_DIR / save_name
    num_samples = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6), dpi=300)

    for i in range(num_samples):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        if i < len(images):
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            ax.imshow(img, cmap='gray')
            lbl_idx = int(labels[i])
            ax.set_title(class_names[lbl_idx], fontsize=8, pad=3)
        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_learning_curves(history, save_name="champion_learning_curves.png"):
    """
    Plots Train Loss vs Val Loss and Train Acc vs Val Acc curves across epochs.
    """
    save_path = FIGURES_DIR / save_name
    epochs = history['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss subplot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', linewidth=2, linestyle='--')
    best_loss_idx = np.argmin(history['val_loss'])
    ax1.scatter([epochs[best_loss_idx]], [history['val_loss'][best_loss_idx]], color='red', s=60, zorder=5,
                label=f'Min Val Loss: {history["val_loss"][best_loss_idx]:.4f}')
    ax1.set_title("Training and Validation Loss", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=10)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], label='Val Accuracy', color='#ff7f0e', linewidth=2, linestyle='--')
    best_acc_idx = np.argmax(history['val_acc'])
    ax2.scatter([epochs[best_acc_idx]], [history['val_acc'][best_acc_idx] * 100], color='darkorange', s=60, zorder=5,
                label=f'Max Val Acc: {history["val_acc"][best_acc_idx]*100:.2f}%')
    ax2.set_title("Training and Validation Accuracy", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_confusion_matrix(cm, class_names=CLASSES, save_name="champion_confusion_matrix.png", normalize=True):
    """
    Plots publication-grade 7x7 confusion matrix heatmap with cell counts and percentages.
    """
    save_path = FIGURES_DIR / save_name
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Proportion' if normalize else 'Sample Count', fontsize=10)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=10, fontweight='bold')

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = f"{cm_norm[i, j]*100:.1f}%" if normalize else f"{cm[i, j]}"
            val_cnt = f"({cm[i, j]})"
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i - 0.12, val_pct, ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')
            ax.text(j, i + 0.15, val_cnt, ha="center", va="center", color=text_color, fontsize=7.5, alpha=0.85)

    ax.set_title("Normalized Confusion Matrix on FER-2013 Test Set", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("True Emotion Label", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_xlabel("Predicted Emotion Label", fontsize=11, fontweight='bold', labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_prediction_cases(images, y_true, y_pred, probs, class_names=CLASSES, is_correct=True, num_samples=16, save_name=None):
    """
    Plots true cases or wrong cases with confidence probability bars (matching demo_report pages 10-11).
    """
    if save_name is None:
        save_name = "champion_true_cases.png" if is_correct else "champion_wrong_cases.png"
    save_path = FIGURES_DIR / save_name

    mask = (y_true == y_pred) if is_correct else (y_true != y_pred)
    cand_indices = np.where(mask)[0]

    if len(cand_indices) > num_samples:
        np.random.seed(42)
        selected_idx = np.random.choice(cand_indices, size=num_samples, replace=False)
    else:
        selected_idx = cand_indices

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), dpi=300)

    for i, idx in enumerate(selected_idx[:rows*cols]):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = images[idx]
        ax.imshow(img, cmap='gray')

        t_lbl = class_names[y_true[idx]]
        p_lbl = class_names[y_pred[idx]]
        conf = probs[idx, y_pred[idx]] * 100

        if is_correct:
            title = f"True: {t_lbl}\nConf: {conf:.1f}%"
            title_color = "darkgreen"
        else:
            title = f"True: {t_lbl}\nPred: {p_lbl} ({conf:.1f}%)"
            title_color = "darkred"

        ax.set_title(title, fontsize=8.5, fontweight='bold', color=title_color, pad=4)
        ax.axis('off')

    main_title = "Correctly Classified Test Samples (True Cases)" if is_correct else "Misclassified Test Samples (Wrong Cases & Error Analysis)"
    plt.suptitle(main_title, fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_final_conv_filters(model, save_name="champion_conv_filters.png"):
    """
    Extracts and visualizes the final convolution filters (kernels) as required in Slide 5.
    """
    save_path = FIGURES_DIR / save_name
    last_conv = model.get_last_conv_layer()
    weights = last_conv.weight.data.cpu().numpy() # shape: (out_channels, in_channels, k_h, k_w)

    out_channels = weights.shape[0]
    num_to_plot = min(64, out_channels) # 8x8 grid
    rows, cols = 8, 8

    fig, axes = plt.subplots(rows, cols, figsize=(9, 9), dpi=300)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < num_to_plot:
            kernel = weights[i, 0, :, :] # kernel channel 0
            # Normalize kernel for visualization
            k_min, k_max = kernel.min(), kernel.max()
            norm_k = (kernel - k_min) / (k_max - k_min + 1e-8)
            ax.imshow(norm_k, cmap='viridis', interpolation='nearest')
        ax.axis('off')

    plt.suptitle(f"Learned Convolutional Filters in Final Layer ({last_conv.__class__.__name__})", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_feature_maps(model, sample_tensor, save_name="champion_feature_maps.png"):
    """
    Hooks intermediate activation maps across CNN blocks for a single sample face image.
    sample_tensor: (1, 1, 48, 48)
    """
    save_path = FIGURES_DIR / save_name
    model.eval()

    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    hooks = []
    # Register hooks on major conv blocks
    if hasattr(model, 'block1'):
        hooks.append(model.block1.register_forward_hook(hook_fn))
        hooks.append(model.block2.register_forward_hook(hook_fn))
        hooks.append(model.block3.register_forward_hook(hook_fn))
    elif hasattr(model, 'stage1'):
        hooks.append(model.stage1.register_forward_hook(hook_fn))
        hooks.append(model.stage2.register_forward_hook(hook_fn))
        hooks.append(model.stage3.register_forward_hook(hook_fn))
        hooks.append(model.stage4.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(sample_tensor)

    for h in hooks:
        h.remove()

    if not activations:
        print("No activations captured for feature maps.")
        return None

    # Plot 8 feature maps from each captured stage
    num_stages = min(3, len(activations))
    fig, axes = plt.subplots(num_stages, 8, figsize=(12, num_stages * 1.8), dpi=300)

    for s in range(num_stages):
        act = activations[s][0] # (C, H, W)
        for f in range(8):
            ax = axes[s, f] if num_stages > 1 else axes[f]
            if f < act.shape[0]:
                feat = act[f]
                ax.imshow(feat, cmap='magma')
            ax.axis('off')
            if f == 0:
                ax.set_title(f"Stage {s+1} ({act.shape[1]}x{act.shape[2]})", fontsize=9, fontweight='bold', loc='left')

    plt.suptitle("Feature Activation Maps Across Sequential CNN Hierarchy", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_experiment_bar_chart(models, accuracies, times, metric_label="Accuracy (%)", title="Experiment Comparison", save_name="exp_comparison.png"):
    """
    Plots dual-axis bar and line chart comparing model accuracy and training time.
    """
    save_path = FIGURES_DIR / save_name
    indices = np.arange(len(models))
    width = 0.45

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax2 = ax1.twinx()

    rects = ax1.bar(indices, accuracies, width, color='#1f77b4', alpha=0.85, label='Test Accuracy')
    lines = ax2.plot(indices, times, color='#d62728', marker='o', linewidth=2.2, label='Epoch Time (s)')

    ax1.set_xlabel("Architecture / Setup", fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_ylabel(metric_label, fontsize=10, fontweight='bold', color='#1f77b4', labelpad=8)
    ax2.set_ylabel("Avg Time per Epoch (s)", fontsize=10, fontweight='bold', color='#d62728', labelpad=8)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(models, fontsize=9, fontweight='bold')

    for r in rects:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path
