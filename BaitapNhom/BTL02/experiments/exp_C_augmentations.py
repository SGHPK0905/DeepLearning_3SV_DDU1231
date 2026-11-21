"""
BTL02 - Thực nghiệm C: Khảo sát ảnh hưởng của các kỹ thuật Data Augmentation
So sánh 3 chế độ:
1. None: Chỉ chuẩn hóa pixel về [-1, 1]
2. Basic: Random Horizontal Flip (p=0.5)
3. Advanced: Flip + Rotation (+-10 deg) + Shift (+-4 px) + Cutout/Erasing
Đo lường: Hiện tượng Overfitting (Generalization Gap = Train Acc - Val Acc), Test Acc và F1.
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

from src.config import DEVICE, SEED, LOGS_DIR, FIGURES_DIR, CLASSES
from src.data import get_dataloaders
from src.models import MediumCNN
from src.trainer import FERTrainer
from src.metrics import compute_classification_report


def run_experiment_C(epochs=6, batch_size=128, subset_size=6000):
    print("=" * 60)
    print("BẮT ĐẦU THỰC NGHIỆM C: KHẢO SÁT KỸ THUẬT DATA AUGMENTATION")
    print("=" * 60)

    aug_modes = ['none', 'basic', 'advanced']
    results = []
    histories = {}

    for mode in aug_modes:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        print(f"\n--- Chuẩn bị DataLoader với chế độ Augmentation: '{mode}' ---")
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, augment_mode=mode, subset_size=subset_size)

        model = MediumCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        trainer = FERTrainer(model, optimizer, criterion, scheduler, device=DEVICE, checkpoint_name=f"expC_{mode}.pt")
        start_t = time.time()
        hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=5, log_csv_name=f"exp_C_{mode}_log.csv", verbose=True)
        total_time = time.time() - start_t
        histories[mode] = hist

        preds, targets, probs, _ = trainer.predict(test_loader)
        _, summary = compute_classification_report(targets, preds, class_names=CLASSES)

        test_acc = summary['accuracy'] * 100
        macro_f1 = summary['macro_f1'] * 100
        final_train_acc = hist['train_acc'][-1] * 100
        final_val_acc = hist['val_acc'][-1] * 100
        gap = final_train_acc - final_val_acc

        results.append({
            'Augmentation_Mode': mode,
            'Final_Train_Acc_Pct': round(final_train_acc, 2),
            'Final_Val_Acc_Pct': round(final_val_acc, 2),
            'Overfitting_Gap_Pct': round(gap, 2),
            'Test_Acc_Pct': round(test_acc, 2),
            'Macro_F1_Pct': round(macro_f1, 2),
            'Total_Time_Sec': round(total_time, 2)
        })

    df_results = pd.DataFrame(results)
    out_csv = LOGS_DIR / "exp_C_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nĐã lưu bảng tổng kết Thực nghiệm C: {out_csv}")
    print(df_results.to_string(index=False))

    # Biểu đồ phân tích khoảng cách Overfitting và Test Acc
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    modes_label = ['No Augmentation\n(Raw Normalize)', 'Basic Augmentation\n(Random Flip)', 'Advanced Augmentation\n(Flip+Rot+Shift+Cutout)']
    x = np.arange(len(modes_label))
    width = 0.28

    rects1 = ax.bar(x - width, df_results['Final_Train_Acc_Pct'], width, label='Train Accuracy', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x, df_results['Final_Val_Acc_Pct'], width, label='Validation Accuracy', color='#ff7f0e', alpha=0.9)
    rects3 = ax.bar(x + width, df_results['Test_Acc_Pct'], width, label='Test Accuracy', color='#2ca02c', alpha=0.9)

    ax.set_ylabel("Accuracy (%)", fontsize=10, fontweight='bold')
    ax.set_title("Experiment C: Impact of Data Augmentation on Generalization Gap", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(modes_label, fontsize=9, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for rects in [rects1, rects2, rects3]:
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    chart_path = FIGURES_DIR / "exp_C_augmentation_impact.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart_path}")

    return df_results


if __name__ == "__main__":
    run_experiment_C()
