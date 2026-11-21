"""
BTL02 - Thực nghiệm D: Khảo sát Regularization (Dropout & Batch Normalization Ablation)
Khảo sát:
1. Dropout thấp: 0.1 conv / 0.2 fc
2. Dropout chuẩn: 0.25 conv / 0.5 fc
3. Dropout cao: 0.4 conv / 0.6 fc
4. Không dùng Batch Normalization (Ablation study)
Mô hình chuẩn: MediumCNN
Đo lường: Tác động điều hòa sai số, ổn định huấn luyện và Test Accuracy.
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


class MediumCNN_NoBN(nn.Module):
    """
    Ablation architecture: MediumCNN without Batch Normalization layers.
    """
    def __init__(self, num_classes=7, dropout_rate=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.block3(self.block2(self.block1(x))))

    def get_last_conv_layer(self):
        return self.block3[2]


def run_experiment_D(epochs=6, batch_size=128, subset_size=6000):
    print("=" * 60)
    print("BẮT ĐẦU THỰC NGHIỆM D: KHẢO SÁT REGULARIZATION & BATCH NORMALIZATION")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, augment_mode='basic', subset_size=subset_size)

    variants = {
        'Dropout_Low (0.1)': lambda: MediumCNN(dropout_rate=0.1),
        'Dropout_Standard (0.25)': lambda: MediumCNN(dropout_rate=0.25),
        'Dropout_High (0.4)': lambda: MediumCNN(dropout_rate=0.4),
        'No_BatchNorm (Ablation)': lambda: MediumCNN_NoBN(dropout_rate=0.25)
    }

    results = []
    for var_name, model_fn in variants.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        print(f"\n--- Huấn luyện biến thể: {var_name} ---")
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        safe_name = var_name.split()[0]
        trainer = FERTrainer(model, optimizer, criterion, scheduler, device=DEVICE, checkpoint_name=f"expD_{safe_name}.pt")
        start_t = time.time()
        hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=5, log_csv_name=f"exp_D_{safe_name}_log.csv", verbose=True)
        total_time = time.time() - start_t

        preds, targets, probs, _ = trainer.predict(test_loader)
        _, summary = compute_classification_report(targets, preds, class_names=CLASSES)

        test_acc = summary['accuracy'] * 100
        macro_f1 = summary['macro_f1'] * 100
        best_val_acc = max(hist['val_acc']) * 100
        min_val_loss = min(hist['val_loss'])

        results.append({
            'Regularization_Setup': var_name,
            'Best_Val_Loss': round(min_val_loss, 4),
            'Best_Val_Acc_Pct': round(best_val_acc, 2),
            'Test_Acc_Pct': round(test_acc, 2),
            'Macro_F1_Pct': round(macro_f1, 2),
            'Total_Time_Sec': round(total_time, 2)
        })

    df_results = pd.DataFrame(results)
    out_csv = LOGS_DIR / "exp_D_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nĐã lưu bảng tổng kết Thực nghiệm D: {out_csv}")
    print(df_results.to_string(index=False))

    # Vẽ biểu đồ so sánh Test Acc
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar(df_results['Regularization_Setup'], df_results['Test_Acc_Pct'], color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'], width=0.5, alpha=0.9)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10, fontweight='bold')
    ax.set_title("Experiment D: Regularization & Batch Normalization Ablation", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}%", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=15, ha='right', fontsize=9, fontweight='bold')
    plt.tight_layout()
    chart_path = FIGURES_DIR / "exp_D_regularization_ablation.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart_path}")

    return df_results


if __name__ == "__main__":
    run_experiment_D()
