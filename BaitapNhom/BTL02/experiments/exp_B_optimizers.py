"""
BTL02 - Thực nghiệm B: Khảo sát ảnh hưởng của thuật toán tối ưu (Optimizers & Learning Rates)
So sánh: SGD (với Momentum=0.9), Adam, AdamW
Mô hình chuẩn: MediumCNN (cố định kiến trúc)
Đo lường: Tốc độ hội tụ loss, độ ổn định của gradient, Test Accuracy và F1.
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


def run_experiment_B(epochs=6, batch_size=128, subset_size=6000):
    print("=" * 60)
    print("BẮT ĐẦU THỰC NGHIỆM B: KHẢO SÁT CÁC THUẬT TOÁN TỐI ƯU")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, augment_mode='basic', subset_size=subset_size)

    optimizers_config = {
        'SGD_Momentum': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4),
        'Adam': lambda params: torch.optim.Adam(params, lr=0.001, weight_decay=1e-4),
        'AdamW': lambda params: torch.optim.AdamW(params, lr=0.001, weight_decay=1e-3)
    }

    results = []
    histories = {}

    for opt_name, opt_fn in optimizers_config.items():
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model = MediumCNN()
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        print(f"\n--- Huấn luyện MediumCNN với Optimizer: {opt_name} ---")
        trainer = FERTrainer(model, optimizer, criterion, scheduler, device=DEVICE, checkpoint_name=f"expB_{opt_name}.pt")
        start_t = time.time()
        hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=5, log_csv_name=f"exp_B_{opt_name}_log.csv", verbose=True)
        total_time = time.time() - start_t
        histories[opt_name] = hist

        preds, targets, probs, _ = trainer.predict(test_loader)
        _, summary = compute_classification_report(targets, preds, class_names=CLASSES)

        test_acc = summary['accuracy'] * 100
        macro_f1 = summary['macro_f1'] * 100
        best_val_acc = max(hist['val_acc']) * 100
        min_val_loss = min(hist['val_loss'])

        results.append({
            'Optimizer': opt_name,
            'Best_Val_Loss': round(min_val_loss, 4),
            'Best_Val_Acc_Pct': round(best_val_acc, 2),
            'Test_Acc_Pct': round(test_acc, 2),
            'Macro_F1_Pct': round(macro_f1, 2),
            'Total_Time_Sec': round(total_time, 2)
        })

    df_results = pd.DataFrame(results)
    out_csv = LOGS_DIR / "exp_B_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nĐã lưu bảng tổng kết Thực nghiệm B: {out_csv}")
    print(df_results.to_string(index=False))

    # Vẽ biểu đồ so sánh đường cong hội tụ loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    colors = {'SGD_Momentum': '#1f77b4', 'Adam': '#ff7f0e', 'AdamW': '#2ca02c'}

    for opt_name, h in histories.items():
        epochs_arr = h['epoch']
        c = colors.get(opt_name, 'black')
        ax1.plot(epochs_arr, h['val_loss'], label=f"{opt_name} (Val Loss)", color=c, linewidth=2)
        ax2.plot(epochs_arr, [a * 100 for a in h['val_acc']], label=f"{opt_name} (Val Acc)", color=c, linewidth=2)

    ax1.set_title("Validation Loss Trajectory by Optimizer", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.set_title("Validation Accuracy Convergence by Optimizer", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    chart_path = FIGURES_DIR / "exp_B_optimizer_convergence.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart_path}")

    return df_results


if __name__ == "__main__":
    run_experiment_B()
