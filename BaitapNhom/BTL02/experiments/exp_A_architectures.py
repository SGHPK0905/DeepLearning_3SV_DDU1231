"""
BTL02 - Thực nghiệm A: Khảo sát và so sánh các kiến trúc mạng CNN (Conv, Pooling, FC)
So sánh 4 kiến trúc: SmallCNN, MediumCNN, VGG16_FER, ResNet_FER
Đo lường: Train/Val Loss, Test Accuracy, Macro F1, Số tham số, Thời gian huấn luyện/epoch.
"""

import time
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.config import DEVICE, SEED, LOGS_DIR, FIGURES_DIR, CLASSES
from src.data import get_dataloaders
from src.models import SmallCNN, MediumCNN, VGG16_FER, ResNet_FER
from src.trainer import FERTrainer
from src.metrics import compute_classification_report
from src.visualize import plot_experiment_bar_chart


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiment_A(epochs=6, batch_size=128, subset_size=6000):
    print("=" * 60)
    print("BẮT ĐẦU THỰC NGHIỆM A: KHẢO SÁT CÁC KIẾN TRÚC MẠNG CNN")
    print("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, augment_mode='basic', subset_size=subset_size)

    architectures = {
        'SmallCNN': SmallCNN(),
        'MediumCNN': MediumCNN(),
        'VGG16_FER': VGG16_FER(),
        'ResNet_FER': ResNet_FER()
    }

    results = []
    history_dict = {}

    for name, model in architectures.items():
        params = count_parameters(model)
        print(f"\n--- Đang huấn luyện mô hình: {name} ({params:,} tham số) ---")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        trainer = FERTrainer(model, optimizer, criterion, scheduler, device=DEVICE, checkpoint_name=f"expA_{name}.pt")
        start_t = time.time()
        hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=5, log_csv_name=f"exp_A_{name}_log.csv", verbose=True)
        total_time = time.time() - start_t
        avg_epoch_time = total_time / len(hist['epoch'])

        history_dict[name] = hist

        # Đánh giá trên test set
        preds, targets, probs, _ = trainer.predict(test_loader)
        df_rep, summary = compute_classification_report(targets, preds, class_names=CLASSES)

        test_acc = summary['accuracy'] * 100
        macro_f1 = summary['macro_f1'] * 100
        best_val_acc = max(hist['val_acc']) * 100
        min_val_loss = min(hist['val_loss'])

        print(f"Kết quả {name}: Test Acc={test_acc:.2f}%, Macro F1={macro_f1:.2f}%, Avg Time/epoch={avg_epoch_time:.1f}s")

        results.append({
            'Model': name,
            'Parameters': params,
            'Best_Val_Loss': round(min_val_loss, 4),
            'Best_Val_Acc_Pct': round(best_val_acc, 2),
            'Test_Acc_Pct': round(test_acc, 2),
            'Macro_F1_Pct': round(macro_f1, 2),
            'Avg_Epoch_Time_Sec': round(avg_epoch_time, 2),
            'Total_Time_Sec': round(total_time, 2)
        })

    df_results = pd.DataFrame(results)
    out_csv = LOGS_DIR / "exp_A_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nĐã lưu bảng tổng kết Thực nghiệm A: {out_csv}")
    print(df_results.to_string(index=False))

    # Vẽ biểu đồ so sánh
    plot_experiment_bar_chart(
        models=df_results['Model'].tolist(),
        accuracies=df_results['Test_Acc_Pct'].tolist(),
        times=df_results['Avg_Epoch_Time_Sec'].tolist(),
        metric_label="Test Accuracy (%)",
        title="Experiment A: CNN Architecture Performance vs Epoch Training Time",
        save_name="exp_A_architecture_comparison.png"
    )

    return df_results


if __name__ == "__main__":
    run_experiment_A()
