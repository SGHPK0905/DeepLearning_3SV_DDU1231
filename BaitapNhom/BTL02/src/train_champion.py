"""
BTL02: Champion Model Training & Complete Evaluation Pipeline
Trains the selected Champion CNN (MediumCNN with AdamW and Advanced Augmentation),
saves checkpoints/best_model.pt, and produces all required visual artifacts:
- champion_learning_curves.png
- champion_confusion_matrix.png
- champion_true_cases.png
- champion_wrong_cases.png
- champion_conv_filters.png (Required by Slide 5)
- champion_feature_maps.png
- champion_classification_report.csv
"""

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from .config import (
    DEVICE,
    SEED,
    CLASSES,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    FIGURES_DIR
)
from .data import get_dataloaders
from .models import MediumCNN, VGG16_FER
from .trainer import FERTrainer
from .metrics import compute_classification_report
from .visualize import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_prediction_cases,
    plot_final_conv_filters,
    plot_feature_maps
)


def train_champion_pipeline(epochs=12, batch_size=128, subset_size=12000):
    print("=" * 65)
    print("HUẤN LUYỆN MÔ HÌNH QUÁN QUÂN (CHAMPION MODEL PIPELINE)")
    print("=" * 65)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        augment_mode='advanced',
        subset_size=subset_size
    )

    model = MediumCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    trainer = FERTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        checkpoint_name="best_model.pt"
    )

    print("\n--- Bắt đầu vòng lặp huấn luyện chính ---")
    start_t = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=5,
        log_csv_name="champion_training_log.csv",
        verbose=True
    )
    total_time = time.time() - start_t
    print(f"Hoàn thành huấn luyện trong {total_time:.1f}s!")

    # 1. Vẽ đường cong học tập
    plot_learning_curves(history, save_name="champion_learning_curves.png")

    # 2. Đánh giá toàn diện trên Test Set (PrivateTest: 3,589 ảnh)
    print("\n--- Đánh giá mô hình tốt nhất trên tập kiểm thử (Test Set) ---")
    checkpoint = torch.load(CHECKPOINTS_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    preds, targets, probs, raw_imgs = trainer.predict(test_loader)
    df_report, summary = compute_classification_report(targets, preds, class_names=CLASSES)

    report_path = LOGS_DIR / "champion_classification_report.csv"
    df_report.to_csv(report_path, index=False)
    print(f"Đã lưu báo cáo phân loại: {report_path}")
    print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT TỪNG LỚP CẢM XÚC:")
    print(df_report.to_string(index=False))
    print(f"\nAccuracy tổng thể: {summary['accuracy']*100:.2f}% | Macro F1: {summary['macro_f1']*100:.2f}%")

    # 3. Vẽ ma trận nhầm lẫn (Confusion Matrix)
    plot_confusion_matrix(summary['confusion_matrix'], class_names=CLASSES, save_name="champion_confusion_matrix.png")

    # 4. Vẽ các trường hợp đúng (True Cases)
    plot_prediction_cases(raw_imgs, targets, preds, probs, class_names=CLASSES, is_correct=True, num_samples=16, save_name="champion_true_cases.png")

    # 5. Vẽ các trường hợp sai (Wrong Cases)
    plot_prediction_cases(raw_imgs, targets, preds, probs, class_names=CLASSES, is_correct=False, num_samples=16, save_name="champion_wrong_cases.png")

    # 6. Trực quan hóa bộ lọc tích chập tầng cuối (Final Conv Filters) - Yêu cầu Slide 5
    plot_final_conv_filters(model, save_name="champion_conv_filters.png")

    # 7. Trực quan hóa bản đồ kích hoạt (Feature Maps)
    sample_tensor, _ = next(iter(test_loader))
    plot_feature_maps(model, sample_tensor[:1].to(DEVICE), save_name="champion_feature_maps.png")

    print("\nTẤT CẢ MINH CHỨNG VÀ HÌNH VẼ ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
    return df_report, summary, history


if __name__ == "__main__":
    train_champion_pipeline()
