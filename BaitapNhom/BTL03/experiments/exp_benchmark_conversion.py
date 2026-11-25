"""
BTL03: Bộ thực nghiệm kiểm chứng và đo lường hiệu năng các mô hình PyTorch (Dữ liệu thật)
Kiểm chứng hội tụ và xuất số liệu thực đo (Accuracy / R² / Pixel Acc / Val Loss / Thời gian) cho:
1. GaussianMLP (Hàm Gaussian 2D giải tích)
2. MNIST_MLP (MNIST thật từ data/mnist.npz)
3. MNIST_CNN (MNIST thật từ data/mnist.npz)
4. Gender_CNN & Emotion_CNN (FER-2013 thật từ BTL02 cache)
5. MNIST_CNN_LSTM (Chuỗi chữ số MNIST thật, L1Loss)
6. SegNet (Ảnh độ sâu bàn tay thật từ depth_hand.zip)
7. CarDetectorCNN (Ảnh xe thật từ car_data.zip & test.jpg)
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    EMOTION_CLASSES
)
from src.models import (
    GaussianMLP,
    MNIST_MLP,
    MNIST_CNN,
    Gender_CNN,
    Emotion_CNN,
    MNIST_CNN_LSTM,
    SegNet,
    CarDetectorCNN
)
from src.data import (
    get_gaussian_dataloaders,
    get_mnist_dataloaders,
    get_gender_dataloaders,
    get_emotion_dataloaders,
    get_mnist_lstm_sequence_dataloaders,
    get_segnet_dataloaders,
    get_car_detection_dataloaders
)
from src.trainer import fit_model
from src.metrics import (
    compute_regression_metrics,
    compute_segmentation_metrics,
    compute_confusion_matrix
)
from src.visualize import (
    plot_learning_curves,
    plot_gaussian_3d_comparison,
    plot_confusion_matrix_heatmap,
    plot_segnet_results,
    plot_car_detection_sample,
    plot_pytorch_benchmark_summary
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_all_benchmarks():
    print("=" * 70)
    print("THỰC NGHIỆM KIỂM CHỨNG HIỆU NĂNG CÁC MÔ HÌNH PYTORCH (DỮ LIỆU THẬT)")
    print(f"Thiết bị tính toán: {DEVICE}")
    print("=" * 70)

    results = []
    crit_ce = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # 1. GAUSSIAN MLP APPROXIMATION
    # -------------------------------------------------------------
    print("\n>>> [1/7] PyTorch GaussianMLP (B=64)...")
    tr_gauss, va_gauss = get_gaussian_dataloaders(batch_size=32)
    model_gauss = GaussianMLP(hidden_dim=64).to(DEVICE)
    opt_gauss = optim.Adam(model_gauss.parameters(), lr=0.01)
    hist_gauss = fit_model(
        model_gauss, tr_gauss, va_gauss, opt_gauss, nn.MSELoss(),
        epochs=250, device=DEVICE, is_regression=True, verbose=False
    )

    model_gauss.eval()
    with torch.no_grad():
        x_val_g, y_val_g = va_gauss.dataset.tensors
        pred_g = model_gauss(x_val_g.to(DEVICE)).cpu().numpy()
        reg_g = compute_regression_metrics(y_val_g.numpy(), pred_g)

    plot_learning_curves(hist_gauss, "GaussianMLP (B=128)", "learning_curves_gaussian_mlp.png", is_regression=True)
    plot_gaussian_3d_comparison(model_gauss, "gaussian_3d_comparison.png", device=DEVICE)
    torch.save(model_gauss.state_dict(), CHECKPOINTS_DIR / "gaussian_mlp.pt")

    results.append({
        'model': 'GaussianMLP',
        'params': count_parameters(model_gauss),
        'metric_name': 'R2 Score (%)',
        'score_pct': round(reg_g['r2'] * 100, 2),
        'val_loss': round(hist_gauss['val_loss'][-1], 6),
        'train_time_s': round(hist_gauss['total_time'], 2),
        'dataset_source': 'Gaussian 2D Grid (32x32 / 10x10)'
    })
    print(f"    R2 = {reg_g['r2']*100:.2f}%, Val MSE = {hist_gauss['val_loss'][-1]:.6e}")

    # -------------------------------------------------------------
    # 2. MNIST MLP (REAL MNIST)
    # -------------------------------------------------------------
    print("\n>>> [2/7] PyTorch MNIST_MLP (Real MNIST)...")
    tr_mnist_full, va_mnist_full = get_mnist_dataloaders(batch_size=128, num_train=60000, num_test=10000)
    model_mnist_mlp = MNIST_MLP(hidden_dim=512, dropout_rate=0.2).to(DEVICE)
    opt_mnist_mlp = optim.SGD(model_mnist_full_params := model_mnist_mlp.parameters(), lr=0.05, momentum=0.9)
    hist_mnist_mlp = fit_model(
        model_mnist_mlp, tr_mnist_full, va_mnist_full, opt_mnist_mlp, crit_ce,
        epochs=15, device=DEVICE, is_regression=False, verbose=False
    )

    plot_learning_curves(hist_mnist_mlp, "MNIST_MLP", "learning_curves_mnist_mlp.png")
    torch.save(model_mnist_mlp.state_dict(), CHECKPOINTS_DIR / "mnist_mlp.pt")

    acc_mlp = hist_mnist_mlp['val_acc'][-1] * 100
    results.append({
        'model': 'MNIST_MLP',
        'params': count_parameters(model_mnist_mlp),
        'metric_name': 'Test Acc (%)',
        'score_pct': round(acc_mlp, 2),
        'val_loss': round(hist_mnist_mlp['val_loss'][-1], 4),
        'train_time_s': round(hist_mnist_mlp['total_time'], 2),
        'dataset_source': 'MNIST Real (60k / 10k)'
    })
    print(f"    Test Acc = {acc_mlp:.2f}%, Test Loss = {hist_mnist_mlp['val_loss'][-1]:.4f}")

    # -------------------------------------------------------------
    # 3. MNIST CNN (REAL MNIST, 3 CONV BLOCKS)
    # -------------------------------------------------------------
    print("\n>>> [3/7] PyTorch MNIST_CNN (Real MNIST)...")
    tr_mnist_sub, va_mnist_sub = get_mnist_dataloaders(batch_size=128, num_train=5000, num_test=1000)
    model_mnist_cnn = MNIST_CNN().to(DEVICE)
    opt_mnist_cnn = optim.Adam(model_mnist_cnn.parameters(), lr=0.001)
    hist_mnist_cnn = fit_model(
        model_mnist_cnn, tr_mnist_sub, va_mnist_sub, opt_mnist_cnn, crit_ce,
        epochs=6, device=DEVICE, is_regression=False, verbose=False
    )

    plot_learning_curves(hist_mnist_cnn, "MNIST_CNN", "learning_curves_mnist_cnn.png")
    torch.save(model_mnist_cnn.state_dict(), CHECKPOINTS_DIR / "mnist_cnn.pt")

    acc_cnn = hist_mnist_cnn['val_acc'][-1] * 100
    results.append({
        'model': 'MNIST_CNN',
        'params': count_parameters(model_mnist_cnn),
        'metric_name': 'Test Acc (%)',
        'score_pct': round(acc_cnn, 2),
        'val_loss': round(hist_mnist_cnn['val_loss'][-1], 4),
        'train_time_s': round(hist_mnist_cnn['total_time'], 2),
        'dataset_source': 'MNIST Real (15k / 3k)'
    })
    print(f"    Test Acc = {acc_cnn:.2f}%, Test Loss = {hist_mnist_cnn['val_loss'][-1]:.4f}")

    # -------------------------------------------------------------
    # 4. GENDER CNN & EMOTION CNN (REAL FER-2013)
    # -------------------------------------------------------------
    print("\n>>> [4/7] PyTorch Gender_CNN & Emotion_CNN...")
    tr_gen, va_gen = get_gender_dataloaders(num_samples=800, batch_size=64)
    model_gen = Gender_CNN().to(DEVICE)
    opt_gen = optim.Adam(model_gen.parameters(), lr=0.0005)
    hist_gen = fit_model(model_gen, tr_gen, va_gen, opt_gen, crit_ce, epochs=6, device=DEVICE, verbose=False)
    plot_learning_curves(hist_gen, "Gender_CNN", "learning_curves_gender_cnn.png")
    torch.save(model_gen.state_dict(), CHECKPOINTS_DIR / "gender_cnn.pt")

    tr_emo, va_emo, te_emo = get_emotion_dataloaders(batch_size=64, subset_size=1200)
    model_emo = Emotion_CNN().to(DEVICE)
    opt_emo = optim.Adam(model_emo.parameters(), lr=0.0005)
    hist_emo = fit_model(model_emo, tr_emo, va_emo, opt_emo, crit_ce, epochs=5, device=DEVICE, verbose=False)
    plot_learning_curves(hist_emo, "Emotion_CNN", "learning_curves_emotion_cnn.png")
    torch.save(model_emo.state_dict(), CHECKPOINTS_DIR / "emotion_cnn.pt")

    model_emo.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in te_emo:
            all_preds.extend(torch.argmax(model_emo(xb.to(DEVICE)), dim=1).cpu().numpy())
            all_trues.extend(yb.numpy())
    cm_emo = compute_confusion_matrix(np.array(all_trues), np.array(all_preds), num_classes=7)
    plot_confusion_matrix_heatmap(cm_emo, EMOTION_CLASSES, "Confusion Matrix — Emotion_CNN (FER-2013)", "confusion_matrix_emotion.png")

    acc_emo = float(np.mean(np.array(all_preds) == np.array(all_trues))) * 100
    results.append({
        'model': 'Emotion_CNN',
        'params': count_parameters(model_emo),
        'metric_name': 'Test Acc (%)',
        'score_pct': round(acc_emo, 2),
        'val_loss': round(hist_emo['val_loss'][-1], 4),
        'train_time_s': round(hist_emo['total_time'], 2),
        'dataset_source': 'FER-2013 Real (1200 / 300)'
    })
    print(f"    Emotion Test Acc = {acc_emo:.2f}%, Val Loss = {hist_emo['val_loss'][-1]:.4f}")

    # -------------------------------------------------------------
    # 5. CNN + LSTM (REAL MNIST DIGIT SUMMATION, L1Loss)
    # -------------------------------------------------------------
    print("\n>>> [5/7] PyTorch MNIST_CNN_LSTM (Real MNIST Sequences)...")
    tr_lstm, va_lstm = get_mnist_lstm_sequence_dataloaders(max_to_add=4, num_train_seqs=1200, num_test_seqs=300, batch_size=64)
    model_lstm = MNIST_CNN_LSTM().to(DEVICE)
    # Kế thừa bộ lọc Conv đã học từ MNIST_CNN để tăng tốc hội tụ trên CPU
    cnn_dict = {k.replace("features.", ""): v for k, v in model_mnist_cnn.state_dict().items() if k.startswith("features.")}
    model_lstm.cnn_encoder.load_state_dict(cnn_dict, strict=False)
    for p in model_lstm.cnn_encoder.parameters():
        p.requires_grad = False

    def _extract_lstm_feats(loader):
        xs, ys = [], []
        model_lstm.eval()
        with torch.no_grad():
            for xb, yb in loader:
                B, T, C, H, W = xb.shape
                f = model_lstm.cnn_encoder(xb.to(DEVICE).reshape(B * T, C, H, W)).view(B, T, -1).cpu()
                xs.append(f)
                ys.append(yb)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.cat(xs), torch.cat(ys)),
            batch_size=64, shuffle=(loader is tr_lstm)
        )

    tr_lstm_f = _extract_lstm_feats(tr_lstm)
    va_lstm_f = _extract_lstm_feats(va_lstm)
    opt_lstm = optim.Adam([p for p in model_lstm.parameters() if p.requires_grad], lr=0.002)
    crit_l1 = nn.L1Loss()
    hist_lstm = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    t_lstm0 = time.time()
    for _ in range(45):
        model_lstm.train()
        tr_l, tr_n = 0.0, 0
        for fb, yb in tr_lstm_f:
            fb, yb = fb.to(DEVICE), yb.to(DEVICE)
            opt_lstm.zero_grad()
            o1, _ = model_lstm.lstm1(fb)
            _, (hn2, _) = model_lstm.lstm2(o1)
            pred = model_lstm.regressor(model_lstm.dropout(hn2[-1])).squeeze(-1)
            loss = crit_l1(pred, yb)
            loss.backward()
            opt_lstm.step()
            tr_l += loss.item() * len(fb)
            tr_n += len(fb)
        model_lstm.eval()
        va_l, va_n = 0.0, 0
        with torch.no_grad():
            for fb, yb in va_lstm_f:
                fb, yb = fb.to(DEVICE), yb.to(DEVICE)
                o1, _ = model_lstm.lstm1(fb)
                _, (hn2, _) = model_lstm.lstm2(o1)
                pred = model_lstm.regressor(hn2[-1]).squeeze(-1)
                va_l += crit_l1(pred, yb).item() * len(fb)
                va_n += len(fb)
        hist_lstm['train_loss'].append(tr_l / tr_n)
        hist_lstm['val_loss'].append(va_l / va_n)
    hist_lstm['total_time'] = time.time() - t_lstm0

    plot_learning_curves(hist_lstm, "MNIST_CNN_LSTM (MAE)", "learning_curves_cnn_lstm.png", is_regression=True)
    torch.save(model_lstm.state_dict(), CHECKPOINTS_DIR / "mnist_cnn_lstm.pt")

    model_lstm.eval()
    with torch.no_grad():
        y_t_lstm, y_p_lstm = [], []
        for fb, yb in va_lstm_f:
            o1, _ = model_lstm.lstm1(fb.to(DEVICE))
            _, (hn2, _) = model_lstm.lstm2(o1)
            y_p_lstm.extend(model_lstm.regressor(hn2[-1]).squeeze(-1).cpu().numpy())
            y_t_lstm.extend(yb.numpy())
        reg_lstm = compute_regression_metrics(np.array(y_t_lstm), np.array(y_p_lstm))

    results.append({
        'model': 'CNN+LSTM',
        'params': count_parameters(model_lstm) + sum(p.numel() for p in model_lstm.cnn_encoder.parameters()),
        'metric_name': 'R2 Score (%)',
        'score_pct': round(max(0.0, reg_lstm['r2']) * 100, 2),
        'val_loss': round(hist_lstm['val_loss'][-1], 4),
        'train_time_s': round(hist_lstm['total_time'], 2),
        'dataset_source': 'MNIST Sequences Real (1200 / 300)'
    })
    print(f"    CNN+LSTM R2 = {reg_lstm['r2']*100:.2f}%, Test MAE = {hist_lstm['val_loss'][-1]:.4f}")

    # -------------------------------------------------------------
    # 6. SEGNET (REAL depth_hand.zip)
    # -------------------------------------------------------------
    print("\n>>> [6/7] PyTorch SegNet (Real depth_hand.zip)...")
    tr_seg, va_seg = get_segnet_dataloaders(batch_size=16, num_train=240, num_test=80)
    model_seg = SegNet(compact=True).to(DEVICE)
    opt_seg = optim.Adam(model_seg.parameters(), lr=0.001)
    hist_seg = fit_model(model_seg, tr_seg, va_seg, opt_seg, crit_ce, epochs=10, device=DEVICE, verbose=False)
    plot_segnet_results(model_seg, va_seg, "segnet_prediction_sample.png", device=DEVICE)
    torch.save(model_seg.state_dict(), CHECKPOINTS_DIR / "segnet.pt")

    model_seg.eval()
    with torch.no_grad():
        seg_t, seg_p = [], []
        for xb, yb in va_seg:
            seg_p.append(torch.argmax(model_seg(xb.to(DEVICE)), dim=1).cpu().numpy())
            seg_t.append(yb.numpy())
        seg_metrics = compute_segmentation_metrics(np.concatenate(seg_t), np.concatenate(seg_p), num_classes=7)

    results.append({
        'model': 'SegNet',
        'params': count_parameters(model_seg),
        'metric_name': 'Pixel Acc (%)',
        'score_pct': round(seg_metrics['pixel_accuracy'] * 100, 2),
        'val_loss': round(hist_seg['val_loss'][-1], 4),
        'train_time_s': round(hist_seg['total_time'], 2),
        'dataset_source': 'depth_hand.zip Real (240 / 80)'
    })
    print(f"    SegNet Pixel Acc = {seg_metrics['pixel_accuracy']*100:.2f}%, mIoU = {seg_metrics['miou']*100:.2f}%")

    # -------------------------------------------------------------
    # 7. CAR OBJECT DETECTION (REAL car_data.zip + test.jpg)
    # -------------------------------------------------------------
    print("\n>>> [7/7] PyTorch CarDetectorCNN (Real car_data.zip)...")
    tr_car, va_car = get_car_detection_dataloaders(batch_size=64, max_per_class=500)
    model_car = CarDetectorCNN(normalize_input=True).to(DEVICE)
    opt_car = optim.Adam(model_car.parameters(), lr=0.001)
    hist_car = fit_model(model_car, tr_car, va_car, opt_car, nn.MSELoss(), epochs=10, device=DEVICE, is_regression=True, verbose=False)

    model_car.eval()
    car_correct, car_total = 0, 0
    with torch.no_grad():
        for xb, yb in va_car:
            out = model_car(xb.to(DEVICE))
            car_correct += ((out > 0).float() == (yb.to(DEVICE) > 0).float()).sum().item()
            car_total += len(xb)
    acc_car = (car_correct / max(1, car_total)) * 100

    plot_car_detection_sample(model_car, "car_detection_sample.png", threshold=0.85)
    torch.save(model_car.state_dict(), CHECKPOINTS_DIR / "car_detector.pt")

    results.append({
        'model': 'CarDetectorCNN',
        'params': count_parameters(model_car),
        'metric_name': 'Test Acc (%)',
        'score_pct': round(acc_car, 2),
        'val_loss': round(hist_car['val_loss'][-1], 4),
        'train_time_s': round(hist_car['total_time'], 2),
        'dataset_source': 'car_data.zip Real (1000 imgs)'
    })
    print(f"    CarDetector Test Acc = {acc_car:.2f}%, Test MSE = {hist_car['val_loss'][-1]:.4f}")

    # Biểu đồ tổng kết thực đo
    plot_pytorch_benchmark_summary(results, "pytorch_benchmark_summary.png")

    df_results = pd.DataFrame(results)
    csv_path = LOGS_DIR / "benchmark_keras_vs_pytorch.csv"
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC ĐO CÁC MÔ HÌNH PYTORCH:")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    return df_results


if __name__ == "__main__":
    run_all_benchmarks()
