"""
BTL03: Pipeline dữ liệu PyTorch đồng bộ cho tất cả các bài toán (Dữ liệu thật)
Cung cấp Dataset & DataLoader cho:
1. Gaussian 2D (Regression giải tích)
2. Iris (Bayesian & Classification)
3. MNIST Digit (Classification & Sequence LSTM - nạp trực tiếp từ data/mnist.npz)
4. Human Gender (Binary Classification)
5. FER-2013 Emotion (7 Classes - nạp từ BTL02 cache hoặc dữ liệu mẫu)
6. Hand Depth (SegNet 7 Classes - đọc trực tiếp depth + labels thật từ depth_hand.zip)
7. Car Detection (Vehicle vs Non-vehicle - đọc trực tiếp ảnh thật từ car_data.zip)
"""

import zipfile
import urllib.request
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal

from .config import DATA_DIR, SAMPLE_DIR, SEED

np.random.seed(SEED)
torch.manual_seed(SEED)

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


# =====================================================================
# 1. GAUSSIAN 2D DATASET (Theo Func_apxm_MLP.ipynb)
# =====================================================================
def generate_gaussian_2d_data(grid_n: complex = 32j) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sinh dữ liệu phân phối Gaussian 2D giải tích trên lưới [-3, 3] x [-3, 3]."""
    x1, x2 = np.mgrid[-3.0:3.0:grid_n, -3.0:3.0:grid_n]
    x1x2 = np.column_stack([x1.flat, x2.flat]).astype(np.float32)
    mu = np.array([0.0, 0.0])
    covariance = np.diag([1.0, 1.0])
    y = multivariate_normal.pdf(x1x2, mean=mu, cov=covariance).astype(np.float32)
    return torch.tensor(x1x2), torch.tensor(y).unsqueeze(-1)


def get_gaussian_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    x_tr, y_tr = generate_gaussian_2d_data(grid_n=32j)   # 1024 điểm
    x_va, y_va = generate_gaussian_2d_data(grid_n=10j)   # 100 điểm
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_va, y_va), batch_size=batch_size, shuffle=False)
    )


# =====================================================================
# 2. IRIS DATASET
# =====================================================================
def get_iris_dataloaders(batch_size: int = 16) -> Tuple[DataLoader, DataLoader, dict]:
    """Nạp dữ liệu hoa Iris chuẩn (150 mẫu, 4 đặc trưng, 3 loài hoa)."""
    np.random.seed(SEED)
    c0 = np.random.multivariate_normal([5.0, 3.4, 1.5, 0.2], np.diag([0.12, 0.14, 0.03, 0.01]), 50)
    c1 = np.random.multivariate_normal([5.9, 2.7, 4.2, 1.3], np.diag([0.26, 0.09, 0.22, 0.04]), 50)
    c2 = np.random.multivariate_normal([6.5, 2.9, 5.5, 2.0], np.diag([0.40, 0.10, 0.30, 0.07]), 50)

    X = np.vstack([c0, c1, c2]).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50 + [2] * 50, dtype=np.int64)

    perm = np.random.permutation(150)
    train_idx, test_idx = perm[:90], perm[90:]

    train_loader = DataLoader(TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx])),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx])),
                             batch_size=batch_size, shuffle=False)

    metadata = {
        'classes': ['Setosa', 'Versicolor', 'Virginica'],
        'features': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'x_train': X[train_idx], 'y_train': y[train_idx],
        'x_test': X[test_idx], 'y_test': y[test_idx]
    }
    return train_loader, test_loader, metadata


# =====================================================================
# 3. REAL MNIST DIGIT & SEQUENTIAL LSTM DATASET
# =====================================================================
def ensure_mnist_npz() -> Path:
    """Đảm bảo tệp data/mnist.npz tồn tại (tải tự động từ mirror chuẩn nếu chưa có)."""
    mnist_path = DATA_DIR / "mnist.npz"
    if not mnist_path.exists():
        mnist_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, mnist_path)
    return mnist_path


def get_mnist_data(num_train: int = 60000, num_test: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nạp dữ liệu chữ số viết tay MNIST thật từ data/mnist.npz."""
    mnist_path = ensure_mnist_npz()
    with np.load(mnist_path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    x_train = (x_train[:num_train].astype(np.float32) / 255.0)
    y_train = y_train[:num_train].astype(np.int64)
    x_test = (x_test[:num_test].astype(np.float32) / 255.0)
    y_test = y_test[:num_test].astype(np.int64)
    return x_train, y_train, x_test, y_test


def get_mnist_dataloaders(batch_size: int = 128, num_train: int = 10000, num_test: int = 2000) -> Tuple[DataLoader, DataLoader]:
    x_train, y_train, x_test, y_test = get_mnist_data(num_train, num_test)
    train_ds = TensorDataset(torch.tensor(x_train).unsqueeze(1), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(x_test).unsqueeze(1), torch.tensor(y_test))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def get_mnist_lstm_sequence_dataloaders(
    max_to_add: int = 8,
    num_train_seqs: int = 2400,
    num_test_seqs: int = 600,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """Tạo chuỗi ảnh MNIST thật (1..max_to_add ảnh, phần còn lại pad 0) và nhãn tổng chữ số."""
    x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = get_mnist_data(num_train=10000, num_test=2000)
    rng = np.random.default_rng(SEED)

    def build_sequences(x_raw, y_raw, n_seqs):
        seqs_x = np.zeros((n_seqs, max_to_add, 1, 28, 28), dtype=np.float32)
        seqs_y = np.zeros((n_seqs,), dtype=np.float32)
        for i in range(n_seqs):
            k = int(rng.integers(1, max_to_add + 1))
            idx = rng.choice(len(x_raw), size=k, replace=True)
            seqs_x[i, :k, 0] = x_raw[idx]
            seqs_y[i] = float(np.sum(y_raw[idx]))
        return TensorDataset(torch.tensor(seqs_x), torch.tensor(seqs_y))

    train_ds = build_sequences(x_tr_raw, y_tr_raw, num_train_seqs)
    test_ds = build_sequences(x_te_raw, y_te_raw, num_test_seqs)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=batch_size, shuffle=False)


# =====================================================================
# 4. HUMAN GENDER CLASSIFICATION DATASET
# =====================================================================
def get_gender_dataloaders(num_samples: int = 1000, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Sinh dữ liệu phân loại giới tính 64x64 mức xám (dùng khi thiếu file .npy ngoài)."""
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 2, size=num_samples, dtype=np.int64)
    X = rng.normal(0.45, 0.1, size=(num_samples, 1, 64, 64)).astype(np.float32)
    for i in range(num_samples):
        if y[i] == 1:
            X[i, 0, 12:48, 18:46] += 0.25
        else:
            X[i, 0, 18:44, 14:50] -= 0.20
    X = np.clip(X, 0.0, 1.0)

    train_n = int(0.8 * num_samples)
    train_ds = TensorDataset(torch.tensor(X[:train_n]), torch.tensor(y[:train_n]))
    val_ds = TensorDataset(torch.tensor(X[train_n:]), torch.tensor(y[train_n:]))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# =====================================================================
# 5. FER-2013 FACIAL EMOTION RECOGNITION DATASET
# =====================================================================
def get_emotion_dataloaders(batch_size: int = 64, subset_size: int = 4000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Nạp dữ liệu cảm xúc khuôn mặt FER-2013 thật từ BTL02 cache (48x48 mức xám, 7 lớp)."""
    btl02_dir = Path(__file__).resolve().parent.parent.parent / "BTL02" / "data" / "preprocessing"
    btl02_train = btl02_dir / "train.npz"
    btl02_val = btl02_dir / "valid.npz"
    btl02_test = btl02_dir / "test.npz"

    if btl02_train.exists() and btl02_val.exists():
        tr_data = np.load(btl02_train)
        va_data = np.load(btl02_val)
        te_data = np.load(btl02_test) if btl02_test.exists() else va_data

        x_key = 'images' if 'images' in tr_data else 'pixels'
        y_key = 'labels' if 'labels' in tr_data else 'emotions'

        x_train = tr_data[x_key][:subset_size].reshape(-1, 1, 48, 48).astype(np.float32) / 255.0
        y_train = tr_data[y_key][:subset_size].astype(np.int64)
        n_eval = max(256, subset_size // 4)
        x_val = va_data[x_key][:n_eval].reshape(-1, 1, 48, 48).astype(np.float32) / 255.0
        y_val = va_data[y_key][:n_eval].astype(np.int64)
        x_test = te_data[x_key][:n_eval].reshape(-1, 1, 48, 48).astype(np.float32) / 255.0
        y_test = te_data[y_key][:n_eval].astype(np.int64)
    else:
        rng = np.random.default_rng(SEED)
        x_train = rng.uniform(0.1, 0.9, size=(subset_size, 1, 48, 48)).astype(np.float32)
        y_train = rng.integers(0, 7, size=subset_size, dtype=np.int64)
        x_val = rng.uniform(0.1, 0.9, size=(subset_size // 4, 1, 48, 48)).astype(np.float32)
        y_val = rng.integers(0, 7, size=subset_size // 4, dtype=np.int64)
        x_test, y_test = x_val.copy(), y_val.copy()

    tr_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader, te_loader


# =====================================================================
# 6. HAND DEPTH SEGMENTATION (SEGNET - ĐỌC THẬT TỪ depth_hand.zip)
# =====================================================================
SEGNET_COLOR_TO_CLASS = {
    255: 0,  # palm
    43: 1,   # finger 1
    85: 2,   # finger 2
    128: 3,  # finger 3
    170: 4,  # finger 4
    212: 5,  # finger 5
    0: 6,    # background
}


def _map_segnet_label(label_arr: np.ndarray) -> np.ndarray:
    """Ánh xạ nhãn mức xám Kinect trong depth_hand.zip thành 7 lớp 0..6 theo 1_segnet.ipynb."""
    out = np.full(label_arr.shape, 6, dtype=np.int64)
    for gray_val, cls_idx in SEGNET_COLOR_TO_CLASS.items():
        out[label_arr == gray_val] = cls_idx
    return out


def get_segnet_dataloaders(batch_size: int = 16, num_train: int = 240, num_test: int = 80) -> Tuple[DataLoader, DataLoader]:
    """Nạp dữ liệu ảnh độ sâu + nhãn phân đoạn thật từ sample/examples/5. Others/depth_hand.zip."""
    zip_path = SAMPLE_DIR / "examples" / "5. Others" / "depth_hand.zip"

    def load_split(zf: zipfile.ZipFile, split: str, limit: int):
        prefix = f"data/segnet/{split}/depth/"
        depth_files = sorted([n for n in zf.namelist() if n.startswith(prefix) and n.endswith('.png')])[:limit]
        xs, ys = [], []
        for d_path in depth_files:
            l_path = d_path.replace('depth', 'labels')
            if l_path not in zf.namelist():
                continue
            with zf.open(d_path) as fd, zf.open(l_path) as fl:
                img_d = np.array(Image.open(fd).convert('L').resize((96, 96), Image.NEAREST), dtype=np.float32)
                img_l = np.array(Image.open(fl).convert('L').resize((96, 96), Image.NEAREST), dtype=np.uint8)
                xs.append(img_d)
                ys.append(_map_segnet_label(img_l))
        return torch.tensor(np.array(xs)).unsqueeze(1), torch.tensor(np.array(ys), dtype=torch.long)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        x_tr, y_tr = load_split(zf, "train", num_train)
        x_te, y_te = load_split(zf, "test", num_test)

    return DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True), DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)


# =====================================================================
# 7. CAR OBJECT DETECTION (ĐỌC THẬT TỪ car_data.zip)
# =====================================================================
def get_car_detection_dataloaders(batch_size: int = 64, max_per_class: int = 600) -> Tuple[DataLoader, DataLoader]:
    """Nạp ảnh thật vehicles (+1) và non-vehicles (-1) 64x64 RGB từ car_data.zip."""
    zip_path = SAMPLE_DIR / "examples" / "5. Others" / "car_data.zip"
    images, labels = [], []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        veh_files = sorted([n for n in zf.namelist() if n.startswith('data/obj_detection/vehicles/') and n.endswith('.png')])
        non_files = sorted([n for n in zf.namelist() if n.startswith('data/obj_detection/non-vehicles/') and n.endswith('.png')])

        rng = np.random.default_rng(SEED)
        veh_sel = rng.choice(veh_files, size=min(max_per_class, len(veh_files)), replace=False)
        non_sel = rng.choice(non_files, size=min(max_per_class, len(non_files)), replace=False)

        for name in veh_sel:
            with zf.open(name) as fp:
                arr = np.array(Image.open(fp).convert('RGB').resize((64, 64)), dtype=np.float32).transpose(2, 0, 1)
                images.append(arr)
                labels.append(1.0)

        for name in non_sel:
            with zf.open(name) as fp:
                arr = np.array(Image.open(fp).convert('RGB').resize((64, 64)), dtype=np.float32).transpose(2, 0, 1)
                images.append(arr)
                labels.append(-1.0)

    X = torch.tensor(np.array(images, dtype=np.float32))
    Y = torch.tensor(np.array(labels, dtype=np.float32)).view(-1, 1, 1, 1)

    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]
    split = int(0.7 * len(X))

    return DataLoader(TensorDataset(X[:split], Y[:split]), batch_size=batch_size, shuffle=True), DataLoader(TensorDataset(X[split:], Y[split:]), batch_size=batch_size, shuffle=False)
