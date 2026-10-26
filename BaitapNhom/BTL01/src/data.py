# -*- coding: utf-8 -*-
"""
src/data.py
-----------
Giai đoạn 1 - Pipeline Dữ liệu Gaussian 2 chiều (Trung Hiếu phụ trách).

Nhiệm vụ:
1. Lập trình hàm mật độ xác suất Gaussian 2D giải tích đúng công thức toán học và ma trận C:
   mu = [0, 0]^T
   C = [[0.3, -0.5], [-0.5, 2.0]]
   det(C) = 0.35
2. Sinh ngẫu nhiên dữ liệu trong miền: -2 <= x1 <= 2, -4 <= x2 <= 4.
3. Sinh độc lập và xuất ra CSV:
   - train_full.csv: 2000 mẫu
   - val.csv: 400 mẫu
   - test.csv: 100 mẫu độc lập (không rò rỉ)
"""

from pathlib import Path
import io
import sys
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đường dẫn tự động giải quyết độc lập trên mọi hệ điều hành
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Các hằng số tham số bài toán theo đúng đề bài
MU = np.array([0.0, 0.0], dtype=np.float64)
C = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)
LOW = np.array([-2.0, -4.0], dtype=np.float64)
HIGH = np.array([2.0, 4.0], dtype=np.float64)

# Seed gốc và định nghĩa quy mô tập mẫu
ROOT_SEED = 20261024
SPLIT_SIZES = {"train_full": 2000, "val": 400, "test": 100}
SPLIT_IDS = {"train_full": 0, "val": 1, "test": 2}


def gaussian_pdf_numpy(x: np.ndarray) -> np.ndarray:
    """
    Tính hàm mật độ xác suất Gaussian 2 chiều bằng NumPy giải tích.
    Hỗ trợ đầu vào shape (..., 2).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] != 2 or not np.isfinite(x).all():
        raise ValueError("x phải có dạng (..., 2) và không chứa NaN/Inf.")
    delta = x - MU
    inv_c = np.linalg.inv(C)
    det_c = np.linalg.det(C)
    quadratic = np.einsum("...i,ij,...j->...", delta, inv_c, delta)
    return np.exp(-0.5 * quadratic) / (2.0 * np.pi * np.sqrt(det_c))


def gaussian_pdf_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Tính hàm mật độ xác suất Gaussian 2 chiều bằng PyTorch Tensor.
    Hỗ trợ đầu vào Tensor shape (..., 2).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    if x.dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float64)
    if x.ndim == 0 or x.shape[-1] != 2 or not torch.isfinite(x).all():
        raise ValueError("x phải có dạng (..., 2) và không chứa NaN/Inf.")

    covariance = x.new_tensor(C.tolist())
    delta = x - x.new_tensor(MU.tolist())
    inv_cov = torch.linalg.inv(covariance)
    det_cov = torch.linalg.det(covariance)
    quadratic = torch.einsum("...i,ij,...j->...", delta, inv_cov, delta)
    return torch.exp(-0.5 * quadratic) / (2.0 * torch.pi * torch.sqrt(det_cov))


def generate_dataset(n_samples: int, split: str):
    """
    Sinh ngẫu nhiên n_samples điểm dữ liệu trong miền [-2, 2] x [-4, 4].
    Sử dụng SeedSequence với spawn_key riêng cho từng split để đảm bảo
    dữ liệu độc lập và không bao giờ rò rỉ/trùng lặp giữa train, val, test.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("Số mẫu phải là số nguyên dương.")
    if split not in SPLIT_IDS:
        raise ValueError(f"Split phải thuộc một trong các giá trị: {list(SPLIT_IDS.keys())}")

    seed = np.random.SeedSequence(ROOT_SEED, spawn_key=(SPLIT_IDS[split],))
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.uniform(LOW, HIGH, size=(n_samples, 2))
    y = gaussian_pdf_numpy(x).reshape(-1, 1)
    return x, y


def save_csv(path: Path, x: np.ndarray, y: np.ndarray):
    """Lưu dữ liệu x (N, 2) và y (N, 1) ra file CSV với độ chính xác số học cao."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    np.savetxt(
        buffer,
        np.column_stack((x, y)),
        delimiter=",",
        fmt="%.17g",
        header="x1,x2,y",
        comments="",
        newline="\n",
    )
    content = buffer.getvalue().encode("utf-8")
    path.write_bytes(content)


def load_csv(path: Path):
    """
    Đọc dữ liệu từ file CSV.
    Trả về x shape (N, 2) và y shape (N, 1) dưới dạng np.float64.
    """
    path = Path(path)
    if not path.is_file():
        alt_path = DATA_DIR / path.name
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    with path.open(encoding="utf-8-sig") as handle:
        header = handle.readline().strip()
        if header != "x1,x2,y":
            raise ValueError(f"CSV phải có header 'x1,x2,y', nhận được: {header}")
        table = np.loadtxt(handle, delimiter=",", ndmin=2)

    if table.shape[1] != 3 or len(table) == 0 or not np.isfinite(table).all():
        raise ValueError("CSV thiếu dữ liệu, sai số cột hoặc chứa NaN/Inf.")
    return table[:, :2].copy(), table[:, 2:3].copy()


def load_torch_csv(path: Path):
    """Đọc dữ liệu CSV và trả về cặp PyTorch FloatTensor (x, y)."""
    x, y = load_csv(path)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def generate_and_save_all_splits(output_dir: Path = DATA_DIR):
    """Sinh toàn bộ các tập train_full, val, test và lưu vào output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = {}
    for split, count in SPLIT_SIZES.items():
        file_path = output_dir / f"{split}.csv"
        x, y = generate_dataset(count, split)
        save_csv(file_path, x, y)
        created_files[split] = file_path
        print(f"[Data Generator] Đã tạo {file_path.name}: {count} mẫu | X: {x.shape} | y: {y.shape}")
    return created_files


if __name__ == "__main__":
    print(f"=== Đang sinh dữ liệu vào thư mục: {DATA_DIR} ===")
    generate_and_save_all_splits(DATA_DIR)
    print("=== Hoàn thành sinh dữ liệu Giai đoạn 1! ===")
