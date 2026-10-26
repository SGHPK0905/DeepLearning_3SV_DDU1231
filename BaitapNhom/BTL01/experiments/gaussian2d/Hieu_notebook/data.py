from pathlib import Path
import io
import numpy as np
import torch

MU = np.array([0.0, 0.0], dtype=np.float64)
C = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)
LOW = np.array([-2.0, -4.0], dtype=np.float64)
HIGH = np.array([2.0, 4.0], dtype=np.float64)
ROOT_SEED = 20261024
SPLIT_SIZES = {'train_full': 2000, 'val': 400, 'test': 100}
SPLIT_IDS = {'train_full': 0, 'val': 1, 'test': 2}


def gaussian_pdf_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[-1] != 2 or not np.isfinite(x).all():
        raise ValueError("x phải có dạng (..., 2) và không chứa NaN/Inf.")
    delta = x - MU
    quadratic = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(C), delta)
    return np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(np.linalg.det(C)))


def gaussian_pdf_torch(x):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float64)
    if x.dtype not in (torch.float32, torch.float64):
        x = x.to(torch.float64)
    if x.ndim == 0 or x.shape[-1] != 2 or not torch.isfinite(x).all():
        raise ValueError("x phải có dạng (..., 2) và không chứa NaN/Inf.")
    covariance = x.new_tensor(C.tolist())
    delta = x - x.new_tensor(MU.tolist())
    quadratic = torch.einsum("...i,ij,...j->...", delta, torch.linalg.inv(covariance), delta)
    return torch.exp(-0.5 * quadratic) / (2 * torch.pi * torch.sqrt(torch.linalg.det(covariance)))


def generate_dataset(n_samples, split):
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("Số mẫu phải là số nguyên dương.")
    if split not in SPLIT_IDS:
        raise ValueError("Split phải là train_full, val hoặc test.")
    seed = np.random.SeedSequence(ROOT_SEED, spawn_key=(SPLIT_IDS[split],))
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.uniform(LOW, HIGH, size=(n_samples, 2))
    y = gaussian_pdf_numpy(x).reshape(-1, 1)
    return x, y


def save_csv(path, x, y):
    buffer = io.StringIO()
    np.savetxt(buffer, np.column_stack((x, y)), delimiter=",", fmt="%.17g",
               header="x1,x2,y", comments="", newline="\n")
    content = buffer.getvalue().encode("utf-8")
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"{path.name} có nội dung khác. Hãy dùng thư mục mới.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "x1,x2,y":
            raise ValueError("CSV phải có header x1,x2,y.")
        table = np.loadtxt(handle, delimiter=",", ndmin=2)
    if table.shape[1] != 3 or len(table) == 0 or not np.isfinite(table).all():
        raise ValueError("CSV thiếu dữ liệu, sai số cột hoặc chứa NaN/Inf.")
    return table[:, :2].copy(), table[:, 2:3].copy()


def load_torch_csv(path):
    x, y = load_csv(path)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
