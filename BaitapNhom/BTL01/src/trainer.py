# -*- coding: utf-8 -*-
"""
src/trainer.py
----------------
Giai đoạn 2 - Vòng lặp huấn luyện / đánh giá / checkpoint PyTorch chuẩn (Thanh Phương phụ trách).

Đặc điểm quan trọng:
1. Huấn luyện batch_size = 32 thống nhất toàn dự án.
2. Quy trình chuẩn: model.train() -> zero_grad() -> forward -> loss -> backward() -> optimizer.step().
3. Đánh giá validation: model.eval() -> torch.no_grad().
4. Cơ chế lưu checkpoint: Luôn giữ trọng số của epoch có validation loss thấp nhất (best_state_dict)
   để đánh giá tập test trên mô hình tối ưu nhất.
5. Đường dẫn file checkpoint được giải quyết động qua pathlib.Path, không phụ thuộc máy cục bộ.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_optimizer, BASELINE
from src.metrics import evaluate_all


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    Bọc (x, y) NumPy thành PyTorch DataLoader.
    y được reshape về (-1, 1) cho khớp với output_dim = 1.
    """
    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(x_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@dataclass
class TrainHistory:
    """Lưu trữ lịch sử huấn luyện qua các epoch và thông tin mô hình tốt nhất."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    training_time_sec: float = 0.0
    best_state_dict: Optional[dict] = None


def train_model(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 300,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> TrainHistory:
    """
    Vòng lặp huấn luyện chuẩn PyTorch:
    - Huấn luyện theo mini-batch trên tập train.
    - Đánh giá val_loss sau mỗi epoch.
    - Checkpoint lại model tốt nhất theo val_loss.
    """
    model = model.to(device)
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_loader = make_dataloader(x_train, y_train, batch_size=batch_size, shuffle=True)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    history = TrainHistory()
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ---- Pha Train ----
        model.train()
        running_loss, n_samples = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        epoch_train_loss = running_loss / max(n_samples, 1)

        # ---- Pha Validation ----
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            epoch_val_loss = loss_fn(val_pred, y_val_t).item()

        history.train_loss.append(epoch_train_loss)
        history.val_loss.append(epoch_val_loss)

        # Lưu lại state_dict tốt nhất
        if epoch_val_loss < history.best_val_loss:
            history.best_val_loss = epoch_val_loss
            history.best_epoch = epoch
            history.best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                ckpt_p = Path(checkpoint_path)
                ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": epoch_val_loss,
                    },
                    str(ckpt_p),
                )

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"[Epoch {epoch:3d}/{epochs}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    history.training_time_sec = time.time() - t0
    return history


def load_best_checkpoint(model: torch.nn.Module, checkpoint_path: Union[str, Path], device: str = "cpu") -> torch.nn.Module:
    """Nạp lại trọng số tốt nhất đã lưu từ file checkpoint .pt."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_p}")
    ckpt = torch.load(str(ckpt_p), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def load_best_from_history(model: torch.nn.Module, history: TrainHistory, device: str = "cpu") -> torch.nn.Module:
    """Nạp lại trọng số tốt nhất trực tiếp từ bộ nhớ RAM (history.best_state_dict)."""
    if history.best_state_dict is None:
        raise RuntimeError("history.best_state_dict rỗng - train_model() chưa chạy epoch nào thành công.")
    model.load_state_dict(history.best_state_dict)
    return model.to(device)


def evaluate_on_test(model: torch.nn.Module, x_test: np.ndarray, y_test: np.ndarray, device: str = "cpu") -> dict:
    """Đánh giá mô hình trên tập test, trả về từ điển các chỉ số MSE, MAE, RMSE, R^2."""
    model.eval()
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(x_test_t).cpu().numpy().reshape(-1)
    return evaluate_all(y_test, y_pred)

