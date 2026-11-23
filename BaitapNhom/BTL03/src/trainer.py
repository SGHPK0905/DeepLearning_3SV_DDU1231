"""
BTL03: Bộ khung Huấn luyện chuẩn trong PyTorch (Trainer Engine)
Tái hiện toàn diện các tiện ích Keras Callbacks:
1. EarlyStopping: Dừng sớm tránh quá khớp
2. ModelCheckpoint: Lưu trọng số mô hình tốt nhất
3. ReduceLROnPlateau: Điều chỉnh learning rate thích nghi
"""

import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEVICE, CHECKPOINTS_DIR


class EarlyStopping:
    """
    Tương đương keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    Theo dõi hàm mất mát kiểm định và dừng vòng lặp nếu không cải thiện sau số epoch chỉ định.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
        return self.early_stop


class ModelCheckpoint:
    """
    Tương đương keras.callbacks.ModelCheckpoint(filepath, save_best_only=True)
    """
    def __init__(self, filepath: Path, save_best_only: bool = True):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_loss = float('inf')

    def __call__(self, val_loss: float, model: nn.Module):
        if not self.save_best_only or val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device = DEVICE,
    is_regression: bool = False
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)

        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if not is_regression and preds.dim() > 1 and preds.size(1) > 1:
            predicted_labels = torch.argmax(preds, dim=1)
            # Hỗ trợ cả y_batch dạng 1D index hoặc 2D/3D (như SegNet)
            if y_batch.dim() == 1:
                correct += (predicted_labels == y_batch).sum().item()
            elif y_batch.dim() == 3: # SegNet (B, H, W)
                correct += (predicted_labels == y_batch).sum().item() / (y_batch.size(1) * y_batch.size(2))

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = (correct / total_samples) if (not is_regression and total_samples > 0) else 0.0
    return avg_loss, avg_acc


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = DEVICE,
    is_regression: bool = False
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if not is_regression and preds.dim() > 1 and preds.size(1) > 1:
                predicted_labels = torch.argmax(preds, dim=1)
                if y_batch.dim() == 1:
                    correct += (predicted_labels == y_batch).sum().item()
                elif y_batch.dim() == 3:
                    correct += (predicted_labels == y_batch).sum().item() / (y_batch.size(1) * y_batch.size(2))

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = (correct / total_samples) if (not is_regression and total_samples > 0) else 0.0
    return avg_loss, avg_acc


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 15,
    device: torch.device = DEVICE,
    is_regression: bool = False,
    callbacks: Optional[List[Callable]] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    verbose: bool = True
) -> Dict[str, list]:
    """
    Vòng lặp huấn luyện PyTorch tiêu chuẩn tương đương model.fit(...) của Keras.
    """
    model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'epoch_times': []
    }

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, is_regression)
        va_loss, va_acc = validate_one_epoch(model, val_loader, criterion, device, is_regression)
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['lr'].append(curr_lr)
        history['epoch_times'].append(epoch_time)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs or epoch == 1):
            if is_regression:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - {epoch_time:.2f}s - Train Loss: {tr_loss:.4e} - Val Loss: {va_loss:.4e} - LR: {curr_lr:.1e}")
            else:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - {epoch_time:.2f}s - Train Loss: {tr_loss:.4f} (Acc: {tr_acc*100:.2f}%) - Val Loss: {va_loss:.4f} (Acc: {va_acc*100:.2f}%) - LR: {curr_lr:.1e}")

        # Xử lý Callbacks
        if callbacks:
            should_stop = False
            for cb in callbacks:
                if isinstance(cb, EarlyStopping):
                    if cb(va_loss, model):
                        if verbose:
                            print(f"--> [EarlyStopping] Dừng sớm tại Epoch {epoch} với Val Loss tối ưu = {cb.best_loss:.4f}")
                        should_stop = True
                elif isinstance(cb, ModelCheckpoint):
                    cb(va_loss, model)
            if should_stop:
                break

    history['total_time'] = time.time() - start_total_time
    return history
