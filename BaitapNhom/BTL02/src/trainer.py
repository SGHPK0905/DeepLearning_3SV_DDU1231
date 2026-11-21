"""
BTL02: Training & Evaluation Pipeline for PyTorch CNNs
Includes:
- Training loop with gradient clipping and mixed precision option
- Validation loop with loss, top-1 accuracy tracking
- Model checkpointing (saving best_model.pt)
- Early stopping mechanism
- Detailed CSV logging per epoch
"""

import time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import DEVICE, CHECKPOINTS_DIR, LOGS_DIR


class FERTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        scheduler=None,
        device=DEVICE,
        checkpoint_name="best_model.pt"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time_sec': []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs=30, patience=8, log_csv_name=None, verbose=True):
        best_val_acc = -1.0
        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print(f"Starting training on device: {self.device} for {epochs} epochs (EarlyStopping patience: {patience})...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            epoch_time = time.time() - start_time

            current_lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(round(train_loss, 4))
            self.history['train_acc'].append(round(train_acc, 4))
            self.history['val_loss'].append(round(val_loss, 4))
            self.history['val_acc'].append(round(val_acc, 4))
            self.history['lr'].append(current_lr)
            self.history['epoch_time_sec'].append(round(epoch_time, 2))

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
                )

            # Checkpoint on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, self.checkpoint_path)
                if verbose:
                    print(f"  --> Saved new best checkpoint to {self.checkpoint_path.name} (Val Acc: {val_acc*100:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc*100:.2f}%")
                    break

        # Save training log to CSV
        if log_csv_name:
            csv_path = LOGS_DIR / log_csv_name
            pd.DataFrame(self.history).to_csv(csv_path, index=False)
            if verbose:
                print(f"Saved training history to {csv_path}")

        return self.history

    @torch.no_grad()
    def predict(self, dataloader):
        """
        Runs inference over dataloader and returns:
        all_preds (N,), all_targets (N,), all_probs (N, num_classes), all_images (N, 48, 48)
        """
        self.model.eval()
        all_preds, all_targets, all_probs, all_imgs = [], [], [], []

        for images, labels in dataloader:
            raw_imgs = images.cpu().numpy()[:, 0, :, :] # (B, 48, 48)
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.extend(probs)
            all_imgs.extend(raw_imgs)

        return (
            np.array(all_preds),
            np.array(all_targets),
            np.array(all_probs),
            np.array(all_imgs)
        )
