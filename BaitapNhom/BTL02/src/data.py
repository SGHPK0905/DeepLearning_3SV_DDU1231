"""
BTL02: Data Processing & Augmentation Module for FER2013
Supports caching to train.npz, valid.npz, test.npz (matching sample code format),
and PyTorch Dataset/DataLoader with configurable augmentations.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate

from .config import (
    RAW_CSV_PATH,
    TRAIN_NPZ_PATH,
    VALID_NPZ_PATH,
    TEST_NPZ_PATH,
    PREPROCESSED_DIR,
    SEED
)


def extract_and_cache_fer2013(csv_path=RAW_CSV_PATH, force=False):
    """
    Extracts raw fer2013.csv and caches into train.npz, valid.npz, test.npz.
    """
    if not force and TRAIN_NPZ_PATH.exists() and VALID_NPZ_PATH.exists() and TEST_NPZ_PATH.exists():
        print("Preprocessed .npz files already exist. Skipping extraction.")
        return

    print(f"Reading raw CSV: {csv_path} (this may take a few seconds)...")
    df = pd.read_csv(csv_path)

    for split_name, save_path in [
        ('Training', TRAIN_NPZ_PATH),
        ('PublicTest', VALID_NPZ_PATH),
        ('PrivateTest', TEST_NPZ_PATH)
    ]:
        sub_df = df[df['Usage'] == split_name]
        print(f"Processing split '{split_name}': {len(sub_df)} samples...")
        
        # Parse pixel strings to uint8 array
        pixels_list = []
        for p_str in sub_df['pixels']:
            arr = np.fromstring(p_str, dtype=np.uint8, sep=' ')
            pixels_list.append(arr.reshape(48, 48))
        
        images = np.stack(pixels_list, axis=0) # (N, 48, 48)
        labels = sub_df['emotion'].values.astype(np.int64) # (N,)

        np.savez_compressed(save_path, images=images, labels=labels)
        print(f"Saved {split_name} to {save_path}: images shape={images.shape}, labels shape={labels.shape}")

    print("Data extraction and caching completed successfully!")


def load_data(data_dir=PREPROCESSED_DIR):
    """
    Loads preprocessed train, valid, and test sets from .npz files.
    Matches sample code: (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    """
    train_path = Path(data_dir) / "train.npz"
    valid_path = Path(data_dir) / "valid.npz"
    test_path = Path(data_dir) / "test.npz"

    if not (train_path.exists() and valid_path.exists() and test_path.exists()):
        extract_and_cache_fer2013()

    train_data = np.load(train_path)
    valid_data = np.load(valid_path)
    test_data = np.load(test_path)

    x_train, y_train = train_data['images'], train_data['labels']
    x_valid, y_valid = valid_data['images'], valid_data['labels']
    x_test, y_test = test_data['images'], test_data['labels']

    print(f"Loaded Train: {len(x_train)} images, Valid: {len(x_valid)} images, Test: {len(x_test)} images.")
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


class FERDataset(Dataset):
    """
    PyTorch Dataset for FER2013 with configurable data augmentation options.
    Augmentation modes:
      - 'none': only normalization
      - 'basic': random horizontal flip
      - 'advanced': random horizontal flip, random rotation (+-10 deg), random crop/shift (+-4 px), random erasing
    """
    def __init__(self, images, labels, augment_mode='none'):
        self.images = images # (N, 48, 48) uint8
        self.labels = labels # (N,) int64
        self.augment_mode = augment_mode

    def __len__(self):
        return len(self.images)

    def _apply_augmentations(self, img):
        """
        img: (48, 48) float in [0, 1]
        """
        # 1. Random horizontal flip (p=0.5)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        if self.augment_mode == 'advanced':
            # 2. Random rotation (+-10 degrees)
            angle = np.random.uniform(-10.0, 10.0)
            img = rotate(img, angle, reshape=False, mode='nearest', order=1)

            # 3. Random shift (+-4 pixels)
            shift_x = np.random.randint(-4, 5)
            shift_y = np.random.randint(-4, 5)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)

            # 4. Random Erasing / Cutout (p=0.3)
            if np.random.rand() < 0.3:
                h_cut, w_cut = np.random.randint(6, 12), np.random.randint(6, 12)
                top = np.random.randint(0, 48 - h_cut)
                left = np.random.randint(0, 48 - w_cut)
                img[top:top+h_cut, left:left+w_cut] = 0.5

        return img

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # Scale to [0, 1]

        if self.augment_mode in ['basic', 'advanced']:
            img = self._apply_augmentations(img)

        # Standard score: (img - 0.5) / 0.5 -> [-1.0, 1.0]
        img = (img - 0.5) / 0.5

        # PyTorch format: (C, H, W) -> (1, 48, 48)
        tensor_img = torch.from_numpy(img).unsqueeze(0).float()
        tensor_label = torch.tensor(self.labels[idx], dtype=torch.long)

        return tensor_img, tensor_label


def get_dataloaders(batch_size=64, augment_mode='advanced', num_workers=0, subset_size=None):
    """
    Constructs train, validation, and test PyTorch DataLoaders.
    If subset_size is provided, randomly samples subset_size training samples for fast experiments.
    """
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data()

    if subset_size is not None and subset_size < len(x_train):
        np.random.seed(SEED)
        indices = np.random.choice(len(x_train), size=subset_size, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]
        print(f"Sampled training subset: {len(x_train)} images for efficient experiment execution.")

    train_ds = FERDataset(x_train, y_train, augment_mode=augment_mode)
    val_ds = FERDataset(x_valid, y_valid, augment_mode='none')
    test_ds = FERDataset(x_test, y_test, augment_mode='none')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
