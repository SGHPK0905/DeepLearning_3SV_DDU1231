"""
PyTorch Dataset, DataLoader, Transforms và Vocabulary loader cho Flickr8k.
Hỗ trợ cả nạp ảnh thô (dùng khi fine-tune / ViT) và nạp đặc trưng ResNet-50 đã cache.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from src.config import Config
    from src.eda_preprocess import Vocabulary
except ImportError:
    from config import Config
    from eda_preprocess import Vocabulary


class _VocabUnpickler(pickle.Unpickler):
    """Đảm bảo load được vocab.pkl bất kể được pickle từ __main__ hay eda_preprocess."""

    def find_class(self, module, name):
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_vocab(vocab_path: str = os.path.join(Config.PROCESSED_DATA_DIR, "vocab.pkl")) -> Vocabulary:
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Không tìm thấy {vocab_path}. Hãy chạy 'python src/eda_preprocess.py' trước."
        )
    with open(vocab_path, "rb") as f:
        return _VocabUnpickler(f).load()


def get_image_transforms(is_train: bool = False) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize(Config.RESIZE_SIZE),
            transforms.RandomCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(Config.RESIZE_SIZE),
        transforms.CenterCrop(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
    ])


def encode_caption(caption: str, vocab: Vocabulary, max_len: int = Config.MAX_CAPTION_LEN) -> torch.Tensor:
    tokens = vocab.numericalize(caption)
    # Truncate leaving room for <start> and <end>
    tokens = tokens[: max_len - 2]
    seq = [vocab.stoi[Config.START_TOKEN]] + tokens + [vocab.stoi[Config.END_TOKEN]]
    return torch.tensor(seq, dtype=torch.long)


class Flickr8kImageDataset(Dataset):
    """Dataset nạp trực tiếp ảnh RGB + chuỗi caption đã mã hóa (cho fine-tuning / ViT)."""

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        vocab: Vocabulary,
        transform: Optional[transforms.Compose] = None,
        max_len: int = Config.MAX_CAPTION_LEN,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform or get_image_transforms(is_train=False)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        cap_tensor = encode_caption(row["caption"], self.vocab, self.max_len)
        return img_tensor, cap_tensor


class Flickr8kCachedFeatureDataset(Dataset):
    """Dataset nạp nhanh từ đặc trưng ResNet-50 đã cache (global 2048-d hoặc spatial 49x2048)."""

    def __init__(
        self,
        csv_path: str,
        features_dict: Dict[str, torch.Tensor],
        vocab: Vocabulary,
        max_len: int = Config.MAX_CAPTION_LEN,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.features_dict = features_dict
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        feat = self.features_dict[row["image"]].float()
        cap_tensor = encode_caption(row["caption"], self.vocab, self.max_len)
        return feat, cap_tensor


class CaptionCollate:
    """Dynamic padding theo độ dài lớn nhất trong mỗi batch."""

    def __init__(self, pad_idx: int = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        visuals = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        padded_captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return visuals, padded_captions, lengths


def build_dataloaders(
    use_cached_features: bool = True,
    feature_mode: str = "global",
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
):
    """Tạo train_loader, val_loader, từ điển vocab và bảng gom nhóm multi-reference cho val/test."""
    vocab = load_vocab()
    pad_idx = vocab.stoi[Config.PAD_TOKEN]
    collate_fn = CaptionCollate(pad_idx=pad_idx)

    train_csv = os.path.join(Config.PROCESSED_DATA_DIR, "train.csv")
    val_csv = os.path.join(Config.PROCESSED_DATA_DIR, "val.csv")

    if use_cached_features:
        feat_file = (
            "features_resnet50_global.pt"
            if feature_mode == "global"
            else "features_resnet50_spatial.pt"
        )
        feat_path = os.path.join(Config.PROCESSED_DATA_DIR, feat_file)
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Không tìm thấy {feat_path}. Hãy chạy 'python src/extract_features.py --mode {feature_mode}' trước."
            )
        features_dict = torch.load(feat_path, map_location="cpu", weights_only=True)
        train_ds = Flickr8kCachedFeatureDataset(train_csv, features_dict, vocab)
        val_ds = Flickr8kCachedFeatureDataset(val_csv, features_dict, vocab)
    else:
        img_dir = os.path.join(Config.RAW_DATA_DIR, "images")
        train_ds = Flickr8kImageDataset(train_csv, img_dir, vocab, transform=get_image_transforms(True))
        val_ds = Flickr8kImageDataset(val_csv, img_dir, vocab, transform=get_image_transforms(False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, vocab
