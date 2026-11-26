"""
Trích xuất và lưu sẵn (cache) đặc trưng ảnh từ ResNet-50 pretrained (IMAGENET1K_V2).
Giúp huấn luyện Baseline (global 2048-d) và Bahdanau Attention (spatial 49x2048)
nhanh gấp 10-30 lần và tiết kiệm VRAM (chạy mượt trên RTX 3050 4GB hoặc Colab T4).
"""

import argparse
import os
import time
from typing import Dict

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

try:
    from src.config import Config
    from src.dataset import get_image_transforms
except ImportError:
    from config import Config
    from dataset import get_image_transforms


class _UniqueImageDataset(Dataset):
    def __init__(self, image_names, image_dir, transform):
        self.image_names = list(image_names)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        return name, self.transform(img)


@torch.no_grad()
def extract_resnet50_features(mode: str = "both", batch_size: int = 64) -> None:
    captions_csv = os.path.join(Config.PROCESSED_DATA_DIR, "captions_clean.csv")
    if not os.path.exists(captions_csv):
        raise FileNotFoundError("Chưa có captions_clean.csv. Hãy chạy 'python src/eda_preprocess.py' trước.")

    df = pd.read_csv(captions_csv)
    unique_images = sorted(df["image"].unique().tolist())
    img_dir = os.path.join(Config.RAW_DATA_DIR, "images")

    ds = _UniqueImageDataset(unique_images, img_dir, get_image_transforms(is_train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = Config.DEVICE
    print(f"Đang trích xuất đặc trưng ResNet-50 ({mode}) cho {len(unique_images)} ảnh trên thiết bị: {device}")

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove avgpool and fc -> outputs spatial map (B, 2048, 7, 7)
    conv_extractor = nn.Sequential(*list(backbone.children())[:-2]).to(device)
    avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
    conv_extractor.eval()

    global_feats: Dict[str, torch.Tensor] = {}
    spatial_feats: Dict[str, torch.Tensor] = {}

    t0 = time.time()
    for names, imgs in tqdm(loader, desc="Extracting ResNet-50"):
        imgs = imgs.to(device)
        feat_map = conv_extractor(imgs)  # (B, 2048, 7, 7)

        if mode in {"global", "both"}:
            pooled = avg_pool(feat_map).flatten(1).half().cpu()  # (B, 2048) float16
            for n, vec in zip(names, pooled):
                global_feats[n] = vec.clone()

        if mode in {"spatial", "both"}:
            # (B, 2048, 7, 7) -> (B, 49, 2048) float16
            spatial = feat_map.flatten(2).permute(0, 2, 1).half().cpu()
            for n, mat in zip(names, spatial):
                spatial_feats[n] = mat.clone()

    elapsed = time.time() - t0
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

    if mode in {"global", "both"}:
        out_g = os.path.join(Config.PROCESSED_DATA_DIR, "features_resnet50_global.pt")
        torch.save(global_feats, out_g)
        sz_mb = os.path.getsize(out_g) / (1024 * 1024)
        sample_g = next(iter(global_feats.values())).shape
        print(f"Đã lưu Global features: {out_g} | shape={tuple(sample_g)} | {sz_mb:.1f} MB")

    if mode in {"spatial", "both"}:
        out_s = os.path.join(Config.PROCESSED_DATA_DIR, "features_resnet50_spatial.pt")
        torch.save(spatial_feats, out_s)
        sz_mb = os.path.getsize(out_s) / (1024 * 1024)
        sample_s = next(iter(spatial_feats.values())).shape
        print(f"Đã lưu Spatial features: {out_s} | shape={tuple(sample_s)} | {sz_mb:.1f} MB")

    print(f"Hoàn tất trích xuất sau {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Cache đặc trưng ResNet-50 cho Flickr8k")
    parser.add_argument("--mode", choices=["global", "spatial", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    extract_resnet50_features(mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
