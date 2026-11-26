"""
Download and standardize Flickr8k raw dataset into data/raw/.
Supports:
  1) Hugging Face `jxie/flickr8k` (default, no Kaggle API token required)
  2) Local Kaggle zip (`adityajn105/flickr8k`) via --source kaggle_zip --zip_path ...
"""

import argparse
import os
import zipfile

# Fix Xet hang/freeze on Windows & suppress symlink warning before importing huggingface/datasets
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
from tqdm import tqdm

try:
    from src.config import Config
except ImportError:
    from config import Config


def save_from_huggingface(raw_dir: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện 'datasets'. Cài đặt bằng: pip install datasets"
        ) from exc

    img_dir = os.path.join(raw_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("Đang tải bộ dữ liệu 'jxie/flickr8k' từ Hugging Face...")
    ds_dict = load_dataset("jxie/flickr8k")
    print("Các split có sẵn trên HF:", {k: len(v) for k, v in ds_dict.items()})

    records = []
    global_idx = 0
    for split_name, ds in ds_dict.items():
        norm_split = "val" if split_name == "validation" else split_name
        cols = ds.column_names
        caption_cols = sorted([c for c in cols if c.startswith("caption")])

        for row in tqdm(ds, desc=f"Xuất ảnh & captions ({norm_split})"):
            # Determine stable image filename
            img_name = None
            for cand in ["image_id", "filename", "file_name", "img_id"]:
                if cand in row and row[cand]:
                    val = str(row[cand]).strip()
                    img_name = val if val.lower().endswith((".jpg", ".jpeg", ".png")) else f"{val}.jpg"
                    break
            if img_name is None:
                img_name = f"flickr8k_{global_idx:05d}.jpg"

            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                pil_img = row["image"]
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(img_path, format="JPEG", quality=95)

            # Extract all captions for this image
            captions = []
            for ccol in caption_cols:
                val = row[ccol]
                if isinstance(val, list):
                    captions.extend([str(x).strip() for x in val if str(x).strip()])
                elif val is not None and str(val).strip():
                    captions.append(str(val).strip())

            for cap in captions:
                records.append({
                    "image": img_name,
                    "caption": cap,
                    "split": norm_split if norm_split in {"train", "val", "test"} else "",
                })
            global_idx += 1

    df = pd.DataFrame(records)
    out_csv = os.path.join(raw_dir, "captions.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Đã lưu {df['image'].nunique()} ảnh vào {img_dir}")
    print(f"Đã lưu {len(df)} dòng captions vào {out_csv}")


def extract_from_kaggle_zip(zip_path: str, raw_dir: str) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Không tìm thấy file zip: {zip_path}")

    os.makedirs(raw_dir, exist_ok=True)
    print(f"Đang giải nén {zip_path} vào {raw_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # Normalize directory/filename if Kaggle archive uses `Images/` and `captions.txt`
    kaggle_imgs = os.path.join(raw_dir, "Images")
    target_imgs = os.path.join(raw_dir, "images")
    if os.path.exists(kaggle_imgs) and not os.path.exists(target_imgs):
        os.rename(kaggle_imgs, target_imgs)

    txt_path = os.path.join(raw_dir, "captions.txt")
    csv_path = os.path.join(raw_dir, "captions.csv")
    if os.path.exists(txt_path) and not os.path.exists(csv_path):
        df = pd.read_csv(txt_path)
        df["split"] = ""
        df.to_csv(csv_path, index=False, encoding="utf-8")
    print("Giải nén & chuẩn hóa Kaggle Flickr8k hoàn tất!")


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu Flickr8k về data/raw/")
    parser.add_argument("--source", choices=["hf", "kaggle_zip"], default="hf")
    parser.add_argument("--zip_path", default="", help="Đường dẫn tới archive.zip nếu tải tay từ Kaggle")
    args = parser.parse_args()

    os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
    if args.source == "hf":
        save_from_huggingface(Config.RAW_DATA_DIR)
    else:
        extract_from_kaggle_zip(args.zip_path, Config.RAW_DATA_DIR)


if __name__ == "__main__":
    main()
