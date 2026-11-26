"""
Giai đoạn 1: Khám phá dữ liệu (EDA), kiểm soát Data Leakage, chia Train/Val/Test
và xây dựng Vocabulary (CHỈ fit trên tập Train với MIN_WORD_FREQ = 5).

Kế thừa các ý tưởng trực quan hóa tốt (Histogram độ dài, Top Tokens, Zipf's Law, WordCloud)
và khắc phục lỗi fit vocabulary toàn tập / đánh giá 1-1.
"""

import argparse
import os
import pickle
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

try:
    from src.config import Config
except ImportError:
    from config import Config


class Vocabulary:
    """Từ điển ánh xạ từ <-> index cho Image Captioning (PyTorch)."""

    def __init__(self, freq_threshold: int = Config.MIN_WORD_FREQ):
        self.freq_threshold = freq_threshold
        self.itos = {
            0: Config.PAD_TOKEN,
            1: Config.START_TOKEN,
            2: Config.END_TOKEN,
            3: Config.UNK_TOKEN,
        }
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str):
        return str(text).lower().strip().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            frequencies.update(self.tokenize(sentence))

        for word, count in frequencies.most_common():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        return frequencies

    def numericalize(self, text: str):
        tokens = self.tokenize(text)
        unk_idx = self.stoi[Config.UNK_TOKEN]
        return [self.stoi.get(tok, unk_idx) for tok in tokens]


def clean_english_caption(text: str) -> str:
    text = str(text).lower().strip()
    # Keep alphabetic words and basic spaces (standard Flickr8k English preprocessing)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_verify_raw(captions_path: str, image_dir: str):
    if not os.path.exists(captions_path):
        raise FileNotFoundError(
            f"Không tìm thấy {captions_path}. Hãy chạy 'python src/download_data.py' trước."
        )
    df = pd.read_csv(captions_path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError(f"File captions cần có cột 'image' và 'caption'. Hiện có: {df.columns.tolist()}")

    df["image"] = df["image"].astype(str).str.strip()
    df["caption_raw"] = df["caption"].astype(str).str.strip()
    df["caption"] = df["caption_raw"].apply(clean_english_caption)
    df = df[df["caption"].str.len() > 0].drop_duplicates(subset=["image", "caption"]).reset_index(drop=True)

    df["exists"] = df["image"].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))
    missing = df[~df["exists"]].drop(columns=["exists"])
    valid = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    valid["num_tokens"] = valid["caption"].apply(lambda s: len(s.split()))
    return valid, missing


def split_by_image(df: pd.DataFrame, seed: int = Config.SEED):
    """Chia tập theo image_id duy nhất để đảm bảo 0 Data Leakage."""
    has_official = (
        "split" in df.columns
        and set(df["split"].dropna().unique()).issuperset({"train", "val", "test"})
        and (df["split"] == "train").any()
    )

    if has_official:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        test_df = df[df["split"] == "test"].copy()
        split_mode = "official_karpathy"
    else:
        unique_imgs = df["image"].drop_duplicates().tolist()
        n_total = len(unique_imgs)
        if n_total >= 8000:
            # Standard Flickr8k Karpathy-style ratio: 6000 / 1000 / 1000 (or remainder)
            val_test_size = 2000 / n_total
            train_imgs, temp_imgs = train_test_split(unique_imgs, test_size=val_test_size, random_state=seed)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)
        else:
            train_imgs, temp_imgs = train_test_split(unique_imgs, test_size=0.2, random_state=seed)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)

        train_df = df[df["image"].isin(train_imgs)].copy()
        val_df = df[df["image"].isin(val_imgs)].copy()
        test_df = df[df["image"].isin(test_imgs)].copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        split_mode = f"image_level_seed_{seed}"

    # Anti-data-leakage assertions
    s_tr = set(train_df["image"].unique())
    s_va = set(val_df["image"].unique())
    s_te = set(test_df["image"].unique())
    assert len(s_tr & s_va) == 0, "Data leakage giữa Train và Val!"
    assert len(s_tr & s_te) == 0, "Data leakage giữa Train và Test!"
    assert len(s_va & s_te) == 0, "Data leakage giữa Val và Test!"
    return train_df, val_df, test_df, split_mode


def generate_eda_figures(df_all, train_df, val_df, test_df, train_freq: Counter, image_dir: str, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    # 1. Caption Length Distribution
    plt.figure(figsize=(8.5, 4.8))
    plt.hist(df_all["num_tokens"], bins=range(1, int(df_all["num_tokens"].max()) + 3), color="#2A558C", edgecolor="white", alpha=0.88)
    mean_len = df_all["num_tokens"].mean()
    med_len = df_all["num_tokens"].median()
    p99_len = np.percentile(df_all["num_tokens"], 99)
    plt.axvline(mean_len, color="#D97706", linestyle="--", linewidth=1.6, label=f"Mean = {mean_len:.2f}")
    plt.axvline(med_len, color="#059669", linestyle=":", linewidth=1.8, label=f"Median = {med_len:.1f}")
    plt.axvline(Config.MAX_CAPTION_LEN, color="#DC2626", linestyle="-.", linewidth=1.6, label=f"MAX_CAPTION_LEN = {Config.MAX_CAPTION_LEN} (P99={p99_len:.0f})")
    plt.title("Phân phối độ dài câu chú thích trên tập Flickr8k")
    plt.xlabel("Số lượng từ (tokens / caption)")
    plt.ylabel("Số lượng câu chú thích")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_caption_length_distribution.png"), dpi=300)
    plt.close()

    # 2. Top 25 Tokens (Train set ONLY)
    top25 = train_freq.most_common(25)
    labels, values = zip(*top25)
    plt.figure(figsize=(10.5, 5.2))
    plt.bar(labels, values, color="#1B365D", alpha=0.9)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 25 từ xuất hiện nhiều nhất (chỉ thống kê trên tập Train)")
    plt.xlabel("Từ vựng (Token)")
    plt.ylabel("Tần suất xuất hiện")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_top_tokens.png"), dpi=300)
    plt.close()

    # 3. Zipf's Law (Train set ONLY)
    freqs = sorted(train_freq.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(ranks, freqs, color="#2A558C", linewidth=1.8, label="Train Vocab Distribution")
    plt.axhline(Config.MIN_WORD_FREQ, color="#DC2626", linestyle="--", label=f"Ngưỡng lọc min_freq = {Config.MIN_WORD_FREQ}")
    plt.title("Kiểm tra phân phối từ vựng theo Định luật Zipf (Log-Log)")
    plt.xlabel("Thứ hạng tần suất — Rank (log)")
    plt.ylabel("Tần suất — Frequency (log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_zipf_law.png"), dpi=300)
    plt.close()

    # 4. WordCloud (Train set ONLY)
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",
        max_words=150,
        collocations=False,
        random_state=Config.SEED,
    ).generate_from_frequencies(train_freq)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud từ vựng đặc trưng tập Train Flickr8k", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_wordcloud.png"), dpi=300)
    plt.close()

    # 5. Sample Images + Ground-truth Captions (4 ảnh mẫu)
    sample_imgs = train_df["image"].drop_duplicates().sample(n=min(4, train_df["image"].nunique()), random_state=Config.SEED).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, img_name in zip(axes.flat, sample_imgs):
        img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        caps = train_df[train_df["image"] == img_name]["caption_raw"].tolist()[:3]
        ax.imshow(img)
        ax.axis("off")
        title_txt = f"{img_name}\n" + "\n".join([f"- {c[:70]}" for c in caps])
        ax.set_title(title_txt, fontsize=8.5, loc="left")
    plt.suptitle("Ảnh mẫu Flickr8k kèm câu chú thích gốc (Ground-truth)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_sample_images.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EDA & Preprocessing Flickr8k (Anti-Data-Leakage)")
    parser.add_argument("--captions", default=os.path.join(Config.RAW_DATA_DIR, "captions.csv"))
    parser.add_argument("--image_dir", default=os.path.join(Config.RAW_DATA_DIR, "images"))
    parser.add_argument("--out_dir", default=Config.PROCESSED_DATA_DIR)
    parser.add_argument("--fig_dir", default=Config.FIGURES_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    df_all, missing = load_and_verify_raw(args.captions, args.image_dir)
    missing.to_csv(os.path.join(args.out_dir, "missing_images.csv"), index=False)

    train_df, val_df, test_df, split_mode = split_by_image(df_all, seed=Config.SEED)

    # Fit Vocabulary ONLY on Train split!
    vocab = Vocabulary(freq_threshold=Config.MIN_WORD_FREQ)
    train_freq = vocab.build_vocabulary(train_df["caption"].tolist())

    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    pd.DataFrame(train_freq.most_common(), columns=["token", "freq"]).to_csv(
        os.path.join(args.out_dir, "vocab_frequency.csv"), index=False
    )

    df_all.to_csv(os.path.join(args.out_dir, "captions_clean.csv"), index=False)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    summary = {
        "dataset": Config.DATASET_NAME,
        "split_mode": split_mode,
        "total_images": int(df_all["image"].nunique()),
        "total_captions": int(len(df_all)),
        "train_images": int(train_df["image"].nunique()),
        "train_captions": int(len(train_df)),
        "val_images": int(val_df["image"].nunique()),
        "val_captions": int(len(val_df)),
        "test_images": int(test_df["image"].nunique()),
        "test_captions": int(len(test_df)),
        "missing_images": int(len(missing)),
        "train_raw_unique_words": int(len(train_freq)),
        "vocab_size_after_min_freq": len(vocab),
        "min_word_freq": Config.MIN_WORD_FREQ,
        "avg_caption_length": round(float(df_all["num_tokens"].mean()), 2),
        "median_caption_length": float(df_all["num_tokens"].median()),
        "max_caption_length_observed": int(df_all["num_tokens"].max()),
        "seed": Config.SEED,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "eda_summary.csv"), index=False)

    generate_eda_figures(df_all, train_df, val_df, test_df, train_freq, args.image_dir, args.fig_dir)
    print("=== HOÀN TẤT EDA & TIỀN XỬ LÝ ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
