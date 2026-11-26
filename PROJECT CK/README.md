# Xây dựng Quy trình Tự động Sinh Chú thích Ảnh và Chuyển ngữ Anh – Việt Thích ứng Ngữ cảnh
### A Deep Learning Pipeline for Automated Image Captioning and Context-Aware English-to-Vietnamese Translation

> **Học phần:** Học sâu (Deep Learning) — **Mã HP:** 858048 — **Lớp:** DDU1231  
> **Trường:** Đại học Sài Gòn (SGU) — Khoa Toán - Ứng dụng | Bộ môn Khoa học Dữ liệu  
> **Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
> **Repository chính thức:** [https://github.com/SGHPK0905/DeepLearning_3SV_DDU1231](https://github.com/SGHPK0905/DeepLearning_3SV_DDU1231)  
> **Hạn nộp chốt cứng:** `14/12/2026`

---

## 1. Thông tin Nhóm & Tỷ lệ đóng góp (Tổng = 100%)

| STT | Họ và tên | MSSV | Vai trò | Nhiệm vụ chính phụ trách | Đóng góp (%) |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **Nguyễn Trương Cao Sơn** | `3123580040` | **Nhóm trưởng** | Thiết kế kiến trúc pipeline, config & `train.py`; trích xuất đặc trưng; triển khai mô hình mở rộng `ViT-B/16 + Transformer Decoder`; tổng hợp Ablation & Báo cáo. | **34%** |
| 2 | **Đặng Thanh Phương** | `3123580036` | Thành viên | Tải, EDA & tiền xử lý **Flickr8k** (`dataset.py`, `vocab.pkl`); kiểm soát chống data leakage; triển khai & huấn luyện Baseline `CNN-LSTM` và `Attention-LSTM`. | **33%** |
| 3 | **Phạm Trung Hiếu** | `3123580014` | Thành viên | Xây dựng bộ đánh giá multi-ref (`BLEU`, `METEOR`, `ROUGE-L`, `CIDEr`), Greedy/Beam Search; tích hợp `VietAI/envit5-translation` & chấm 100 ảnh; `app.py` (Streamlit) & Slide/Poster. | **33%** |
| | **Tổng cộng** | | | | **100%** |

---

## 2. Điều chỉnh Phạm vi Dữ liệu so với Proposal ban đầu
- **Tập dữ liệu triển khai chính thức:** **Flickr8k** trích từ [`jxie/flickr8k`](https://huggingface.co/datasets/jxie/flickr8k) (đã đóng gói sẵn theo **Karpathy Split**: `6.000 train / 1.000 val / 1.000 test` = `8.000` ảnh, `40.000` captions gốc, còn `39.981` câu hợp lệ sau khử trùng lặp; từ điển `2.538` tokens fit riêng trên Train với `min_freq = 5`; `0` ảnh trùng giữa 3 tập). *(So với bản Kaggle `adityajn105/flickr8k` có 8.091 ảnh và `captions.txt`, bản `jxie/flickr8k` dùng chuẩn 8.000 ảnh Karpathy và lưu thành `captions.csv` kèm cột `split`).*
- **Lý do điều chỉnh từ Flickr30k (Proposal) sang Flickr8k:** Nhằm đảm bảo tính khả thi về tài nguyên tính toán (RTX 3050 4GB VRAM / Google Colab T4 giới hạn phiên) và thời hạn hoàn thành (`14/12/2026`), giúp nhóm chạy đầy đủ và công bằng toàn bộ **Baseline + Attention + ViT + 4 Ablation Studies + Multi-seed (`mean ± std`)** thay vì chạy thiếu kiểm chứng trên tập lớn.
- **Hướng mở rộng tương lai:** Mở rộng quy mô huấn luyện sang **Flickr30k** (31.783 ảnh) và MS-COCO sẽ được phân tích trong **Chương 5 (Hướng phát triển tương lai)**.

---

## 3. Quy trình 2 Giai đoạn & Câu hỏi Nghiên cứu (RQs)

```text
[Ảnh đầu vào (224x224x3)]
        │
        ▼ (Giai đoạn 1: PyTorch Deep Learning trên Flickr8k)
┌────────────────────────────────────────────────────────┐
│  Visual Encoder (ResNet-50 / ViT-B/16)                 │
│        ▼                                               │
│  Bahdanau Spatial Attention / Cross-Attention          │
│        ▼                                               │
│  Language Decoder (LSTM / Transformer Decoder)         │
│  + Decoding (Greedy k=1 vs Beam Search k=3, 5, 7)      │
└────────────────────────────────────────────────────────┘
        │
        ▼ [Caption Tiếng Anh (Y_EN)] → BLEU-1..4, METEOR, ROUGE-L, CIDEr
        │
        ▼ (Giai đoạn 2: Chuyển ngữ Anh – Việt thích ứng ngữ cảnh)
┌────────────────────────────────────────────────────────┐
│  Pre-trained NMT (VietAI/envit5-translation)           │
│  + Đánh giá thủ công 100 ảnh test (Rubric 1–5)         │
└────────────────────────────────────────────────────────┘
        │
        ▼ [Caption Tiếng Việt (Y_VI)] → Web Demo Streamlit + Attention Heatmap
```

1. **RQ1 (Cơ chế chú ý):** Khi giữ nguyên ResNet-50 và LSTM, bổ sung Bahdanau Spatial Attention có cải thiện BLEU-1..4, METEOR, ROUGE-L và CIDEr so với mô hình không Attention không?
2. **RQ2 (Cấu hình Encoder–Decoder):** Trong cùng điều kiện dữ liệu và đánh giá trên Flickr8k, cấu hình `ResNet-50 + Bahdanau Attention + LSTM` và `ViT-B/16 + Transformer Decoder` khác nhau thế nào về chất lượng caption và chi phí tính toán?
3. **RQ3 (Chiến lược giải mã):** Trên cùng checkpoint đã huấn luyện, độ rộng Beam Search $k \in \{1, 3, 5, 7\}$ tác động ra sao đến BLEU/CIDEr và độ trễ suy luận (ms/ảnh)?

---

## 4. Cấu trúc Thư mục Dự án

```text
PROJECT CK/
├── data/
│   ├── raw/                  # Ảnh Flickr8k & captions (bị ignore trên Git)
│   └── processed/            # vocab.pkl, karpathy_splits.json, cached features
├── notebooks/
│   ├── 01_eda_and_vocab.ipynb
│   └── 02_error_analysis_and_attention.ipynb
├── references/               # 9 bài báo khoa học gốc (PDF), README & BibTeX
├── src/
│   ├── __init__.py
│   ├── config.py             # Seed, paths, hyperparams (min_freq=5, Flickr8k)
│   ├── download_data.py      # Tải jxie/flickr8k (HF) hoặc Kaggle zip
│   ├── eda_preprocess.py     # EDA 300 DPI (WordCloud, Zipf), split chống leakage & vocab.pkl
│   ├── extract_features.py   # Cache ResNet-50 global & spatial features
│   ├── dataset.py            # Flickr8kDataset, CachedFeatureDataset, collate_fn
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn_lstm.py   # Model 1: Show and Tell (ResNet-50 + LSTM)
│   │   ├── attention_model.py     # Model 2: Show, Attend and Tell (Bahdanau Attention)
│   │   └── vit_transformer.py     # Model 3: ViT-B/16 + Transformer Decoder
│   ├── translation.py        # Module chuyển ngữ VietAI/envit5-translation
│   ├── metrics.py            # Multi-ref BLEU-1..4, METEOR, ROUGE-L, CIDEr
│   ├── inference.py          # Greedy Search & Beam Search (k=1,3,5,7)
│   ├── train.py              # Script huấn luyện & lưu logs/checkpoints
│   └── evaluate.py           # Đánh giá trên tập Test & xuất báo cáo lỗi
├── checkpoints/              # Trọng số .pt (link tải Google Drive nếu >100MB)
├── figures/                  # Biểu đồ 300 DPI, Attention heatmaps & sơ đồ diagrams/
├── logs/                     # CSV history huấn luyện & ablation_results.csv
├── app/
│   └── app.py                # Web Demo Streamlit (Ảnh + Heatmap + Anh/Việt)
├── report/
│   ├── Nhom05_ImageCaptioning_Proposal.docx / .pdf
│   └── Nhom05_BaoCao_CuoiKy.docx
├── NhatKy_ThucNghiem_CK.xlsx # Sổ tay theo dõi tiến độ, thực nghiệm & nguồn hình ảnh
├── requirements.txt
└── README.md
```

---

## 5. Tài liệu Theo dõi & Báo cáo
- **Sổ tay Excel theo dõi hàng ngày & nguồn hình/pipeline:** [NhatKy_ThucNghiem_CK.xlsx](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/NhatKy_ThucNghiem_CK.xlsx)
- **Khung Báo cáo Cuối kỳ (5 Chương + TOC):** [Nhom05_BaoCao_CuoiKy.docx](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_BaoCao_CuoiKy.docx)
- **Proposal (đã cập nhật GitHub & phạm vi Flickr8k):** [Nhom05_ImageCaptioning_Proposal.pdf](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_ImageCaptioning_Proposal.pdf)
