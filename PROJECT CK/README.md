# Xây dựng Quy trình Tự động Sinh Chú thích Ảnh và Chuyển ngữ Anh – Việt Thích ứng Ngữ cảnh
### A Deep Learning Pipeline for Automated Image Captioning and Context-Aware English-to-Vietnamese Translation
**Đề tài Cuối kỳ Môn Học Sâu (Deep Learning) — Khoa Toán - Ứng dụng | Bộ môn Khoa học Dữ liệu — SGU**  
**Nhóm 05**: Nguyễn Trương Cao Sơn (Nhóm trưởng), Đặng Thanh Phương, Phạm Trung Hiếu

---

## 1. Tổng quan Đề tài

Hệ thống xây dựng quy trình nối tầng 2 giai đoạn (**Two-Stage Cascaded Multimodal Pipeline**) kết hợp giữa Thị giác máy tính (Computer Vision) và Xử lý ngôn ngữ tự nhiên (NLP):
1. **Giai đoạn 1 (Vision-to-Language Image Captioning trên PyTorch):** Tiếp nhận ảnh màu tự nhiên, trích xuất đặc trưng thị giác (CNN/ViT), kết hợp cơ chế chú ý (Spatial Attention) để sinh chuỗi chú thích tiếng Anh ($Y_{EN}$) chính xác trên tập dữ liệu chuẩn mực **Flickr30k**.
2. **Giai đoạn 2 (Context-Aware Neural Machine Translation):** Tiếp nhận câu mô tả tiếng Anh, đưa qua mô hình Dịch máy nơ-ron (NMT) tiền huấn luyện chuyên sâu cho cặp ngôn ngữ Anh – Việt kết hợp kỹ thuật hậu xử lý làm mịn ngữ cảnh (Contextual Smoothing) để tạo câu chú thích tiếng Việt ($Y_{VI}$) tự nhiên, mượt mà và phù hợp với đời sống.

```text
[Hình ảnh đầu vào]
        │
        ▼ (Giai đoạn 1: PyTorch Deep Learning)
┌────────────────────────────────────────────────────────┐
│  Visual Encoder (ResNet-50 / ViT-B/16)                │
│                         │ (Spatial Feature Maps / Patches)
│                         ▼                              │
│  Bahdanau Attention / Cross-Attention Mechanism        │
│                         │ (Context Vector)             │
│                         ▼                              │
│  Language Decoder (LSTM / Transformer Decoder)        │
└────────────────────────────────────────────────────────┘
        │
        ▼ [Caption Tiếng Anh gốc (Flickr30k Benchmark)]
        │   → Đánh giá: BLEU-1..4, METEOR, CIDEr
        │
        ▼ (Giai đoạn 2: Context-Aware NMT Module)
┌────────────────────────────────────────────────────────┐
│  Pre-trained NMT Model (VinAI-Translate / EnViT5)     │
│                         │                              │
│  Contextual Smoothing & Entity Alignment Post-process  │
└────────────────────────────────────────────────────────┘
        │
        ▼ [Caption Tiếng Việt tự nhiên, chuẩn ngữ cảnh]
        │   → Đánh giá: BLEU-vi, Human Qualitative Score (1-5)
        │
        ▼ (Ứng dụng thực tế)
[Giao diện Web Demo Streamlit: Ảnh + Attention Heatmap + Song ngữ Anh-Việt]
```

---

## 2. Câu hỏi nghiên cứu (Research Questions - RQs)

1. **RQ1 (Về cơ chế chú ý - Spatial Attention):** Khi giữ nguyên bộ mã hóa ResNet-50 và bộ giải mã LSTM, việc bổ sung cơ chế chú ý không gian Bahdanau có cải thiện chất lượng chú thích tiếng Anh so với mô hình không sử dụng Attention hay không? Mức cải thiện được đo bằng BLEU-1 đến BLEU-4, METEOR và CIDEr.
2. **RQ2 (Về cấu hình Encoder-Decoder & Transformer):** Trong cùng điều kiện dữ liệu và đánh giá, cấu hình ResNet-50 kết hợp LSTM có Attention và cấu hình ViT-B/16 kết hợp Transformer Decoder khác nhau như thế nào về chất lượng chú thích tiếng Anh và chi phí tính toán?
3. **RQ3 (Về chiến lược giải mã & Trade-off):** Với cùng một mô hình đã huấn luyện, độ rộng Beam Search $k \in \{1, 3, 5, 7\}$ ảnh hưởng như thế nào đến chất lượng câu được sinh ra (BLEU, CIDEr) và thời gian suy luận (latency) trên mỗi ảnh?

---

## 3. Cấu trúc thư mục dự án (Project Directory)

```text
PROJECT CK/
├── data/
│   ├── raw/                  # Ảnh gốc và file annotations Flickr30k
│   └── processed/            # File vocab.pkl, annotations train/val/test splits
├── notebooks/                # Jupyter Notebooks EDA và chạy thử nghiệm nhanh
│   ├── 01_eda_and_vocab.ipynb
│   └── 02_visualize_attention.ipynb
├── src/                      # Source code chính viết bằng PyTorch
│   ├── __init__.py
│   ├── config.py             # Toàn bộ cấu hình siêu tham số và đường dẫn
│   ├── dataset.py            # Custom Dataset, DataLoader và Collate Function
│   ├── models/               # Các kiến trúc Deep Learning
│   │   ├── __init__.py
│   │   ├── baseline_cnn_lstm.py   # Baseline: ResNet-50 + LSTM (Show and Tell)
│   │   ├── attention_model.py     # ResNet-50 + Bahdanau Attention + LSTM (Show, Attend & Tell)
│   │   └── vit_transformer.py     # SOTA: ViT-B/16 + Transformer Decoder
│   ├── translation.py        # Module chuyển ngữ Anh - Việt thích ứng ngữ cảnh
│   ├── utils.py              # Đánh giá (BLEU, CIDEr), checkpoint, early stopping
│   ├── train.py              # Script huấn luyện chính
│   └── evaluate.py           # Script đánh giá trên tập Test
├── experiments/              # Lưu model checkpoints (.pt) và loss logs
├── app/                      # Web Demo tương tác
│   └── app.py                # Giao diện tương tác Streamlit (Song ngữ & Attention Map)
├── report/                   # Tài liệu báo cáo
│   ├── Nhom05_ImageCaptioning_Proposal.docx
│   └── Nhom05_ImageCaptioning_Proposal.pdf
├── requirements.txt          # Các thư viện cần thiết
└── README.md
```

---

## 4. Kế hoạch Ablation Studies

- **Ablation 1 (Trả lời RQ1)**: No-Attention (Baseline) vs. Bahdanau Spatial Attention vs. Scaled Dot-Product Attention.
- **Ablation 2 (Trả lời RQ2)**: CNN Backbone (ResNet-50) vs. Vision Transformer (ViT-B/16).
- **Ablation 3 (Trả lời RQ3)**: Greedy Search ($k=1$) vs. Beam Search ($k \in \{3, 5, 7\}$).
- **Ablation 4 (Bổ trợ)**: Freeze toàn bộ Encoder vs. Fine-tune các tầng tích chập cuối cùng.

---

## 5. Tài liệu Proposal chính thức
Bản thuyết minh đề tài (Proposal) chuẩn 5 trang đã được đồng bộ tại:
- [Nhom05_ImageCaptioning_Proposal.docx](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_ImageCaptioning_Proposal.docx)
- [Nhom05_ImageCaptioning_Proposal.pdf](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_ImageCaptioning_Proposal.pdf)
