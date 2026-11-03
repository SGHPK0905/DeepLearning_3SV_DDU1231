# Image Captioning with Deep Learning (PyTorch)
### Đề tài Cuối kỳ Môn Học Sâu (Deep Learning) — Nhóm 05

---

## 1. Giới thiệu đề tài
Hệ thống tự động sinh chú thích ảnh (**Image Captioning**) đa phương thức (Multimodal) kết hợp Thị giác máy tính (Computer Vision) và Xử lý ngôn ngữ tự nhiên (Natural Language Processing).

- **Mục tiêu**: Xây dựng pipeline hoàn chỉnh trên **PyTorch** từ tiền xử lý dữ liệu, trích xuất đặc trưng hình ảnh, mô hình sinh chuỗi tuần tự với cơ chế **Attention**, so sánh đối chứng với kiến trúc **Transformer (ViT + Decoder)** và triển khai ứng dụng Web Demo bằng **Streamlit/Gradio**.
- **Bộ dữ liệu chính**: **Flickr8k** (8,092 ảnh, 40,460 captions tiếng Anh).

---

## 2. Câu hỏi nghiên cứu (Research Questions - RQs)
1. **RQ1 (Cơ chế Attention)**: Cơ chế Bahdanau Spatial Attention cải thiện chất lượng câu mô tả như thế nào so với mô hình Encoder-Decoder thuần túy (ResNet-50 + LSTM không có Attention) khi đo lường bằng BLEU-1..4, METEOR, ROUGE-L và CIDEr?
2. **RQ2 (Visual Backbone & Transformer)**: Việc sử dụng Vision Transformer (ViT-Base) thay thế cho mạng CNN (ResNet-50) có giúp nắm bắt ngữ cảnh toàn cục tốt hơn và giảm hiện tượng sinh từ sai lệch (hallucination) không?
3. **RQ3 (Chiến lược giải mã & Trade-off)**: Chiến lược giải mã Beam Search ($k \in \{1, 3, 5, 7\}$) tác động như thế nào đến sự đánh đổi giữa chất lượng câu sinh ra (điểm CIDEr/BLEU) và tốc độ suy luận (độ trễ thời gian thực)?

---

## 3. Cấu trúc thư mục dự án

```text
PROJECT CK/
├── data/
│   ├── raw/                  # Ảnh gốc và file captions (Images + captions.txt)
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
│   │   ├── baseline_cnn_lstm.py   # Baseline: ResNet-50 + LSTM
│   │   ├── attention_model.py     # ResNet-50 + Bahdanau Attention + LSTM
│   │   └── vit_transformer.py     # SOTA: ViT + Transformer Decoder
│   ├── utils.py              # Đánh giá (BLEU, CIDEr), checkpoint, early stopping
│   ├── train.py              # Script huấn luyện chính
│   └── evaluate.py           # Script đánh giá trên tập Test
├── experiments/              # Lưu model checkpoints (.pt) và loss logs
├── app/                      # Web Demo tương tác
│   └── app.py                # Giao diện tương tác Streamlit
├── report/                   # Tài liệu báo cáo
│   ├── Nhom05_ImageCaptioning_Proposal.docx
│   └── Nhom05_ImageCaptioning_Proposal.pdf
├── requirements.txt          # Các thư viện cần thiết
└── README.md
```

---

## 4. Kế hoạch Ablation Studies
- **Ablation 1 (Trả lời RQ1)**: No-Attention (Baseline) vs. Bahdanau Spatial Attention vs. Scaled Dot-Product Attention.
- **Ablation 2 (Trả lời RQ2)**: CNN Backbone (ResNet-50) vs. Vision Transformer (ViT-Base/16).
- **Ablation 3 (Trả lời RQ3)**: Greedy Search ($k=1$) vs. Beam Search ($k \in \{3, 5, 7\}$).
- **Ablation 4 (Bổ trợ)**: Freeze toàn bộ Encoder vs. Fine-tune các tầng tích chập cuối cùng.

---

## 5. Hướng dẫn nộp Proposal (Deadline: 12h đêm nay)
File Proposal hoàn chỉnh (5 trang, chuẩn quy định) đã được tạo sẵn tại:
- [Nhom05_ImageCaptioning_Proposal.docx](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_ImageCaptioning_Proposal.docx)
- [Nhom05_ImageCaptioning_Proposal.pdf](file:///d:/Data-Science---SGU/Nam%204/DeepLearning/Team/PROJECT%20CK/report/Nhom05_ImageCaptioning_Proposal.pdf)

> **Lưu ý trước khi nộp lên Google Classroom**:
> 1. Mở file Word hoặc PDF lên kiểm tra lại: số thứ tự nhóm (ví dụ đổi `Nhom05` thành `Nhom02`, `Nhom08`... theo số nhóm chính thức của lớp nếu có).
> 2. Kiểm tra lại MSSV của các thành viên trong nhóm.
> 3. Xuất lại PDF và nộp file PDF đúng theo cú pháp: `Nhom??_TenDeTai_Proposal.pdf`.

