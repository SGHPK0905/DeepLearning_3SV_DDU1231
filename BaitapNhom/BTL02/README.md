# BÀI TẬP LỚN 02: NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG NƠ-RON TÍCH CHẬP (CNN) TRONG PYTORCH

> **Môn học:** Học sâu (Deep Learning)  
> **Mã học phần:** 858048 — **Lớp:** DDU1231  
> **Trường:** Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin  
> **Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
> **Nhóm thực hiện:** Nhóm 05  
> **Thời gian thực hiện:** Tháng 11/2026  

---

## 👥 THÀNH VIÊN NHÓM VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | MSSV | Vai trò | Nhiệm vụ chính phụ trách | Đóng góp |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **Nguyễn Trương Cao Sơn** | `3123580040` | **Nhóm trưởng** | **Giai đoạn 3, 6 & 7:** Chủ trì thiết kế kiến trúc CNN VGG16_FER & ResNet_FER; điều phối 4 chuỗi thực nghiệm A/B/C/D; xây dựng CLI `main.py`, lập trình module trích xuất bộ lọc Conv tầng cuối (Slide 5) & Feature Maps; chủ trì biên soạn Báo cáo học thuật (.docx/.pdf) và Slide thuyết trình (.pptx/.pdf); đóng gói sản phẩm nộp bài. | **100%** |
| 2 | **Đặng Thanh Phương** | `3123580036` | Thành viên | **Giai đoạn 2 & 4:** Thiết kế kiến trúc `SmallCNN` và `MediumCNN` (theo gợi ý của Thầy); xây dựng bộ khung huấn luyện PyTorch chuẩn (`src/trainer.py`), cơ chế Early Stopping, Learning Rate Scheduler; phụ trách thực nghiệm B (Optimizers) và thực nghiệm D (Regularization & BatchNorm Ablation). | **100%** |
| 3 | **Phạm Trung Hiếu** | `3123580014` | Thành viên | **Giai đoạn 1 & 5:** Xây dựng pipeline dữ liệu FER-2013 (`src/data.py`), trích xuất và cache nhanh `.npz`, lập trình bộ Data Augmentation đa dạng (Flip, Rotate +-10°, Shift +-4px, Cutout); tính toán bộ chỉ số đánh giá (`src/metrics.py`); phụ trách thực nghiệm C (Data Augmentation) và biên soạn Nhật ký thực nghiệm Excel. | **100%** |

---

## 📌 1. BÀI TOÁN VÀ CƠ SỞ LÝ THUYẾT

### 1.1. Phát biểu bài toán
Bài toán Nhận diện cảm xúc khuôn mặt (**Facial Emotion Recognition - FER**) trên bộ dữ liệu **Kaggle FER-2013** yêu cầu ánh xạ ảnh khuôn mặt mức xám kích thước $48 \times 48$ pixel vào một trong 7 biểu cảm chuẩn:
- `0`: **Angry** (Tức giận)
- `1`: **Disgust** (Ghê tởm)
- `2`: **Fear** (Sợ hãi)
- `3`: **Happy** (Vui vẻ)
- `4`: **Sad** (Buồn bã)
- `5`: **Surprise** (Ngạc nhiên)
- `6`: **Neutral** (Trung tính)

### 1.2. Cơ sở toán học của Mạng nơ-ron Tích chập (CNN)
Khác với MLP làm mất cấu trúc không gian, mạng CNN bảo toàn hình học 2D thông qua:
1. **Phép tích chập 2D (2D Convolution):**
   $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) K(m, n)$$
   Nhờ cơ chế chia sẻ trọng số (*Weight Sharing*) và trường tiếp nhận cục bộ (*Local Receptive Field*), mạng trích xuất hiệu quả các đặc trưng biên cạnh, hình khối và vi biểu cảm.
2. **Tầng Chuẩn hóa theo lô (Batch Normalization):**
   $$\hat{x}^{(k)} = \frac{x^{(k)} - \mathrm{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}] + \epsilon}}, \quad y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
   Giúp ổn định phân phối kích hoạt qua các tầng sâu, chống bão hòa hàm phi tuyến.
3. **Tầng Gộp cực đại (MaxPooling 2x2):**
   Giảm một nửa chiều không gian ($H/2, W/2$), tăng tính bất biến tịnh tiến và mở rộng trường nhìn cho các tầng tiếp theo.
4. **Hàm mất mát Cross-Entropy đa lớp:**
   $$\mathcal{L}_{CE} = -\sum_{c=0}^{6} y_c \log(\hat{y}_c) = -\sum_{c=0}^{6} y_c \log\left(\frac{e^{z_c}}{\sum_{j=0}^{6} e^{z_j}}\right)$$

---

## 📊 2. BỘ DỮ LIỆU FER-2013 & TIỀN XỬ LÝ

- **Quy mô:** Tổng cộng 35,887 ảnh mức xám $48 \times 48$.
  - `Training`: 28,709 ảnh.
  - `PublicTest` (Validation): 3,589 ảnh.
  - `PrivateTest` (Test): 3,589 ảnh.
- **Thách thức:** Mất cân bằng lớp nghiêm trọng (`Happy`: 8,989 ảnh vs `Disgust`: chỉ 547 ảnh — chênh lệch hơn 16 lần).
- **Data Augmentation:** Lật ngang ngẫu nhiên ($p=0.5$), Xoay ngẫu nhiên $[ -10^\circ, +10^\circ ]$, Dịch chuyển ngẫu nhiên $[ -4\text{px}, +4\text{px} ]$ và Cutout/Random Erasing $8 \times 8$.

---

## 🏗️ 3. KIẾN TRÚC CÁC MÔ HÌNH KHẢO SÁT

1. **SmallCNN (2 Blocks - 4.8M tham số):**
   - 2 khối Conv(32)-Conv(64)-MaxPool-Dropout và Conv(128)-Conv(128)-MaxPool-Dropout.
   - Phân loại: Flatten $\to$ Linear(512) $\to$ BN $\to$ ReLU $\to$ Dropout(0.5) $\to$ Linear(7).
2. **MediumCNN (3 Blocks - 13.7M tham số - Đề xuất tối ưu của Thầy):**
   - Block 1 (64 kênh, MaxPool $\to 24 \times 24$).
   - Block 2 (128 kênh, MaxPool $\to 12 \times 12$).
   - Block 3 (256 kênh, MaxPool $\to 6 \times 6$).
   - Phân loại: Flatten $\to$ Linear(1024) $\to$ Linear(256) $\to$ Linear(7).
3. **VGG16_FER (4 Blocks - 18.5M tham số):**
   - 4 giai đoạn tích chập sâu (64 $\to$ 128 $\to$ 256 $\to$ 512), 4 tầng pooling đưa về $3 \times 3$ pixel.
4. **ResNet_FER (11.2M tham số):**
   - 4 khối Residual Blocks với Skip Connection $y = \mathcal{F}(x) + x$ và Global Average Pooling (GAP).

---

## 🔬 4. TỔNG HỢP KẾT QUẢ THỰC NGHIỆM KIỂM SOÁT

### Thực nghiệm A: So sánh 4 Họ Kiến trúc CNN
| Kiến trúc | Số tham số | Val Loss | Val Acc (%) | Test Acc (%) | Thời gian/epoch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SmallCNN** | 4,821,575 | 1.1824 | 55.82% | 55.45% | 8.5 s |
| **MediumCNN** | 13,742,471 | 1.0542 | 61.35% | 60.88% | 14.2 s |
| **VGG16_FER** | 18,452,103 | 1.0215 | 62.80% | **62.45%** | 21.6 s |
| **ResNet_FER** | 11,173,959 | 1.0850 | 59.90% | 59.35% | 17.8 s |

### Thực nghiệm B: So sánh Thuật toán Tối ưu
| Thuật toán | Learning Rate | Weight Decay | Best Val Loss | Val Acc (%) | Test Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD với Momentum** | 0.010 | $1\times 10^{-4}$ | 1.1420 | 57.25% | 56.80% |
| **Adam** | 0.001 | $1\times 10^{-4}$ | 1.0542 | 61.35% | 60.88% |
| **AdamW** | 0.001 | $1\times 10^{-3}$ | **1.0318** | **62.15%** | **61.70%** |

### Thực nghiệm C: Khảo sát Data Augmentation & Generalization Gap
| Chế độ Augmentation | Train Acc (%) | Val Acc (%) | Overfitting Gap | Test Acc (%) |
| :--- | :---: | :---: | :---: | :---: |
| **None (Chỉ chuẩn hóa)** | 74.20% | 56.40% | **+17.80% (Quá khớp)** | 55.80% |
| **Basic (Lật ngang)** | 69.50% | 61.35% | +8.15% | 60.88% |
| **Advanced (Đầy đủ)** | 68.20% | 63.20% | **+5.00% (Tối ưu)** | **62.90%** |

### Thực nghiệm D: Nghiên cứu Triệt tiêu (Ablation Study)
- **Dropout=0.10:** 58.40% Test Acc (Chưa triệt tiêu đồng thích nghi nơ-ron).
- **Dropout=0.25 (Chuẩn):** 60.88% Test Acc (Cân bằng tối ưu).
- **Dropout=0.40:** 57.30% Test Acc (Thiếu khớp Underfitting nhẹ).
- **Không dùng BatchNorm (Ablation):** Sụt giảm xuống **51.40%** (-9.48%), chứng minh BatchNorm là thành phần sống còn để ổn định phân phối kích hoạt.

### 🏆 Mô hình Quán quân (Champion Model)
- **Cấu hình:** MediumCNN + AdamW ($lr=10^{-3}$) + Advanced Augmentation + ReduceLROnPlateau.
- **Kết quả PrivateTest (3,589 ảnh):** **65.85% Accuracy**, **63.70% Macro F1**.
- Sát ngưỡng kỷ lục thế giới trên Kaggle FER-2013 (69% public, 71% private).

---

## 🖼️ 5. MINH CHỨNG HÌNH VẼ & BỘ LỌC CONVOLUTION (THEO YÊU CẦU SLIDE 5)

| Tên hình vẽ | Mô tả chi tiết |
| :--- | :--- |
| `figures/dataset_distribution_histogram.png` | Phân bố 7 lớp cảm xúc trên 3 phân vùng Train/Val/Test. |
| `figures/trainining_images.png` | Lưới 32 ảnh mẫu huấn luyện (4x8) kèm nhãn cảm xúc. |
| `figures/champion_learning_curves.png` | Đường cong Train/Val Loss và Train/Val Accuracy qua các epoch. |
| `figures/champion_confusion_matrix.png` | Ma trận nhầm lẫn chuẩn hóa 7x7 phân tích chi tiết từng cảm xúc. |
| `figures/champion_true_cases.png` | Các trường hợp dự đoán đúng kèm độ tự tin Softmax cao. |
| `figures/champion_wrong_cases.png` | Phân tích các trường hợp đoán sai và nguyên nhân mơ hồ nhãn. |
| `figures/champion_conv_filters.png` | **Trực quan hóa 64 bộ lọc tích chập tầng cuối (Yêu cầu bắt buộc Slide 5).** |
| `figures/champion_feature_maps.png` | Bản đồ kích hoạt đặc trưng phân tầng qua Block 1, Block 2, Block 3. |

---

## 💡 6. PHÂN TÍCH CA THẤT BẠI: VÌ SAO FINE-TUNE VGG16 BỊ SUY THOÁI?
*(Trả lời câu hỏi trong mục 5.1 demo_report.pdf của Thầy TS. Đỗ Như Tài)*

1. **Tiêu biến kích thước không gian cực hạn (Extreme Spatial Downsampling):**  
   Mạng VGG-16 gốc thiết kế cho ảnh ImageNet $224 \times 224$ với 5 tầng MaxPooling. Khi đưa ảnh FER-2013 ($48 \times 48$) qua 5 tầng:
   $$48 \to 24 \to 12 \to 6 \to 3 \to \mathbf{1 \times 1 \text{ pixel!}}$$
   Tại tầng pooling cuối cùng, bản đồ chỉ còn $1 \times 1$ pixel, làm phẳng và mất hoàn toàn cấu trúc hình học tương quan giữa mắt, mũi, miệng.
2. **Domain Mismatch:**  
   ImageNet học trên 1.2 triệu ảnh màu RGB tự nhiên, trong khi FER-2013 là ảnh mức xám $48 \times 48$ độ phân giải thấp. Việc nhân bản 3 kênh xám không mang lại thêm thông tin mà còn lãng phí tính toán.
3. **Bài học kiến trúc:**  
   Với ảnh $48 \times 48$, mô hình thiết kế riêng với 3 tầng pooling (như `MediumCNN`) giữ lại kích thước $6 \times 6$ trước classifier cho hiệu năng vượt trội và tính ổn định cao nhất.

---

## 📂 7. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BTL02/
├── checkpoints/
│   └── best_model.pt                      # Điểm kiểm tra trọng số mô hình tốt nhất
├── data/
│   ├── fer2013.csv                        # Bộ dữ liệu gốc (301 MB)
│   └── preprocessing/
│       ├── train.npz                      # 28,709 ảnh huấn luyện
│       ├── valid.npz                      # 3,589 ảnh kiểm định
│       └── test.npz                       # 3,589 ảnh kiểm thử
├── experiments/
│   ├── exp_A_architectures.py             # Thực nghiệm A: So sánh 4 kiến trúc CNN
│   ├── exp_B_optimizers.py                # Thực nghiệm B: So sánh SGD, Adam, AdamW
│   ├── exp_C_augmentations.py             # Thực nghiệm C: Khảo sát Data Augmentation
│   └── exp_D_regularization.py            # Thực nghiệm D: Khảo sát Regularization & BatchNorm
├── figures/                               # Toàn bộ biểu đồ, hình mẫu, bộ lọc và confusion matrix
├── logs/                                  # Toàn bộ file CSV nhật ký thực nghiệm và metrics
├── report/
│   ├── BTL02_BaoCao.docx                  # Báo cáo học thuật định dạng Word (8 chương chuẩn)
│   └── BTL02_BaoCao.pdf                   # Báo cáo học thuật xuất bản PDF chính thức
├── Slide/
│   ├── BTL02_ThuyetTrinh.pptx             # Slide thuyết trình PowerPoint 16 slide SGU
│   └── Slide.pdf                          # Slide thuyết trình xuất bản PDF
├── src/
│   ├── config.py                          # Cấu hình tập trung, seed, đường dẫn, tham số
│   ├── data.py                            # Module tiền xử lý, cache npz, Augmentation
│   ├── export_pdf.py                      # Module xuất PDF tự động qua Word/PowerPoint COM
│   ├── generate_excel_log.py              # Tạo workbook phân công & nhật ký thực nghiệm Excel
│   ├── generate_slide_deck.py             # Xây dựng slide thuyết trình PowerPoint
│   ├── generate_word_report.py            # Xây dựng báo cáo học thuật Word
│   ├── metrics.py                         # Tính toán Accuracy, F1, Confusion Matrix
│   ├── models.py                          # 4 kiến trúc CNN (Small, Medium, VGG16, ResNet)
│   ├── train_champion.py                  # Huấn luyện mô hình quán quân & trực quan hóa
│   ├── trainer.py                         # Vòng lặp huấn luyện PyTorch chuẩn
│   └── visualize.py                       # Trực quan hóa hình ảnh, bộ lọc conv, feature maps
├── main.py                                # CLI điều phối toàn bộ hệ thống
├── Phan_cong_nhiem_vu_va_Nhat_ky_BTL02.xlsx # File Excel phân công & nhật ký thực nghiệm
├── requirements.txt                       # Danh sách thư viện phụ thuộc
└── README.md                              # Hướng dẫn chi tiết dự án
```

---

## 🚀 8. HƯỚNG DẪN CÀI ĐẶT & CHẠY LẠI THỰC NGHIỆM

### Cài đặt môi trường:
```bash
pip install -r requirements.txt
```

### Chạy toàn bộ pipeline tự động từ đầu đến cuối:
```bash
python main.py --all
```

### Hoặc chạy từng phân đoạn độc lập:
```bash
# 1. Tiền xử lý dữ liệu và tạo biểu đồ phân bố:
python main.py --prepare_data

# 2. Chạy 4 chuỗi thực nghiệm A, B, C, D:
python main.py --run_experiments

# 3. Huấn luyện Mô hình Quán quân và trích xuất bộ lọc Conv:
python main.py --train_champion

# 4. Xuất Báo cáo Word/PDF, Slide PPTX/PDF và Excel:
python main.py --generate_reports

# 5. Đóng gói sản phẩm nộp bài theo quy cách:
python main.py --pack_zip
```

---
*© 2026 — Nhóm 05 — Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn (SGU)*
