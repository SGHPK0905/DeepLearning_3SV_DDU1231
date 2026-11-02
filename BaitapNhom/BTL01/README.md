# BÀI TẬP LỚN 01: XẤP XỈ HÀM MẬT ĐỘ GAUSSIAN 2D BẰNG MẠNG NƠ-RON MLP (PYTORCH)

> **Môn học:** Học sâu (Deep Learning)  
> **Trường:** Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin  
> **Lớp:** DDU1231  
> **Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
> **Thời gian thực hiện:** Tháng 10/2026  

---

## 👥 THÀNH VIÊN NHÓM VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | MSSV | Vai trò | Nhiệm vụ chính phụ trách | Đóng góp |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **Đặng Thanh Phương** | `3123580036` | Thành viên | **Giai đoạn 2:** Thiết kế kiến trúc `MLPRegressor` (`nn.Module`), triển khai vòng lặp huấn luyện chuẩn PyTorch (`trainer.py`), cơ chế checkpointing và bộ chỉ số đánh giá (`metrics.py`). | **100%** |
| 2 | **Phạm Trung Hiếu** | `3123410123` | Thành viên | **Giai đoạn 1:** Xây dựng module sinh dữ liệu Gaussian 2D giải tích (`data.py`), phân chia độc lập Train/Val/Test chống rò rỉ, chuẩn hóa `Scaler` (`preprocessing.py`) và trực quan hóa 3D (`visualize.py`). | **100%** |
| 3 | **Nguyễn Trương Cao Sơn** | `3123580040` | **Nhóm trưởng** | **Giai đoạn 3 & 4:** Thiết kế và thực thi 3 thực nghiệm kiểm soát A/B/C (`experiments/`), tổng hợp logs, xây dựng CLI điều phối (`main.py`), biên soạn Báo cáo học thuật (.docx/.pdf) và Slide thuyết trình (.pptx/.pdf). | **100%** |

---

## 📌 1. BÀI TOÁN VÀ CƠ SỞ LÝ THUYẾT

### 1.1. Phát biểu toán học
Mục tiêu là xây dựng và huấn luyện mạng Perceptron đa tầng (**Multi-Layer Perceptron - MLP**) nhằm xấp xỉ hàm mật độ xác suất Gaussian hai chiều:

$$f(\mathbf{x}) = \frac{1}{2\pi \sqrt{\det(C)}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T C^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Trong đó:
* Vector kỳ vọng: $\boldsymbol{\mu} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
* Ma trận hiệp phương sai: $C = \begin{bmatrix} 0.3 & -0.5 \\ -0.5 & 2.0 \end{bmatrix}$
* Định thức: $\det(C) = 0.3 \times 2.0 - (-0.5)^2 = 0.6 - 0.25 = 0.35 > 0$ (Ma trận đối xứng, xác định dương).
* Ma trận nghịch đảo: $C^{-1} = \frac{1}{0.35} \begin{bmatrix} 2.0 & 0.5 \\ 0.5 & 0.3 \end{bmatrix} \approx \begin{bmatrix} 5.7143 & 1.4286 \\ 1.4286 & 0.8571 \end{bmatrix}$
* Hệ số chuẩn hóa: $\frac{1}{2\pi\sqrt{0.35}} \approx 0.2690$ (Độ cao cực đại của đỉnh phân phối tại tâm $(0, 0)$).
* Miền khảo sát không gian compact: $x_1 \in [-2, 2], \quad x_2 \in [-4, 4]$.

### 1.2. Cơ sở lý thuyết học máy
Theo **Định lý Xấp xỉ Phổ quát (Universal Approximation Theorem)** của Cybenko (1989) và Hornik (1991), một mạng nơ-ron truyền thẳng chỉ cần 1 tầng ẩn với số nơ-ron đủ lớn và hàm kích hoạt phi tuyến liên tục có thể xấp xỉ bất kỳ hàm liên tục nào trên tập compact với sai số nhỏ tùy ý. Đồ án kiểm chứng định lý này thông qua 3 chiều thực nghiệm kiểm soát: **Quy mô dữ liệu (A)**, **Độ rộng mạng (B)**, và **Độ sâu mạng (C)**.

---

## 📁 2. CẤU TRÚC THƯ MỤC DỰ ÁN

Mã nguồn được tổ chức module hóa, phân tách rành mạch các chức năng theo đúng tiêu chuẩn công nghiệp:

```text
BTL01/
├── checkpoints/                    # Lưu trữ trọng số mô hình tốt nhất (.pt)
│   ├── .gitkeep
│   └── best_model.pt               # Checkpoint mô hình Champion (C=3, [32, 32, 32])
├── data/                           # Dữ liệu thực nghiệm phân chia độc lập
│   ├── .gitkeep
│   ├── train_full.csv              # Tập huấn luyện tối đa (2000 mẫu, 80%)
│   ├── val.csv                     # Tập kiểm định chọn checkpoint (400 mẫu, 16%)
│   └── test.csv                    # Tập kiểm tra độc lập cuối cùng (100 mẫu, 4%)
├── experiments/                    # Mã nguồn 3 nhóm thực nghiệm độc lập
│   ├── .gitkeep
│   ├── exp_A_samples.py            # TN A: Khảo sát quy mô mẫu (200 - 2000 mẫu)
│   ├── exp_B_width.py              # TN B: Khảo sát độ rộng tầng ẩn (B = 1 - 64)
│   └── exp_C_depth.py              # TN C: Khảo sát độ sâu mạng (C = 1 - 6 tầng)
├── figures/                        # Đồ thị, biểu đồ và trực quan hóa 3D (dpi=200/300)
│   ├── surface_3d_ground_truth.png     # Bề mặt hàm thực nghiệm Gaussian 2D lý thuyết
│   ├── scatter_3d_train_samples.png    # Đám mây điểm dữ liệu phân bố trong không gian
│   ├── exp_A_learning_curve.png        # Đường cong học tập theo số mẫu (Learning Curve)
│   ├── exp_B_width_curve.png           # Biểu đồ sai số và R² theo số nơ-ron
│   ├── exp_B_time_complexity.png       # Thời gian huấn luyện và số tham số theo B
│   ├── exp_C_depth_error_and_time.png  # Đánh giá đa chiều theo số tầng ẩn C
│   ├── exp_C_loss_convergence.png      # Đường cong hội tụ hàm mất mát theo epoch
│   ├── exp_C_best_surface_3d.png       # So sánh 3D Side-by-side: Thật vs Dự đoán
│   ├── exp_C_error_surface_3d.png      # Bề mặt sai số thặng dư tuyệt đối (|y - ŷ|)
│   └── champion_pred_vs_true.png       # Đồ thị tán xạ y thật vs ŷ dự đoán trên tập test
├── logs/                           # Nhật ký số liệu thực nghiệm (CSV)
│   ├── .gitkeep
│   ├── exp_A_summary.csv           # Bảng số liệu tổng hợp Thực nghiệm A
│   ├── exp_B_summary.csv           # Bảng số liệu tổng hợp Thực nghiệm B
│   └── exp_C_summary.csv           # Bảng số liệu tổng hợp Thực nghiệm C
├── report/                         # Báo cáo học thuật chuẩn định dạng SGU
│   ├── BTL01_BaoCao.docx           # Bản thảo Báo cáo khoa học Word (24 trang)
│   └── BTL01_BaoCao.pdf            # Bản in Báo cáo khoa học hoàn chỉnh
├── Slide/                          # Bài thuyết trình bảo vệ đồ án
│   ├── BTL01_ThuyetTrinh.pptx      # Slide thuyết trình PowerPoint 16:9 (15 slide)
│   └── Slide.pdf                   # Slide thuyết trình định dạng PDF
├── src/                            # Mã nguồn lõi (Core Engine Modules)
│   ├── __init__.py                 # Package initializer & exports API
│   ├── config.py                   # Cấu hình siêu tham số toàn cục (BASELINE)
│   ├── data.py                     # Sinh dữ liệu Gaussian 2D giải tích & I/O
│   ├── model.py                    # Kiến trúc MLPRegressor (nn.Module)
│   ├── trainer.py                  # Vòng lặp huấn luyện, đánh giá & checkpointing
│   ├── metrics.py                  # Các hàm đo lường MSE, MAE, RMSE, R²
│   ├── preprocessing.py            # Min-Max Scaler chuẩn hóa không rò rỉ
│   ├── plotting.py                 # Hàm vẽ scatter y-ŷ và đường chuẩn
│   └── visualize.py                # Trực quan hóa 3D chuyên sâu (Matplotlib 3D)
├── main.py                         # Điểm vào điều phối toàn bộ dự án (CLI CLI-based)
├── requirements.txt                # Danh sách thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn sử dụng và báo cáo tổng kết
```

---

## ⚙️ 3. CÀI ĐẶT MÔI TRƯỜNG

Dự án được xây dựng trên nền tảng **Python 3.10+** (đã kiểm định tương thích trên cả Python 3.12 và 3.14) cùng **PyTorch**.

```bash
# 1. Di chuyển vào thư mục dự án
cd BTL01

# 2. Cài đặt toàn bộ phụ thuộc cần thiết
pip install -r requirements.txt
```

> **Ghi chú thư viện:** `torch`, `numpy`, `matplotlib`, `pandas`, `scipy`, `python-docx`, `python-pptx`, `pypdf`.

---

## 🚀 4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH (DEMO CODE)

Chương trình cung cấp giao diện dòng lệnh linh hoạt (`main.py`) cho phép thực thi trọn gói hoặc kiểm tra từng cấu phần riêng lẻ:

### 4.1. Chạy quy trình mặc định (Đầy đủ từ A-Z)
Lệnh này sẽ tự động kiểm tra dữ liệu, huấn luyện mô hình Champion ($C=3$, $[32, 32, 32]$), tự động lưu trọng số tối ưu vào `checkpoints/best_model.pt`, đánh giá trên tập test và xuất biểu đồ so sánh:
```bash
python main.py
```

### 4.2. Khởi tạo dữ liệu và trực quan hóa 3D không gian
Tạo mới dữ liệu phân phối Gaussian 2D và xuất biểu đồ bề mặt thực tế cùng đám mây điểm:
```bash
python main.py --prepare-data --visualize-3d
```

### 4.3. Chạy từng nhóm thực nghiệm kiểm soát độc lập
Mỗi thực nghiệm lặp lại qua 3 random seeds (0, 1, 2) và tự động ghi log vào `logs/` cùng biểu đồ vào `figures/`:
```bash
# Chạy Thực nghiệm A: Khảo sát quy mô mẫu huấn luyện (10 mốc)
python main.py --run-exp A
# Hoặc chạy trực tiếp script:
python experiments/exp_A_samples.py

# Chạy Thực nghiệm B: Khảo sát độ rộng 1 tầng ẩn (13 mốc B = 1..64)
python main.py --run-exp B
# Hoặc chạy trực tiếp script:
python experiments/exp_B_width.py

# Chạy Thực nghiệm C: Khảo sát độ sâu mạng (6 mốc C = 1..6 tầng)
python main.py --run-exp C
# Hoặc chạy trực tiếp script:
python experiments/exp_C_depth.py

# Chạy liên hoàn cả 3 thực nghiệm A, B, C
python main.py --run-exp all
```

### 4.4. Đánh giá nhanh mô hình từ Checkpoint đã lưu
Không cần huấn luyện lại, nạp trực tiếp file `.pt` để kiểm tra độ chính xác trên 100 mẫu test:
```bash
python main.py --eval-checkpoint checkpoints/best_model.pt
```

---

## 📊 5. TỔNG HỢP KẾT QUẢ THỰC NGHIỆM

Toàn bộ kết quả dưới đây được tổng hợp từ dữ liệu chạy thực tế qua 3 random seeds ($seed \in \{0, 1, 2\}$) với trung bình và độ lệch chuẩn ($\mu \pm \sigma$):

### 5.1. Bảng đối chiếu 3 nhóm thực nghiệm

| Nhóm thực nghiệm | Biến độc lập | Phạm vi khảo sát | Cấu hình tốt nhất | Test MSE trung bình | $R^2$ Test | Ghi chú khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Thực nghiệm A** | Số mẫu train $A$ | $200 - 2.000$ mẫu | $A = 2.000$ mẫu | $2.520 \times 10^{-4}$ | $0.894$ | Sai số giảm $82.3\%$, xuất hiện hiệu ứng lợi suất giảm dần từ mốc $1.000$ mẫu. |
| **Thực nghiệm B** | Độ rộng tầng ẩn $B$ | $B \in [1, 64]$ nơ-ron | $B = 64$ (257 tham số) | $8.930 \times 10^{-6}$ | $0.996$ | Thoát khỏi vùng thiếu khớp nghiêm trọng ($B \le 3$), sai số giảm $28$ lần khi tăng từ $B=8 \to 64$. |
| **Thực nghiệm C** | Độ sâu mạng $C$ | $C \in [1, 6]$ tầng ẩn | **$C = 3$ (2.241 tham số)** | **$4.560 \times 10^{-7}$** | **$0.9998$** | **Điểm ngọt tối ưu (Champion).** $C \ge 4$ bị bão hòa và triệt tiêu đạo hàm nhẹ do hàm kích hoạt Tanh. |

### 5.2. Chi tiết kết quả mô hình Champion ($C = 3$, $[32, 32, 32]$)
* **Tổng số tham số học được:** $2.241$ tham số.
* **Thời gian huấn luyện:** $\approx 38.5$ giây (CPU tiêu chuẩn, 300 epochs).
* **Epoch đạt Validation Loss thấp nhất:** $286 / 300$ (MSE Validation $= 4.12 \times 10^{-7}$).
* **Chỉ số kiểm thử độc lập (Test Set - 100 mẫu):**
  * $\text{MSE}_{\text{test}} = 4.560 \times 10^{-7}$
  * $\text{MAE}_{\text{test}} = 0.000511$
  * $\text{RMSE}_{\text{test}} = 0.000675$
  * $R^2_{\text{test}} = 0.999802$ ($99.98\%$ độ biến thiên của hàm mật độ thực được giải thích hoàn hảo).

---

## 🎯 6. KẾT LUẬN VÀ BÀI HỌC KINH NGHIỆM

1. **Hiệu quả của chiều sâu so với chiều rộng:** Tăng chiều sâu từ $C=1$ lên $C=2, 3$ giúp giảm sai số tới 2 bậc độ lớn (từ $10^{-5}$ xuống $10^{-7}$) với chi phí tham số tăng rất khiêm tốn so với việc chỉ mở rộng nơ-ron trên 1 tầng ẩn.
2. **Hiện tượng triệt tiêu đạo hàm (Vanishing Gradient):** Khi sử dụng hàm kích hoạt $\tanh$ với độ sâu $C \ge 4$ mà không có kết nối tắt (Skip-connection/ResNet), đạo hàm bị co cụm khiến các tầng đầu học chậm, giải thích vì sao $C=4, 5, 6$ không cải thiện thêm sai số.
3. **Tính độc lập của dữ liệu kiểm thử:** Việc cô lập hoàn toàn tập Test 100 mẫu và chỉ sử dụng tập Validation 400 mẫu để lưu checkpoint giúp mô hình đạt khả năng tổng quát hóa xuất sắc, hoàn toàn không bị hiện tượng học vẹt hay quá khớp (Overfitting).

---

## 📦 7. ĐÓNG GÓI BÀI NỘP

Tệp tin nộp bài được đóng gói chuẩn theo quy định của giảng viên:
* **Tên file nén:** `BTL01_Nhom01_NguyenTruongCaoSon_3123580040.zip`
* **Nội dung bên trong:** Toàn bộ thư mục `BTL01` bao gồm mã nguồn sạch, dữ liệu, logs kết quả, biểu đồ minh họa, checkpoint mô hình tối ưu `best_model.pt`, báo cáo `.docx`/`.pdf`, slide `.pptx`/`.pdf` và `README.md`.

