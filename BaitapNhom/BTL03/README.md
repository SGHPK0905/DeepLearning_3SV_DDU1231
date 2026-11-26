# BÀI TẬP LỚN 03: CHUYỂN ĐỔI MÃ NGUỒN TỪ KERAS/TENSORFLOW SANG PYTORCH TRONG HỌC SÂU

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64%20%7C%20Linux-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed%20100%25-success.svg)]()

> **Học phần:** Học sâu (Deep Learning)  
> **Mã học phần:** 858048 — **Lớp:** DDU1231  
> **Trường:** Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin  
> **Giảng viên hướng dẫn:** TS. Đỗ Như Tài  
> **Nhóm thực hiện:** Nhóm 05  
> **Thời gian thực hiện:** Tháng 11/2026 — **Hạn nộp:** 23:59 29/11/2026  

---

## 👥 THÀNH VIÊN NHÓM VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | MSSV | Vai trò | Phân công nhiệm vụ chính | Đóng góp |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **Nguyễn Trương Cao Sơn** | `3123580040` | **Nhóm trưởng** | Thiết kế kiến trúc PyTorch CNN + LSTM, SegNet và Car Object Detection; xây dựng CLI `main.py`; xây dựng bộ chuyển đổi và thực thi tự động 13 notebook Keras sang PyTorch. | **100%** |
| 2 | **Đặng Thanh Phương** | `3123580036` | Thành viên | Chuyển đổi kiến trúc MLP (`GaussianMLP`, `MNIST_MLP`) và CNNs (`MNIST_CNN`, `Gender_CNN`, `Emotion_CNN`); xây dựng vòng lặp huấn luyện chuẩn (`src/trainer.py`) và hệ thống Callbacks. | **100%** |
| 3 | **Phạm Trung Hiếu** | `3123580014` | Thành viên | Xây dựng pipeline dữ liệu thật PyTorch (`src/data.py`), bộ chỉ số đánh giá (`src/metrics.py`), trực quan hóa (`src/visualize.py`) và nhật ký Excel. | **100%** |

---

## 📌 1. MỤC TIÊU VÀ PHẠM VI CHUYỂN ĐỔI

Đồ án kế thừa toàn bộ 13 file Jupyter Notebook mẫu viết bằng TensorFlow/Keras trong [sample/examples/](sample/examples/) do **TS. Đỗ Như Tài** cung cấp, chuyển đổi toàn diện sang **PyTorch 2.x** theo cấu trúc module hóa chuẩn công nghiệp tương đồng BTL01/BTL02:

1. **13/13 Notebook PyTorch độc lập (`converted_pytorch/`):** Khớp cấu trúc 1-1 với `sample/examples/`, giữ nguyên kiến trúc từng tầng mạng, sử dụng dữ liệu thật (`data/mnist.npz`, `depth_hand.zip`, `car_data.zip`, `test.jpg`, `BTL02/data/preprocessing/*.npz`) và đã được thực thi sẵn qua IPython kernel kèm đầy đủ output & đồ thị.
2. **Thư viện lõi hướng đối tượng (`src/`):** Chuẩn hóa mô hình kế thừa `nn.Module`, vòng lặp huấn luyện tường minh, callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`), metrics và module trực quan hóa 300 DPI.

---

## 🗺️ 2. BẢN ĐỒ ÁNH XẠ KERAS ↔ PYTORCH

| Thành phần | Keras / TensorFlow | PyTorch 2.x | Ghi chú kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Bố cục Tensor ảnh** | `NHWC` `(B, H, W, C)` | `NCHW` `(B, C, H, W)` | Hoán vị trục bằng `permute(0, 3, 1, 2)` |
| **Định nghĩa mô hình** | `Sequential` / `Model(inputs, outputs)` | Kế thừa `nn.Module`, cài đặt `forward(x)` | Đồ thị động (Dynamic Graph / Autograd) |
| **Tầng tích chập** | `Conv2D(F, (3,3), padding='same'/'valid')` | `nn.Conv2d(C_in, F, 3, padding=1/0)` | `padding='same'` tương ứng `padding=K//2` |
| **Phân loại đa lớp** | `Dense(K, activation='softmax')` + `categorical_crossentropy` | `nn.Linear(D, K)` trả logits thô + `nn.CrossEntropyLoss()` | Tự tích hợp LogSoftmax + NLLLoss chống tràn số |
| **Chuỗi ảnh (TimeDistributed)** | `TimeDistributed(Conv2D(...))` + `LSTM` | **Batch Folding:** `x.reshape(B*T, C, H, W)` qua CNN rồi `.view(B, T, -1)` vào `nn.LSTM` | Chạy song song toàn bộ bước thời gian trong 1 batch CNN |
| **Phân đoạn (SegNet)** | `UpSampling2D((2,2))` + `Reshape` + `Permute` | `nn.Upsample(scale_factor=2, mode='nearest')` + `nn.Conv2d(..., 1)` | `CrossEntropyLoss` nhận trực tiếp `(B, C, H, W)` |
| **Callbacks** | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` | Lớp `EarlyStopping`, `ModelCheckpoint` trong `src/trainer.py` + `optim.lr_scheduler.ReduceLROnPlateau` | Lưu và khôi phục `state_dict()` tốt nhất |

---

## 📁 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
BTL03/
├── converted_pytorch/              # 13 Notebooks PyTorch đã chuyển đổi 1-1 & thực thi sẵn output
│   ├── 0. Getting Started/
│   │   ├── getting_started.ipynb
│   │   └── Iris_Bayesian_decision_making.ipynb
│   ├── 1. MLP/
│   │   ├── Func_apxm_MLP.ipynb
│   │   └── MNIST_MLP.ipynb
│   ├── 2. CNN + MLP/
│   │   ├── 1_MNIST_CNN.ipynb
│   │   ├── 2_Human_Gender_CNN.ipynb
│   │   └── 3_Emotion_Reg_CNN.ipynb
│   ├── 3. CNN + LSTM/
│   │   └── MNIST_CNN_LSTM.ipynb
│   ├── 4. Training Utilities/
│   │   ├── 2_Human_Gender_CNN.ipynb
│   │   ├── 3_Emotion_Reg_CNN.ipynb
│   │   ├── checkpoints.PNG
│   │   └── tfboard.PNG
│   ├── 5. Others/
│   │   ├── 1_segnet.ipynb
│   │   ├── 2_obj_detection.ipynb
│   │   └── test.jpg
│   └── colab_getting_started.ipynb
├── src/                            # Mã nguồn lõi module hóa (OOP)
│   ├── models/                     # GaussianMLP, MNIST_MLP, MNIST_CNN, Gender_CNN, Emotion_CNN, MNIST_CNN_LSTM, SegNet, CarDetectorCNN
│   ├── config.py
│   ├── keras_to_pytorch_mapping.py
│   ├── data.py
│   ├── trainer.py
│   ├── metrics.py
│   ├── visualize.py
│   ├── convert_all_notebooks.py
│   └── generate_excel_log.py
├── experiments/
│   └── exp_benchmark_conversion.py # Thực nghiệm kiểm chứng trên dữ liệu thật
├── data/
│   └── mnist.npz                   # Bộ dữ liệu MNIST thật (60,000 train / 10,000 test)
├── figures/                        # Đồ thị 300 DPI minh chứng hội tụ & dự đoán
├── checkpoints/                    # Trọng số mô hình PyTorch (.pt)
├── logs/
│   └── benchmark_keras_vs_pytorch.csv
├── sample/                         # Kho mã nguồn Keras gốc của Giảng viên
├── main.py                         # CLI điều phối toàn dự án
├── requirements.txt
└── Phan_cong_nhiem_vu_va_Nhat_ky_BTL03.xlsx
```

---

## 🔬 4. KẾT QUẢ THỰC NGHIỆM THỰC ĐO (LOGS CHÍNH THỨC)

Kết quả chạy thực tế trên CPU (`logs/benchmark_keras_vs_pytorch.csv`):

| Mô hình PyTorch | Số tham số | Chỉ số đánh giá | Kết quả thực đo | Val / Test Loss | Thời gian (s) | Nguồn dữ liệu |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GaussianMLP** (`B=64`) | 257 | $R^2$ Score | **99.93%** | `0.000001` | 12.44 | Gaussian 2D Grid (`32x32` / `10x10`) |
| **MNIST_MLP** | 669,706 | Test Acc | **98.46%** | `0.0616` | 65.69 | MNIST Real (`60,000` / `10,000`) |
| **MNIST_CNN** (3 Blocks) | 1,817,610 | Test Acc | **96.40%** | `0.0969` | 65.23 | MNIST Real (`15,000` / `3,000`) |
| **Emotion_CNN** (19M params) | 19,028,287 | Test Acc | **22.33%** | `1.8515` | 128.96 | FER-2013 Real (subset CPU `1,200` / `300`, 5 epochs)* |
| **CNN + LSTM** (Digit Sum) | 1,748,097 | $R^2$ Score | **79.35%** | `2.3866` (MAE) | 31.89 | MNIST Sequences Real (`1,200` / `300`) |
| **SegNet** (7 classes) | 574,229 | Pixel Acc | **81.46%** | `0.8357` | 126.76 | `depth_hand.zip` Real (`240` / `80`) |
| **CarDetectorCNN** (FCN) | 83,367 | Test Acc | **94.33%** | `0.2539` (MSE) | 8.61 | `car_data.zip` Real (`1,000` imgs) |

> *\*Ghi chú: Kiến trúc `Emotion_CNN` (6 lớp Conv tới 218 kênh, không có BatchNorm, ~19M tham số) tái hiện đúng 1-1 cấu trúc trong `3_Emotion_Reg_CNN.ipynb`; trên CPU chạy nhanh 5 epoch với 1,200 ảnh mẫu chỉ đạt ngưỡng cơ bản, cần chạy đầy đủ toàn tập FER-2013 (28,709 ảnh, 15–35 epoch trên GPU) như đã khảo sát chuyên sâu tại BTL02.*

---

## 🚀 5. HƯỚNG DẪN CHẠY LẠI HỆ THỐNG

```bash
pip install -r requirements.txt

# Chạy toàn bộ quy trình (chuyển đổi + chạy 13 notebooks + benchmark + xuất Excel)
python main.py --all

# Hoặc chạy từng bước:
python main.py --convert_notebooks   # Tạo 13 file .ipynb
python main.py --execute_notebooks   # Tạo và thực thi sẵn output cho 13 file .ipynb
python main.py --run_benchmarks      # Chạy kiểm chứng 7 mô hình và xuất biểu đồ
python main.py --generate_excel      # Xuất file Excel phân công và nhật ký
```
