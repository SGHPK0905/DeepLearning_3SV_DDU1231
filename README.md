# DeepLearning_3SV_DDU1231
# House Prices - Advanced Regression Techniques

## 1. Giới thiệu

Đây là bài tập nhóm môn **Học sâu (Deep Learning)**, thực hiện trên cuộc thi
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Mục tiêu của bài toán là dự đoán **SalePrice** của các căn nhà tại Ames, Iowa dựa trên các đặc trưng mô tả nhiều khía cạnh của căn nhà như diện tích, chất lượng, năm xây dựng, khu vực, garage, tầng hầm,...

Nhóm xây dựng **MLP bằng PyTorch** theo yêu cầu của học phần, sau đó mở rộng thực nghiệm với **XGBoost, CatBoost, Ridge, Lasso và ElasticNet**, đồng thời sử dụng **weighted ensemble** để kết hợp các mô hình có khả năng tạo ra các dự đoán bổ sung cho nhau.

Toàn bộ quy trình được tổ chức theo **CRISP-DM**.

---

## 2. Mục tiêu

Các mục tiêu chính của dự án:

- Khảo sát và hiểu bộ dữ liệu Ames Housing.
- Xử lý missing values và outlier.
- Chuẩn hóa dữ liệu đầu vào cho mô hình MLP.
- Xây dựng mô hình hồi quy bằng **PyTorch MLP**.
- Tuning hyperparameter để cải thiện MLP.
- Thử nghiệm các mô hình:
  - XGBoost
  - CatBoost
  - Ridge
  - Lasso
  - ElasticNet
- Kết hợp các mô hình bằng weighted ensemble.
- Kiểm tra khả năng tổng quát hóa bằng **5-Fold Out-of-Fold (OOF)**.
- Tạo file `submission.csv` để nộp lên Kaggle.

---

## 3. Dataset

### Train

- 1,460 mẫu ban đầu
- 81 cột, bao gồm `Id` và `SalePrice`

### Test

- 1,459 mẫu
- 80 cột
- Không có `SalePrice`

Sau khi xử lý outlier:

- Loại bỏ 4 mẫu có `GrLivArea > 4000`
- Train còn **1,456 mẫu**

Sau preprocessing:

- `X`: **1456 × 234**
- `X_test_kaggle`: **1459 × 234**
- Target được biến đổi bằng:

\[
y = \log(1 + SalePrice)
\]

---

## 4. Pipeline

```text
Raw Dataset
     ↓
Data Understanding
     ↓
EDA
     ↓
Outlier Removal
     ↓
Missing Value Imputation
     ↓
One-Hot Encoding
     ↓
Train / Validation Split
     ↓
StandardScaler
     ↓
log1p(SalePrice)
     ↓
┌─────────────────────────────┐
│       Modeling              │
│                             │
│  MLP → Tuning               │
│  XGBoost                    │
│  CatBoost                   │
│  Ridge / Lasso / ElasticNet │
└─────────────────────────────┘
     ↓
Weighted Ensemble
     ↓
5-Fold OOF
     ↓
Evaluation
     ↓
Submission

```

## 5. Data Preprocessing

Các bước tiền xử lý chính:

### Outlier Removal

Loại bỏ các mẫu trong tập train có:

```python
GrLivArea > 4000

Test không bị loại bỏ dòng.

### Missing Values

* Biến số: điền bằng `mean`
* Biến categorical: điền bằng `mode`

### Categorical Encoding

Sử dụng:

```python
pd.get_dummies(..., drop_first=True)
```

### Target Transformation

Biến mục tiêu được chuyển đổi:

```python
y = np.log1p(SalePrice)
```

### Standardization

Đối với MLP, `StandardScaler` được `fit` trên `X_train`, sau đó được sử dụng để transform `X_val` và `X_test_kaggle`.

Trong thí nghiệm OOF, scaler được fit riêng trên tập train của từng fold.

---

## 6. MLP bằng PyTorch

MLP được xây dựng bằng PyTorch với kiến trúc:

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Dropout
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

### MLP Baseline

```text
Input size      : 234
Hidden layer 1  : 32
Hidden layer 2  : 16
Dropout         : 0.20
Learning rate   : 0.001
Weight decay    : 0.0001
Batch size      : 32
Epochs          : 100
Optimizer       : Adam
Loss            : MSELoss
```

Validation RMSE baseline:

```text
1.0887
```

### MLP sau Tuning

Cấu hình tốt nhất:

```text
Hidden layer 1  : 24
Hidden layer 2  : 12
Dropout         : 0.20
Learning rate   : 0.00375
Weight decay    : 0.005
Batch size      : 32
Epochs          : 125
```

Validation RMSE:

```text
0.130914
```

---

## 7. Tuning MLP

Nhóm thử nghiệm bốn cấu hình với các giá trị Dropout khác nhau:

| Experiment | Hidden 1 | Hidden 2 | Learning Rate | Weight Decay |  Dropout | Validation RMSE |
| ---------: | -------: | -------: | ------------: | -----------: | -------: | --------------: |
|          1 |       24 |       12 |       0.00375 |        0.005 |     0.15 |        0.132151 |
|          2 |       24 |       12 |       0.00375 |        0.005 | **0.20** |    **0.130914** |
|          3 |       24 |       12 |       0.00375 |        0.005 |     0.25 |        0.133304 |
|          4 |       24 |       12 |       0.00375 |        0.005 |     0.30 |        0.134710 |

Cấu hình Experiment 2 được chọn làm MLP tốt nhất cho các thí nghiệm tiếp theo.

---

## 8. Các mô hình bổ sung

Sau khi có MLP tuned, nhóm thử nghiệm thêm các mô hình có cơ chế học khác nhau:

| Model        | Nhóm phương pháp  | Validation RMSE |
| ------------ | ----------------- | --------------: |
| MLP baseline | Neural Network    |          1.0887 |
| MLP tuned    | Neural Network    |        0.130914 |
| XGBoost      | Gradient Boosting |        0.124530 |
| CatBoost     | Gradient Boosting |          0.1242 |
| Ridge        | Linear + L2       |          0.1252 |
| Lasso        | Linear + L1       |          0.1240 |
| ElasticNet   | Linear + L1/L2    |          0.1239 |

Mục tiêu không chỉ là tìm mô hình đơn lẻ có RMSE thấp mà còn kiểm tra khả năng bổ sung lẫn nhau giữa các prediction khi kết hợp.

---

## 9. Ensemble

Các prediction được kết hợp bằng weighted average:

$$
\hat{y}_{ensemble}
=
\sum_k w_k \hat{y}_k,
\qquad
\sum_k w_k = 1
$$

### 9.1. MLP + XGBoost

```text
MLP : 0.38
XGB : 0.62

Validation RMSE = 0.119930
```

### 9.2. MLP + XGBoost + CatBoost

```text
MLP : 0.355
XGB : 0.355
CAT : 0.290

Validation RMSE = 0.119519
```

CatBoost được thử nghiệm trong một nhánh ensemble riêng để đánh giá khả năng bổ sung dự đoán.

Sau khi so sánh, nhóm quay lại champion MLP + XGBoost để tiếp tục thử nghiệm với các mô hình tuyến tính.

### 9.3. MLP + XGBoost + Lasso

Nhóm thử nghiệm Ridge, Lasso và ElasticNet trên champion MLP + XGBoost.

Kết quả tốt nhất đạt được khi:

```text
Champion MLP + XGBoost : 0.66
Lasso                  : 0.34
```

Suy ra trọng số cuối:

```text
MLP   : 0.2508
XGB   : 0.4092
Lasso : 0.3400
```

Validation RMSE:

```text
0.118342
```

---

## 10. 5-Fold Out-of-Fold

Nhóm sử dụng 5-Fold Cross-Validation theo phương pháp Out-of-Fold để kiểm tra khả năng tổng quát hóa của các mô hình.

### OOF RMSE

| Model    | OOF RMSE |
| -------- | -------: |
| MLP      | 0.146882 |
| XGBoost  | 0.119370 |
| CatBoost | 0.116351 |

Tối ưu trọng số trực tiếp trên prediction OOF cho kết quả:

```text
MLP       : 0.16
XGBoost   : 0.20
CatBoost  : 0.64

OOF RMSE  : 0.114819
```

OOF được sử dụng như một thí nghiệm kiểm tra khả năng tổng quát hóa và không được sử dụng trực tiếp làm công thức submission cuối cùng.

---

## 11. Evaluation

Mô hình cuối cùng sử dụng:

```text
MLP + XGBoost + Lasso
```

Trên 437 mẫu validation:

| Metric       |       Result |
| ------------ | -----------: |
| RMSE (log1p) | **0.118342** |
| MAE (log1p)  |     0.077854 |
| RMSE (USD)   |    19,585.99 |
| MAE (USD)    |    12,917.53 |

Kết quả đánh giá được lưu tại:

```text
experiments/house_mlp_v1/evaluation_metrics.csv
```

---

## 12. Submission

Notebook tạo file:

```text
submission.csv
```

File gồm đúng hai cột:

```text
Id
SalePrice
```

Số dòng:

```text
1459
```

Quy trình dự đoán:

```text
Test Data
    ↓
Preprocessing
    ↓
MLP Prediction
    ↓
XGBoost Prediction
    ↓
Lasso Prediction
    ↓
Weighted Ensemble
    ↓
expm1()
    ↓
SalePrice
```

Trọng số sử dụng trong submission:

```text
MLP   : 0.2508
XGB   : 0.4092
Lasso : 0.3400
```

---

## 13. Project Structure

```text
Lab02/
│
├── train.csv
├── test.csv
├── data_description.txt
│
├── House_price_pytorch.ipynb
│
├── experiments/
│   └── house_mlp_v1/
│       ├── best_model.pt
│       ├── best_tuned_model.pt
│       ├── params.json
│       ├── tuning_results.csv
│       ├── evaluation_metrics.csv
│       └── images/
│
├── submission.csv
├── README.md
└── report/
    └── Lab02.docx
```

---

## 14. Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch xgboost catboost
```

---

## 15. Cách chạy

### Bước 1

Clone repository:

```bash
git clone <repository-url>
cd Lab02
```

### Bước 2

Đặt các file dữ liệu vào thư mục project:

```text
train.csv
test.csv
data_description.txt
```

### Bước 3

Mở notebook bằng Jupyter Notebook hoặc VS Code.

```bash
jupyter notebook
```

### Bước 4

Chạy notebook theo thứ tự từ Section 1 đến Section 11.

Các file kết quả sẽ được tạo trong:

```text
experiments/house_mlp_v1/
```

File submission được tạo tại:

```text
submission.csv
```

---

## 16. Thành viên

| Thành viên            | Phụ trách                                |
| --------------------- | ---------------------------------------- |
| Đặng Thanh Phương     | Data Understanding & Data Preprocessing  |
| Nguyễn Trương Cao Sơn | Modeling, MLP PyTorch, Tuning & Ensemble |
| Phạm Trung Hiếu       | Evaluation, Deployment & Submission      |

---

## 17. Tài liệu tham khảo

* Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
* Dean De Cock: [Ames Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project](https://jse.amstat.org/v19n3/decock.pdf)

---

## 18. Ghi chú

Validation RMSE và Kaggle Public Score được tính trên các tập dữ liệu khác nhau. Vì vậy, Validation RMSE không được xem là Public Score Kaggle.

Các kết quả trong README cần được cập nhật nếu nhóm thay đổi preprocessing, model, ensemble weights hoặc chạy lại notebook với một pipeline mới.

