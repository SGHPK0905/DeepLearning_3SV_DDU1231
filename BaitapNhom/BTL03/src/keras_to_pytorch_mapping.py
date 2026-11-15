"""
BTL03: Bảng đối chiếu và Từ điển ánh xạ toàn diện từ Keras/TensorFlow sang PyTorch
Cung cấp lý thuyết, nguyên lý chuyển đổi kiến trúc, định dạng tensor và hàm tương đương.
"""

# Bảng đối chiếu kiến trúc và tư tưởng cốt lõi
PARADIGM_COMPARISON = [
    {
        'aspect': 'Mô hình tính toán (Computational Graph)',
        'keras': 'Static Graph / Declarative (Graph được compile trước bằng model.compile)',
        'pytorch': 'Dynamic Graph / Imperative (Eager execution, đồ thị dựng theo từng forward pass với Autograd)',
        'advantage': 'PyTorch linh hoạt vượt trội trong gỡ lỗi, điều kiện hóa luồng dữ liệu (if/else), RNN/LSTM độ dài biến đổi.'
    },
    {
        'aspect': 'Thứ tự chiều Tensor ảnh (Dimension Ordering)',
        'keras': 'NHWC (Batch, Height, Width, Channels) - mặc định TensorFlow',
        'pytorch': 'NCHW (Batch, Channels, Height, Width) - tối ưu phần cứng cuDNN',
        'advantage': 'Phải thực hiện permute(0, 3, 1, 2) khi chuyển đổi dữ liệu từ Keras sang PyTorch.'
    },
    {
        'aspect': 'Định nghĩa mô hình',
        'keras': 'Sequential(...) hoặc Functional API: Model(inputs=..., outputs=...)',
        'pytorch': 'Kế thừa class nn.Module, override __init__() và forward(x)',
        'advantage': 'Cấu trúc OOP rõ ràng, dễ dàng can thiệp vào từng bước biến đổi trung gian.'
    },
    {
        'aspect': 'Hàm mất mát và Softmax',
        'keras': 'Layer cuối dùng Dense(..., activation="softmax"), loss="categorical_crossentropy"',
        'pytorch': 'Layer cuối trả về Unnormalized Logits, loss là nn.CrossEntropyLoss() (tự tích hợp LogSoftmax + NLLLoss)',
        'advantage': 'Tránh lỗi tràn số (numerical underflow/overflow) của hàm exp trong Softmax thông thường.'
    },
    {
        'aspect': 'Vòng lặp huấn luyện',
        'keras': 'Hàm bao đóng model.fit(...) với các tham số epochs, batch_size, callbacks',
        'pytorch': 'Vòng lặp tường minh: optimizer.zero_grad() -> forward -> loss.backward() -> optimizer.step()',
        'advantage': 'Kiểm soát tuyệt đối gradient accumulation, gradient clipping, và multiple optimizers.'
    },
    {
        'aspect': 'Hệ thống tiện ích (Training Utilities)',
        'keras': 'Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)',
        'pytorch': 'Lớp tự định nghĩa kết hợp torch.optim.lr_scheduler và torch.utils.tensorboard.SummaryWriter',
        'advantage': 'Tùy biến không giới hạn logic dừng sớm và lưu trữ trọng số checkpoint.'
    }
]

# Bảng tra cứu trực tiếp từng Layer & API
API_MAPPING_TABLE = [
    # Tầng kết nối đầy đủ & Tensor cơ bản
    ('Input(shape=(H, W, C))', 'Tensor đầu vào x shape (B, C, H, W) trong forward(x)', 'Khai báo kích thước đầu vào'),
    ('Dense(units, activation="relu")', 'nn.Linear(in_features, units) + nn.ReLU()', 'Tầng kết nối đầy đủ (Fully Connected)'),
    ('Dense(1, activation="sigmoid")', 'nn.Linear(in_features, 1) + nn.Sigmoid()', 'Tầng hồi quy / phân loại nhị phân'),
    ('Flatten()', 'nn.Flatten() hoặc x.view(x.size(0), -1)', 'Làm phẳng tensor nhiều chiều thành vector 1D'),
    ('Dropout(rate=0.5)', 'nn.Dropout(p=0.5)', 'Triệt tiêu ngẫu nhiên nơ-ron chống quá khớp'),
    
    # Tầng tích chập & Pooling
    ('Conv2D(F, (K, K), padding="same", activation="relu")', 'nn.Conv2d(in_c, F, kernel_size=K, padding=K//2) + nn.ReLU()', 'Tích chập 2D giữ nguyên kích thước không gian'),
    ('Conv2D(F, (K, K), padding="valid", activation="relu")', 'nn.Conv2d(in_c, F, kernel_size=K, padding=0) + nn.ReLU()', 'Tích chập 2D không đệm biên'),
    ('MaxPooling2D(pool_size=(2, 2))', 'nn.MaxPool2d(kernel_size=2, stride=2)', 'Gộp cực đại giảm 1/2 kích thước không gian'),
    ('BatchNormalization()', 'nn.BatchNorm2d(num_features)', 'Chuẩn hóa theo lô cho feature map ảnh'),
    ('UpSampling2D(size=(2, 2))', 'nn.Upsample(scale_factor=2, mode="nearest")', 'Mở rộng kích thước ảnh cho mô hình SegNet'),
    
    # Tầng chuỗi thời gian & LSTM
    ('LSTM(units, return_sequences=True)', 'nn.LSTM(input_size, hidden_size, batch_first=True)', 'Tầng nơ-ron tái hiện LSTM cho dữ liệu chuỗi'),
    ('TimeDistributed(Conv2D(...))', 'Lặp forward qua các bước thời gian: [cnn(x[:, t]) for t in range(T)]', 'Áp dụng chung một mạng CNN cho từng frame chuỗi'),
    
    # Hàm mất mát (Loss functions)
    ('loss="categorical_crossentropy"', 'nn.CrossEntropyLoss() (với nhãn class index)', 'Hàm mất mát phân loại đa lớp'),
    ('loss="binary_crossentropy"', 'nn.BCEWithLogitsLoss() (với nhãn nhị phân float)', 'Hàm mất mát phân loại nhị phân'),
    ('loss="mse" / "mean_squared_error"', 'nn.MSELoss()', 'Hàm mất mát sai số bình phương trung bình'),
    
    # Thuật toán tối ưu (Optimizers)
    ('keras.optimizers.SGD(lr=0.01, momentum=0.9)', 'torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)', 'Gradient Descent với quán tính'),
    ('keras.optimizers.Adam(lr=0.001)', 'torch.optim.Adam(model.parameters(), lr=0.001)', 'Thuật toán Adam với adaptive learning rate'),
    ('keras.optimizers.RMSprop(lr=0.001)', 'torch.optim.RMSprop(model.parameters(), lr=0.001)', 'Thuật toán RMSprop ổn định gradient'),
    
    # Tiện ích huấn luyện
    ('ModelCheckpoint(filepath, save_best_only=True)', 'torch.save(model.state_dict(), filepath) khi val_loss cải thiện', 'Lưu trọng số mô hình tối ưu'),
    ('EarlyStopping(patience=5)', 'Class EarlyStopping theo dõi patience và gọi flag early_stop', 'Dừng huấn luyện sớm tránh quá khớp'),
    ('ReduceLROnPlateau(factor=0.5, patience=2)', 'torch.optim.lr_scheduler.ReduceLROnPlateau(...)', 'Giảm tốc độ học khi hàm mất mát chững lại'),
    ('ImageDataGenerator(...)', 'torchvision.transforms.Compose([RandomCrop, RandomHorizontalFlip, ...])', 'Tăng cường dữ liệu ảnh theo thời gian thực')
]
