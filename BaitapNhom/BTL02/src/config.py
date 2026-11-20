"""
BTL02: Facial Emotion Recognition (FER2013) using PyTorch CNNs
Configuration and Constants
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = BASE_DIR / "report"
SLIDE_DIR = BASE_DIR / "Slide"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

for d in [DATA_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR, REPORT_DIR, SLIDE_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset constants
RAW_CSV_PATH = DATA_DIR / "fer2013.csv"
PREPROCESSED_DIR = DATA_DIR / "preprocessing"
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_NPZ_PATH = PREPROCESSED_DIR / "train.npz"
VALID_NPZ_PATH = PREPROCESSED_DIR / "valid.npz"
TEST_NPZ_PATH = PREPROCESSED_DIR / "test.npz"

# 7 Emotion classes in FER2013
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CLASSES_VI = ['Tức giận', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Trung tính']
NUM_CLASSES = 7
IMAGE_SIZE = 48
IMAGE_CHANNELS = 1

# Hardware device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility seed
SEED = 42

# Default Training Hyperparameters
DEFAULT_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 35,
    'early_stopping_patience': 10,
    'lr_patience': 4,
    'lr_factor': 0.5,
    'min_lr': 1e-6,
    'dropout_rate': 0.25,
    'fc_dropout_rate': 0.5,
}

# Group metadata
GROUP_INFO = {
    'subject': 'Học sâu (Deep Learning)',
    'course_id': '858048',
    'class_name': 'DDU1231',
    'university': 'Trường Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin',
    'instructor': 'TS. Đỗ Như Tài',
    'leader': {
        'name': 'Nguyễn Trương Cao Sơn',
        'student_id': '3123580040',
        'role': 'Nhóm trưởng'
    },
    'members': [
        {
            'name': 'Nguyễn Trương Cao Sơn',
            'student_id': '3123580040',
            'role': 'Nhóm trưởng',
            'contribution': '100%',
            'tasks': 'Chủ trì thiết kế kiến trúc CNN VGG/ResNet; điều phối 4 chuỗi thực nghiệm A/B/C/D; xây dựng CLI main.py, trực quan hóa bộ lọc Conv Filters & Feature Maps; chủ trì biên soạn Báo cáo học thuật (.docx/.pdf) và Slide thuyết trình (.pptx/.pdf).'
        },
        {
            'name': 'Đặng Thanh Phương',
            'student_id': '3123580036',
            'role': 'Thành viên',
            'contribution': '100%',
            'tasks': 'Thiết kế kiến trúc Small-CNN và Medium-CNN; xây dựng vòng lặp huấn luyện PyTorch chuẩn (trainer.py), cơ chế Early Stopping, Learning Rate Scheduler; phụ trách thực nghiệm B (Optimizers) và thực nghiệm D (Regularization).'
        },
        {
            'name': 'Phạm Trung Hiếu',
            'student_id': '3123580014',
            'role': 'Thành viên',
            'contribution': '100%',
            'tasks': 'Xây dựng pipeline dữ liệu FER-2013 (data.py), tiền xử lý chuẩn hóa và bộ Data Augmentation đa dạng; tính toán bộ chỉ số đánh giá (metrics.py); phụ trách thực nghiệm C (Data Augmentation) và nhật ký thực nghiệm Excel.'
        }
    ]
}
