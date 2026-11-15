"""
BTL03: Chuyển đổi mã nguồn Deep Learning từ Keras/TensorFlow sang PyTorch
Cấu hình tập trung và thông tin nhóm
"""

from pathlib import Path
import torch

# Đường dẫn thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CONVERTED_DIR = BASE_DIR / "converted_pytorch"
SAMPLE_DIR = BASE_DIR / "sample"

# Đảm bảo các thư mục làm việc luôn sẵn sàng (không tạo report/Slide theo yêu cầu)
for d in [DATA_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_DIR, CONVERTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Thiết bị phần cứng (GPU nếu có, ngược lại CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hạt ngẫu nhiên tái lập kết quả
SEED = 42

# Thông tin môn học & nhóm thực hiện đồ án
GROUP_INFO = {
    'subject': 'Học sâu (Deep Learning)',
    'course_id': '858048',
    'class_name': 'DDU1231',
    'university': 'Trường Đại học Sài Gòn (SGU) – Khoa Công nghệ Thông tin',
    'instructor': 'TS. Đỗ Như Tài',
    'project_title': 'BTL03: CHUYỂN ĐỔI MÃ NGUỒN TỪ KERAS/TENSORFLOW SANG PYTORCH TRONG HỌC SÂU',
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
            'tasks': 'Chủ trì thiết kế kiến trúc PyTorch CNN + LSTM, SegNet và Car Object Detection; xây dựng CLI main.py; xây dựng bộ chuyển đổi tự động 13 notebook Keras sang PyTorch và kiểm chứng hội tụ.'
        },
        {
            'name': 'Đặng Thanh Phương',
            'student_id': '3123580036',
            'role': 'Thành viên',
            'contribution': '100%',
            'tasks': 'Chuyển đổi kiến trúc MLP (Gaussian Approximation & MNIST MLP) và CNNs (MNIST, Gender, Emotion); xây dựng vòng lặp huấn luyện PyTorch chuẩn (trainer.py), hệ thống Callbacks (EarlyStopping, Checkpoint, ReduceLROnPlateau).'
        },
        {
            'name': 'Phạm Trung Hiếu',
            'student_id': '3123580014',
            'role': 'Thành viên',
            'contribution': '100%',
            'tasks': 'Xây dựng pipeline nạp và xử lý dữ liệu PyTorch Dataset & DataLoader (data.py); chuyển đổi bộ Data Augmentation và Training Utilities; lập trình các chỉ số đánh giá (metrics.py); trực quan hóa (visualize.py) và tạo nhật ký Excel.'
        }
    ]
}

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_CLASSES_VI = ['Tức giận', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Buồn bã', 'Ngạc nhiên', 'Trung tính']
HAND_SEG_CLASSES = ['Palm', 'Finger 1', 'Finger 2', 'Finger 3', 'Finger 4', 'Finger 5', 'Background']
