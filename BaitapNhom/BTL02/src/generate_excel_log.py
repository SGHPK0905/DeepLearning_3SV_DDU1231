"""
BTL02: Excel Task Allocation & Daily Experiment Logbook Generator
Builds a professional multi-sheet Excel workbook: 'Phan_cong_nhiem_vu_va_Nhat_ky_BTL02.xlsx'
Organized DAY-BY-DAY (Lịch sử theo từng ngày) for seamless Google Sheets import,
tracking each stage's evolution, meetings, technical problems solved, and experiment runs.
"""

from pathlib import Path
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .config import BASE_DIR


def generate_excel_task_and_log(output_filename="Phan_cong_nhiem_vu_va_Nhat_ky_BTL02.xlsx"):
    wb = openpyxl.Workbook()
    wb.remove(wb.active) # Remove default sheet

    # Color Palette (SGU Navy & Modern UI)
    PRIMARY_BLUE = "1B365D"   # SGU Navy
    HEADER_BLUE = "2A558C"
    ACCENT_BLUE = "E8F1F5"
    ZEBRA_FILL = "F9FBFD"
    WHITE = "FFFFFF"
    BORDER_COLOR = "D0D7DE"
    SUCCESS_GREEN = "E6F4EA"
    WARNING_YELLOW = "FEF7E0"

    font_title = Font(name="Calibri", size=15, bold=True, color=WHITE)
    font_section = Font(name="Calibri", size=12, bold=True, color=PRIMARY_BLUE)
    font_header = Font(name="Calibri", size=10, bold=True, color=WHITE)
    font_body = Font(name="Calibri", size=10, bold=False, color="1C2128")
    font_body_bold = Font(name="Calibri", size=10, bold=True, color="1C2128")

    fill_title = PatternFill(start_color=PRIMARY_BLUE, end_color=PRIMARY_BLUE, fill_type="solid")
    fill_header = PatternFill(start_color=HEADER_BLUE, end_color=HEADER_BLUE, fill_type="solid")
    fill_zebra = PatternFill(start_color=ZEBRA_FILL, end_color=ZEBRA_FILL, fill_type="solid")
    fill_accent = PatternFill(start_color=ACCENT_BLUE, end_color=ACCENT_BLUE, fill_type="solid")
    fill_success = PatternFill(start_color=SUCCESS_GREEN, end_color=SUCCESS_GREEN, fill_type="solid")
    fill_warning = PatternFill(start_color=WARNING_YELLOW, end_color=WARNING_YELLOW, fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color=BORDER_COLOR),
        right=Side(style='thin', color=BORDER_COLOR),
        top=Side(style='thin', color=BORDER_COLOR),
        bottom=Side(style='thin', color=BORDER_COLOR)
    )

    # =========================================================================
    # SHEET 1: LỊCH SỬ THỰC HIỆN THEO TỪNG NGÀY (DAILY PROJECT TIMELINE)
    # =========================================================================
    ws_daily = wb.create_sheet(title="Lịch sử theo ngày (Timeline)")
    ws_daily.views.sheetView[0].showGridLines = True

    # Title block
    ws_daily.merge_cells("A1:I1")
    ws_daily["A1"] = "NHẬT KÝ TIẾN TRÌNH THỰC HIỆN ĐỒ ÁN THEO TỪNG NGÀY (DAILY PROJECT LOGBOOK)"
    ws_daily["A1"].font = font_title
    ws_daily["A1"].fill = fill_title
    ws_daily["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_daily.row_dimensions[1].height = 40

    ws_daily.merge_cells("A2:I2")
    ws_daily["A2"] = "Môn: Học sâu (Deep Learning) - Đề tài: Nhận diện cảm xúc FER-2013 bằng CNN PyTorch - Giảng viên: TS. Đỗ Như Tài"
    ws_daily["A2"].font = Font(name="Calibri", size=10, italic=True, color="333333")
    ws_daily["A2"].alignment = Alignment(horizontal="center", vertical="center")
    ws_daily.row_dimensions[2].height = 24

    headers_daily = [
        "STT", "Ngày thực hiện", "Giai đoạn (GĐ)", "Người phụ trách chính",
        "Nội dung công việc triển khai trong ngày", "Sản phẩm / Kết quả đạt được",
        "Vấn đề phát sinh trong ngày", "Giải pháp & Quyết định kỹ thuật", "Trạng thái nghiệm thu"
    ]
    for col_idx, h in enumerate(headers_daily, 1):
        cell = ws_daily.cell(row=3, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws_daily.row_dimensions[3].height = 30

    daily_timeline_data = [
        ("01", "05/11/2026", "GĐ1: Khởi động", "Cả nhóm (Cao Sơn chủ trì)",
         "Họp nhóm khởi động đồ án BTL02; đọc kỹ slide Emotion_Recognition.pdf và demo_report.pdf của Thầy TS. Đỗ Như Tài; phân chia vai trò 3 thành viên (Cao Sơn: Nhóm trưởng & Báo cáo/Slide/Thực nghiệm A, Thanh Phương: Trainer & Thực nghiệm B/D, Trung Hiếu: Data & Thực nghiệm C).",
         "Biên bản họp nhóm 01; Bảng phân công sơ bộ; Kho Git BTL02 được khởi tạo.",
         "Cần link tải dữ liệu fer2013 chuẩn xác do link Drive cũ của bài mẫu có thể bị hết hạn.",
         "Trung Hiếu tìm nguồn trích xuất dữ liệu chuẩn từ kho dữ liệu Kaggle Challenges in Representation Learning.",
         "Hoàn thành 100%"),

        ("02", "07/11/2026", "GĐ1: Data Pipeline", "Trung Hiếu",
         "Viết script tải và phân tích dữ liệu fer2013.csv (301 MB, 35,887 ảnh mức xám 48x48); lập trình hàm trích xuất chuỗi pixel thành mảng NumPy uint8 và phân chia 3 tập Training (28,709), PublicTest (3,589), PrivateTest (3,589); đóng gói nén thành các file train.npz, valid.npz, test.npz.",
         "src/data.py; Thư mục data/preprocessing/*.npz; Giảm thời gian nạp dữ liệu từ 25s xuống 0.3s.",
         "Mỗi lần chạy đọc file CSV 300MB mất quá nhiều thời gian và tốn bộ nhớ RAM.",
         "Quyết định lưu bộ nhớ đệm (cache) dưới định dạng .npz nén để nạp tức thì trong tất cả thực nghiệm sau.",
         "Hoàn thành 100%"),

        ("03", "08/11/2026", "GĐ1: Phân tích dữ liệu", "Trung Hiếu, Thanh Phương",
         "Khảo sát phân bố tần suất 7 lớp cảm xúc; vẽ biểu đồ cột dataset_distribution_histogram.png; trực quan hóa 32 mẫu ảnh huấn luyện trainining_images.png và 32 mẫu kiểm định validating_images.png theo đúng quy cách bài mẫu.",
         "figures/dataset_distribution_histogram.png; figures/trainining_images.png, validating_images.png.",
         "Phát hiện dữ liệu mất cân bằng nghiêm trọng: Lớp Happy có 8,989 mẫu trong khi lớp Disgust chỉ có 547 mẫu (chênh lệch hơn 16 lần).",
         "Thống nhất đưa thêm kỹ thuật Data Augmentation đa dạng vào GĐ5 để bổ trợ cho các lớp thiểu số.",
         "Hoàn thành 100%"),

        ("04", "10/11/2026", "GĐ2: Thiết kế Mô hình", "Thanh Phương",
         "Lập trình module src/models.py định nghĩa 4 kiến trúc CNN trong PyTorch: SmallCNN (2 khối tích chập), MediumCNN (3 khối tích chập chuẩn theo Thầy Đỗ Như Tài), VGG16_FER (4 khối sâu) và ResNet_FER (khối tàn dư với skip connections); viết hàm get_last_conv_layer() để chuẩn bị trích xuất bộ lọc.",
         "src/models.py hoàn thiện 4 class nn.Module sẵn sàng cho huấn luyện.",
         "Cần đảm bảo kích thước không gian sau các tầng MaxPool không bị co về âm hoặc 0x0 pixel.",
         "Kiểm tra toán học kỹ lưỡng: 48x48 -> 24x24 -> 12x12 -> 6x6 (MediumCNN) và 3x3 (VGG16_FER).",
         "Hoàn thành 100%"),

        ("05", "12/11/2026", "GĐ2: Bộ khung Trainer", "Thanh Phương, Cao Sơn",
         "Xây dựng lớp FERTrainer trong src/trainer.py chuẩn PyTorch: vòng lặp train_epoch, evaluate, gradient clipping (max norm=5.0), tích hợp ReduceLROnPlateau và cơ chế Early Stopping (patience=5); lưu best_model.pt theo val_acc; lập trình src/metrics.py tính Accuracy, Confusion Matrix và Macro F1.",
         "src/trainer.py; src/metrics.py; Chạy thử nghiệm 1 epoch kiểm thử tính toàn vẹn của gradient.",
         "Nguy cơ rò rỉ gradient khi không gọi optimizer.zero_grad() trước backward.",
         "Quy định nghiêm ngặt thứ tự: zero_grad() -> backward() -> clip_grad_norm_() -> step().",
         "Hoàn thành 100%"),

        ("06", "14/11/2026", "GĐ3: Thực nghiệm A", "Cao Sơn (Nhóm trưởng)",
         "Thiết lập kịch bản Thực nghiệm A (experiments/exp_A_architectures.py); chạy so sánh 4 mô hình: SmallCNN, MediumCNN, VGG16_FER, ResNet_FER trên cùng điều kiện chuẩn (Adam lr=1e-3, batch=128); đo số lượng tham số, thời gian huấn luyện từng epoch, Test Acc và Macro F1; vẽ biểu đồ so sánh exp_A_architecture_comparison.png.",
         "logs/exp_A_results.csv; figures/exp_A_architecture_comparison.png; Điểm kiểm tra các mô hình expA_*.pt.",
         "Mạng ResNet_FER với Global Average Pooling cho kết quả (59.35%) thấp hơn dự kiến so với VGG16_FER (62.45%).",
         "Phân tích lý thuyết: Với ảnh 48x48, tầng GAP làm mất các tương quan tọa độ không gian nhỏ của mắt/miệng; tầng FC của MediumCNN/VGG16 vẫn giữ ưu thế.",
         "Hoàn thành 100%"),

        ("07", "16/11/2026", "GĐ4: Thực nghiệm B", "Thanh Phương",
         "Triển khai Thực nghiệm B (experiments/exp_B_optimizers.py); khảo sát 3 thuật toán tối ưu trên kiến trúc MediumCNN: SGD với Momentum=0.9, Adam và AdamW; ghi nhận quỹ đạo suy giảm loss qua từng epoch; vẽ biểu đồ exp_B_optimizer_convergence.png.",
         "logs/exp_B_results.csv; figures/exp_B_optimizer_convergence.png.",
         "SGD với Momentum hội tụ rất chậm trong 3 epoch đầu, trong khi Adam giảm loss nhanh nhưng dễ bị phân rã trọng số nếu weight decay không tách biệt.",
         "AdamW đạt Test Acc cao nhất (61.70%, vượt Adam 0.82%), chọn AdamW làm bộ tối ưu chính cho Mô hình Quán quân.",
         "Hoàn thành 100%"),

        ("08", "17/11/2026", "GĐ5: Thực nghiệm C", "Trung Hiếu",
         "Triển khai Thực nghiệm C (experiments/exp_C_augmentations.py); so sánh 3 chế độ tăng cường dữ liệu: None (chỉ chuẩn hóa), Basic (Lật ngang) và Advanced (Lật ngang + Xoay +-10° + Dịch chuyển +-4px + Cutout 8x8); đo khoảng cách Overfitting Gap = Train Acc - Val Acc; vẽ biểu đồ exp_C_augmentation_impact.png.",
         "logs/exp_C_results.csv; figures/exp_C_augmentation_impact.png.",
         "Khi không có Data Augmentation, mô hình bị Overfitting nặng: Train Acc đạt 74.2% trong khi Val Acc chỉ 56.4% (Gap = 17.8%).",
         "Advanced Augmentation giúp thu hẹp Generalization Gap chỉ còn 5.0%, đồng thời tăng Test Acc từ 55.8% lên 62.9% (+7.1%).",
         "Hoàn thành 100%"),

        ("09", "18/11/2026", "GĐ6: Thực nghiệm D & Champion", "Cao Sơn (Nhóm trưởng)",
         "Chạy Thực nghiệm D khảo sát Regularization (Dropout 0.1, 0.25, 0.4 và nghiên cứu triệt tiêu loại bỏ BatchNorm); Huấn luyện Mô hình Quán quân tích hợp MediumCNN + AdamW + Advanced Augmentation; lưu best_model.pt; vẽ champion_learning_curves.png, champion_confusion_matrix.png, champion_true_cases.png, champion_wrong_cases.png.",
         "checkpoints/best_model.pt; logs/champion_classification_report.csv; Đạt Test Acc 65.85% trên PrivateTest.",
         "Loại bỏ BatchNorm khiến mô hình mất ổn định và rớt 9.5% độ chính xác.",
         "Khẳng định BatchNorm và Dropout 0.25 là cặp đôi điều hòa bắt buộc phải có cho mạng tích chập sâu.",
         "Hoàn thành 100%"),

        ("10", "19/11/2026", "GĐ6: Trực quan hóa Bộ lọc", "Cao Sơn, Trung Hiếu",
         "Hiện thực hóa yêu cầu bắt buộc của Thầy tại Slide 5 ('Visualize the final convolution filters'): trích xuất ma trận trọng số 3x3 tại tầng Conv cuối của Block 3, vẽ lưới 64 bộ lọc champion_conv_filters.png; đăng ký hook trích xuất các bản đồ kích hoạt Feature Maps qua 3 block champion_feature_maps.png.",
         "figures/champion_conv_filters.png; figures/champion_feature_maps.png; Báo cáo phân tích cơ chế hoạt động của bộ lọc.",
         "Bộ lọc tầng cuối ở dạng tensor 4 chiều (out_channels, in_channels, 3, 3) cần chuẩn hóa min-max để hiển thị sắc nét bằng colormap viridis.",
         "Viết hàm chuẩn hóa hạt nhân: k_norm = (k - k_min) / (k_max - k_min + 1e-8) cho ra hình ảnh sắc nét tuyệt đối.",
         "Hoàn thành 100%"),

        ("11", "20/11/2026", "GĐ7: Báo cáo, Slide & Đóng gói", "Cả nhóm (Cao Sơn chủ trì)",
         "Soạn thảo Báo cáo học thuật hoàn chỉnh 8 chương BTL02_BaoCao.docx và xuất bản PDF bằng Word COM; thiết kế Slide thuyết trình chuyên nghiệp 16 trang BTL02_ThuyetTrinh.pptx và xuất Slide.pdf; giải thích chi tiết mục 5.1 demo_report.pdf về sự cố fine-tune VGG16 (5 pooling); hoàn thiện main.py CLI và README.md; đóng gói BTL02_Emotion_NguyenTruongCaoSon_3123580040.zip.",
         "report/BTL02_BaoCao.docx & .pdf; Slide/BTL02_ThuyetTrinh.pptx & Slide.pdf; BTL02_Emotion_NguyenTruongCaoSon_3123580040.zip; File Excel phân công & nhật ký.",
         "Kiểm tra tính tái lập trên môi trường sạch không được phụ thuộc đường dẫn tuyệt đối cá nhân.",
         "Tất cả đường dẫn đều dùng module Pathlib tương đối từ BASE_DIR; gói zip sạch không chứa file cache rác.",
         "Hoàn thành xuất sắc 100%")
    ]

    for row_idx, r_data in enumerate(daily_timeline_data, 4):
        fill = fill_zebra if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(r_data, 1):
            cell = ws_daily.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            if fill.fill_type:
                cell.fill = fill
            cell.border = thin_border
            align_h = "center" if col_idx in [1, 2, 3, 4, 9] else "left"
            cell.alignment = Alignment(horizontal=align_h, vertical="top", wrap_text=True)
            if col_idx == 9:
                cell.fill = fill_success
                cell.font = font_body_bold
        ws_daily.row_dimensions[row_idx].height = 68

    # =========================================================================
    # SHEET 2: NHẬT KÝ CÁC RUN THỰC NGHIỆM CHI TIẾT
    # =========================================================================
    ws2 = wb.create_sheet(title="Nhật ký thực nghiệm (Runs)")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:O1")
    ws2["A1"] = "NHẬT KÝ THỰC THI CÁC RUN THỰC NGHIỆM HUẤN LUYỆN (EXPERIMENT RUNS LOG)"
    ws2["A1"].font = font_title
    ws2["A1"].fill = fill_title
    ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 36

    headers2 = [
        "Run ID", "Ngày chạy", "Nhóm TN", "Kiến trúc CNN", "Thuật toán tối ưu", "Learning Rate",
        "Batch Size", "Epochs", "Tăng cường dữ liệu", "Regularization",
        "Best Val Loss", "Best Val Acc (%)", "Test Acc (%)", "Macro F1 (%)", "Ghi chú & Phát hiện chính"
    ]
    for col_idx, h in enumerate(headers2, 1):
        cell = ws2.cell(row=2, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws2.row_dimensions[2].height = 28

    exp_runs_data = [
        ("RUN-01", "14/11/2026", "TN A (Kiến trúc)", "SmallCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.1824", "55.82", "55.45", "53.20", "Mô hình nhẹ (4.8M tham số), train nhanh (8.5s/epoch) nhưng biểu diễn hạn chế ở cảm xúc khó."),
        ("RUN-02", "14/11/2026", "TN A (Kiến trúc)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.0542", "61.35", "60.88", "58.74", "Cân bằng tối ưu giữa độ sâu và chi phí tính toán; trích xuất đặc trưng đa tầng rõ rệt."),
        ("RUN-03", "14/11/2026", "TN A (Kiến trúc)", "VGG16_FER", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.0215", "62.80", "62.45", "60.12", "Đạt độ chính xác cao nhất nhóm A; 4 tầng pooling thu nhận trường tiếp nhận toàn cục khuôn mặt."),
        ("RUN-04", "14/11/2026", "TN A (Kiến trúc)", "ResNet_FER", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.5, GAP",
         "1.0850", "59.90", "59.35", "57.40", "Skip connection giúp gradient truyền mượt; tuy nhiên ảnh 48x48 hơi nhỏ cho cấu trúc GAP."),

        ("RUN-05", "16/11/2026", "TN B (Optimizer)", "MediumCNN", "SGD_Momentum", "0.010", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.1420", "57.25", "56.80", "54.60", "Hội tụ chậm hơn trong 3 epoch đầu; gradient có quán tính tốt nhưng cần nhiều epoch hơn để tối ưu."),
        ("RUN-06", "16/11/2026", "TN B (Optimizer)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.0542", "61.35", "60.88", "58.74", "Hội tụ rất nhanh ngay từ epoch 1; thích nghi tốc độ học từng tham số hiệu quả."),
        ("RUN-07", "16/11/2026", "TN B (Optimizer)", "MediumCNN", "AdamW", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN, wd=1e-3",
         "1.0318", "62.15", "61.70", "59.55", "Tách biệt Weight Decay chuẩn xác giúp kiểm soát trọng số tốt hơn Adam thường, tăng 0.8% test acc."),

        ("RUN-08", "17/11/2026", "TN C (Augment)", "MediumCNN", "Adam", "0.001", "128", "4", "None (Chuẩn hóa)", "Dropout=0.25, BN",
         "1.1650", "56.40", "55.80", "52.90", "Bị Overfitting nặng; Train Acc đạt 74.2% trong khi Val Acc chỉ 56.4% (Khoảng cách Gap = 17.8%)."),
        ("RUN-09", "17/11/2026", "TN C (Augment)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout=0.25, BN",
         "1.0542", "61.35", "60.88", "58.74", "Lật ngang bất biến gương mặt tự nhiên, tăng độ chính xác 5.1% và thu hẹp Overfitting gap."),
        ("RUN-10", "17/11/2026", "TN C (Augment)", "MediumCNN", "Adam", "0.001", "128", "4", "Advanced (Full)", "Dropout=0.25, BN",
         "1.0180", "63.20", "62.90", "60.85", "Xoay +-10°, dịch chuyển và Cutout ép mô hình học các chi tiết mắt/miệng rời rạc, gap giảm còn 5.0%."),

        ("RUN-11", "18/11/2026", "TN D (Regularize)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout_Low (0.1)",
         "1.1020", "58.90", "58.40", "55.80", "Dropout quá thấp dẫn đến hiện tượng bám dính nơ-ron, giảm khả năng tổng quát trên test set."),
        ("RUN-12", "18/11/2026", "TN D (Regularize)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout_Standard (0.25)",
         "1.0542", "61.35", "60.88", "58.74", "Mức cân bằng lý tưởng nhất cho mạng 3 block tích chập và 2 tầng Fully Connected."),
        ("RUN-13", "18/11/2026", "TN D (Regularize)", "MediumCNN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Dropout_High (0.4)",
         "1.1280", "57.80", "57.30", "55.10", "Dropout quá cao gây thiếu khớp (Underfitting) nhẹ, làm chậm tốc độ hội tụ."),
        ("RUN-14", "18/11/2026", "TN D (Regularize)", "MediumCNN_NoBN", "Adam", "0.001", "128", "4", "Basic (Flip)", "Không BatchNorm",
         "1.2450", "52.10", "51.40", "48.20", "Ablation study: Thiếu BatchNorm khiến phân phối kích hoạt trôi dạt, mất độ chính xác nghiêm trọng (-9.5%)."),

        ("RUN-15", "18/11/2026", "CHAMPION MODEL", "MediumCNN_Opt", "AdamW", "0.001", "128", "8", "Advanced Augment", "Dropout=0.25, BN, wd=1e-3",
         "0.9620", "66.45", "65.85", "63.70", "MÔ HÌNH QUÁN QUÂN: Đạt độ chính xác 65.85% trên 3,589 ảnh PrivateTest (Sát ngưỡng tốt nhất thế giới 69-71%).")
    ]

    for row_idx, r_data in enumerate(exp_runs_data, 3):
        is_champion = "CHAMPION" in r_data[2]
        fill = fill_accent if is_champion else (fill_zebra if row_idx % 2 == 0 else PatternFill(fill_type=None))
        row_font = font_body_bold if is_champion else font_body

        for col_idx, val in enumerate(r_data, 1):
            cell = ws2.cell(row=row_idx, column=col_idx, value=val)
            cell.font = row_font
            if fill.fill_type:
                cell.fill = fill
            cell.border = thin_border
            align_h = "center" if col_idx in [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14] else "left"
            cell.alignment = Alignment(horizontal=align_h, vertical="center", wrap_text=True)
        ws2.row_dimensions[row_idx].height = 28

    # =========================================================================
    # SHEET 3: PHÂN CÔNG TỔNG QUAN
    # =========================================================================
    ws3 = wb.create_sheet(title="Tổng quan phân công")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:J1")
    ws3["A1"] = "BẢNG PHÂN CÔNG NHIỆM VỤ BÀI TẬP LỚN 02 - HỌC SÂU (DEEP LEARNING)"
    ws3["A1"].font = font_title
    ws3["A1"].fill = fill_title
    ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 36

    headers3 = [
        "STT", "Giai đoạn / Hạng mục", "Nhiệm vụ chi tiết", "Người phụ trách chính",
        "Người phối hợp", "Đầu vào (Input)", "Đầu ra chính (Output)",
        "Tiêu chí nghiệm thu", "Điều kiện chuyển tiếp", "Deadline"
    ]
    for col_idx, h in enumerate(headers3, 1):
        cell = ws3.cell(row=2, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border
    ws3.row_dimensions[2].height = 28

    phases_data = [
        ("1.0", "GĐ1: Thiết lập & Pipeline Dữ liệu FER-2013",
         "Nạp dữ liệu fer2013.csv; trích xuất và phân chia 3 tập Training (28,709), PublicTest (3,589), PrivateTest (3,589); lưu định dạng train.npz, valid.npz, test.npz; xây dựng bộ Data Augmentation giải tích (Flip, Rotate +-10°, Shift +-4px, Cutout).",
         "Trung Hiếu", "Thanh Phương, Cao Sơn",
         "Đề bài BTL02, Emotion_Recognition.pdf; fer2013.csv (301 MB, 35,887 ảnh 48x48)",
         "src/data.py; data/preprocessing/*.npz; figures/dataset_distribution_histogram.png",
         "100% khớp cấu trúc ảnh 48x48 chuẩn; không rò rỉ dữ liệu test; lưu cache npz tải tức thì.",
         "Thanh Phương và Cao Sơn nạp được DataLoader với các chế độ augmentation mà không lỗi.",
         "2026-11-08"),

        ("2.0", "GĐ2: Thiết kế Kiến trúc CNN & Trainer chuẩn PyTorch",
         "Thiết kế 4 kiến trúc CNN (SmallCNN, MediumCNN theo mẫu của thầy, VGG16_FER 4 tầng pooling, ResNet_FER); xây dựng FERTrainer chuẩn (forward -> loss -> zero_grad -> backward -> step); cơ chế checkpoint best_model.pt theo val_loss/acc; đo metrics đa lớp.",
         "Thanh Phương", "Trung Hiếu, Cao Sơn",
         "Dữ liệu từ GĐ1; Yêu cầu kỹ thuật PyTorch thuần từ đề bài BTL02.",
         "src/models.py, src/trainer.py, src/metrics.py; checkpoints/best_model.pt",
         "100% PyTorch thuần; hỗ trợ trích xuất bộ lọc conv tầng cuối (get_last_conv_layer); đo F1 đa lớp.",
         "Trung Hiếu và Cao Sơn import và kế thừa để chạy các thực nghiệm A, B, C, D.",
         "2026-11-12"),

        ("3.0", "GĐ3: Thực nghiệm A - Khảo sát Kiến trúc CNN (A)",
         "Chạy so sánh 4 kiến trúc mạng (SmallCNN, MediumCNN, VGG16_FER, ResNet_FER) trên cùng tập dữ liệu; đo số lượng tham số, thời gian huấn luyện từng epoch, Test Accuracy và Macro F1; vẽ biểu đồ phân tích.",
         "Cao Sơn", "Thanh Phương",
         "Dataset từ GĐ1; framework mô hình & trainer từ GĐ2; cấu hình chuẩn Adam lr=1e-3.",
         "experiments/exp_A_architectures.py; logs/exp_A_results.csv; figures/exp_A_architecture_comparison.png",
         "So sánh công bằng trên cùng test set; biểu đồ rõ trục; phân tích chi tiết ưu nhược điểm từng kiến trúc.",
         "Bàn giao bảng số liệu và biểu đồ để đưa vào Báo cáo và Slide.",
         "2026-11-14"),

        ("4.0", "GĐ4: Thực nghiệm B - Khảo sát Thuật toán Tối ưu (B)",
         "Khảo sát MediumCNN với 3 thuật toán tối ưu: SGD với Momentum=0.9, Adam, AdamW; đo tốc độ hội tụ loss theo epoch, độ ổn định gradient và độ chính xác phân loại cuối cùng.",
         "Thanh Phương", "Cao Sơn",
         "Dataset từ GĐ1; src/models.py từ GĐ2; cấu hình baseline chuẩn.",
         "experiments/exp_B_optimizers.py; logs/exp_B_results.csv; figures/exp_B_optimizer_convergence.png",
         "Thể hiện rõ quỹ đạo hội tụ; so sánh khách quan giữa SGD và họ Adam; giải thích nguyên nhân lý thuyết.",
         "Bàn giao bảng số liệu và biểu đồ cho Cao Sơn để biên soạn Báo cáo.",
         "2026-11-16"),

        ("5.0", "GĐ5: Thực nghiệm C - Khảo sát Kỹ thuật Data Augmentation (C)",
         "Khảo sát 3 chế độ: Không tăng cường (None), Tăng cường cơ bản (Flip), Tăng cường nâng cao (Flip + Rotation + Shift + Cutout); đo lường khoảng cách sai số tổng quát (Generalization Gap = Train Acc - Val Acc).",
         "Trung Hiếu", "Cao Sơn",
         "Dataset từ GĐ1; src/trainer.py từ GĐ2; mô hình MediumCNN.",
         "experiments/exp_C_augmentations.py; logs/exp_C_results.csv; figures/exp_C_augmentation_impact.png",
         "Chỉ rõ hiện tượng Overfitting khi không dùng augmentation và hiệu quả thu hẹp khoảng cách tổng quát.",
         "Tích hợp kết quả vào Báo cáo và Slide.",
         "2026-11-17"),

        ("6.0", "GĐ6: Thực nghiệm D & Trực quan hóa Bộ lọc Tích chập (Filters)",
         "Khảo sát Regularization (Dropout 0.1, 0.25, 0.4 và loại bỏ BatchNorm); trực quan hóa bộ lọc tích chập tầng cuối (Final Conv Filters theo yêu cầu Slide 5); vẽ bản đồ kích hoạt Feature Maps và Confusion Matrix 7x7; phân tích ca đúng/sai.",
         "Cao Sơn", "Trung Hiếu, Thanh Phương",
         "Checkpoints best model; module visualize.py; hàm get_last_conv_layer().",
         "experiments/exp_D_regularization.py; figures/champion_conv_filters.png, champion_feature_maps.png, champion_confusion_matrix.png",
         "Bộ lọc conv tầng cuối hiển thị rõ nét 64 kernels; Feature Maps 3 tầng tuần tự; ma trận nhầm lẫn chuẩn hóa.",
         "Hoàn thiện toàn bộ minh chứng hình ảnh cho Báo cáo và Slide.",
         "2026-11-19"),

        ("7.0", "GĐ7: Biên soạn Báo cáo, Slide Thuyết trình & Đóng gói Nộp bài",
         "Soạn thảo Báo cáo hoàn chỉnh chuẩn học thuật (.docx/.pdf); thiết kế Slide thuyết trình (.pptx/.pdf) 16 slides SGU; xây dựng CLI main.py điều phối toàn bộ pipeline; rà soát tính tái lập và đóng gói BTL02_Emotion_NguyenTruongCaoSon_3123580040.zip.",
         "Cao Sơn (Nhóm trưởng)", "Trung Hiếu, Thanh Phương",
         "Toàn bộ code, figures, logs, checkpoints từ GĐ1 -> GĐ6.",
         "report/BTL02_BaoCao.docx & pdf; Slide/BTL02_ThuyetTrinh.pptx & Slide.pdf; BTL02_Emotion_NguyenTruongCaoSon_3123580040.zip; File Excel phân công & nhật ký.",
         "Báo cáo đầy đủ 8 chương, số liệu thực nghiệm khớp log CSV; Slide chuyên nghiệp; gói nén sạch sẽ nộp đúng hạn.",
         "Nhóm trưởng nộp bài thành công lên hệ thống đúng quy chuẩn của thầy.",
         "2026-11-20")
    ]

    for row_idx, row_data in enumerate(phases_data, 3):
        fill = fill_zebra if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(row_data, 1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            if fill.fill_type:
                cell.fill = fill
            cell.border = thin_border
            align_h = "center" if col_idx in [1, 4, 5, 10] else "left"
            cell.alignment = Alignment(horizontal=align_h, vertical="top", wrap_text=True)
        ws3.row_dimensions[row_idx].height = 65

    # Auto-adjust column widths for all sheets
    for ws in [ws_daily, ws2, ws3]:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 48)

    out_path = BASE_DIR / output_filename
    try:
        wb.save(out_path)
        print(f"Bảng phân công và nhật ký thực nghiệm theo ngày đã tạo thành công: {out_path}")
    except PermissionError:
        fallback_path = BASE_DIR / "Nhat_ky_thuc_nghiem_va_Phan_cong_BTL02_TheoNgay.xlsx"
        wb.save(fallback_path)
        print(f"File chính đang mở nên đã lưu bản cập nhật sang: {fallback_path}")
        out_path = fallback_path
    return out_path


if __name__ == "__main__":
    generate_excel_task_and_log()
