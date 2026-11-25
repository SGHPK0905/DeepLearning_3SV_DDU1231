"""
BTL03: Bộ tạo bảng tính Excel Phân công Nhiệm vụ và Nhật ký Thực nghiệm
Xuất file 'Phan_cong_nhiem_vu_va_Nhat_ky_BTL03.xlsx' gồm 3 Sheet:
1. Lịch sử theo ngày (Timeline)
2. Phân công nhiệm vụ (Tasks)
3. Kết quả thực nghiệm PyTorch (đọc trực tiếp từ logs/benchmark_keras_vs_pytorch.csv)
"""

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .config import BASE_DIR, LOGS_DIR, GROUP_INFO


def generate_excel_task_and_log(output_filename="Phan_cong_nhiem_vu_va_Nhat_ky_BTL03.xlsx"):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    PRIMARY_BLUE = "1B365D"
    HEADER_BLUE = "2A558C"
    ZEBRA_FILL = "F9FBFD"
    WHITE = "FFFFFF"
    BORDER_COLOR = "D0D7DE"
    SUCCESS_GREEN = "E6F4EA"

    font_title = Font(name="Calibri", size=15, bold=True, color=WHITE)
    font_header = Font(name="Calibri", size=10, bold=True, color=WHITE)
    font_body = Font(name="Calibri", size=10, bold=False, color="1C2128")
    font_body_bold = Font(name="Calibri", size=10, bold=True, color="1C2128")

    fill_title = PatternFill(start_color=PRIMARY_BLUE, end_color=PRIMARY_BLUE, fill_type="solid")
    fill_header = PatternFill(start_color=HEADER_BLUE, end_color=HEADER_BLUE, fill_type="solid")
    fill_zebra = PatternFill(start_color=ZEBRA_FILL, end_color=ZEBRA_FILL, fill_type="solid")
    fill_success = PatternFill(start_color=SUCCESS_GREEN, end_color=SUCCESS_GREEN, fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color=BORDER_COLOR),
        right=Side(style='thin', color=BORDER_COLOR),
        top=Side(style='thin', color=BORDER_COLOR),
        bottom=Side(style='thin', color=BORDER_COLOR)
    )

    # =========================================================================
    # SHEET 1: LỊCH SỬ THỰC HIỆN THEO TỪNG NGÀY
    # =========================================================================
    ws_daily = wb.create_sheet(title="Lịch sử theo ngày (Timeline)")
    ws_daily.views.sheetView[0].showGridLines = True

    ws_daily.merge_cells("A1:I1")
    ws_daily["A1"] = "NHẬT KÝ TIẾN TRÌNH THỰC HIỆN ĐỒ ÁN THEO TỪNG NGÀY (DAILY PROJECT LOGBOOK)"
    ws_daily["A1"].font = font_title
    ws_daily["A1"].fill = fill_title
    ws_daily["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_daily.row_dimensions[1].height = 40

    ws_daily.merge_cells("A2:I2")
    ws_daily["A2"] = f"Môn: {GROUP_INFO['subject']} - Đề tài: {GROUP_INFO['project_title']} - GVHD: {GROUP_INFO['instructor']}"
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
    ws_daily.row_dimensions[3].height = 28

    daily_records = [
        (1, "15/11/2026", "GĐ 1: Khảo sát & Phân tích", "Cả nhóm (Sơn, Phương, Hiếu)",
         "Khảo sát toàn bộ 13 file Jupyter Notebook mẫu trong sample/examples của TS. Đỗ Như Tài.",
         "Bản đặc tả 13 notebook Keras 1.x/2.x thuộc 5 nhóm bài toán.",
         "Keras cũ dùng API khác biệt hoàn toàn và bố cục ảnh NHWC.",
         "Quy hoạch kiến trúc PyTorch nn.Module chuẩn NCHW và cấu trúc giống BTL01/BTL02.", "Hoàn thành 100%"),
        (2, "18/11/2026", "GĐ 2: Chuyển đổi MLP & CNN", "Đặng Thanh Phương",
         "Chuyển đổi Getting Started, Iris Bayesian, Func_apxm_MLP, MNIST_MLP, và 3 kiến trúc CNN (MNIST, Gender, Emotion).",
         "Các module src/models/mlp.py, src/models/cnn.py khớp chính xác 1-1 từng tầng Conv/Pool/FC.",
         "GaussianMLP với Sigmoid bị bão hòa khi lr quá lớn (0.05).",
         "Hạ lr=0.01 với batch_size=32, giúp GaussianMLP đạt R² = 99.93%.", "Hoàn thành 100%"),
        (3, "21/11/2026", "GĐ 3: CNN+LSTM, SegNet & Detection", "Nguyễn Trương Cao Sơn",
         "Chuyển đổi MNIST_CNN_LSTM (Batch Folding), SegNet Encoder-Decoder và CarDetectorCNN (FCN Heatmap).",
         "Các module src/models/cnn_lstm.py, segnet.py, obj_detection.py cùng pipeline đọc zip thật.",
         "Tên file labels trong depth_hand.zip đổi cả thư mục và tiền tố (_depth_ -> _labels_).",
         "Áp dụng replace('depth', 'labels') và ánh xạ đủ 7 mức xám Kinect [255,43,85,128,170,212,0].", "Hoàn thành 100%"),
        (4, "23/11/2026", "GĐ 4: DataLoader, Trainer & Utilities", "Phạm Trung Hiếu",
         "Xây dựng src/data.py (nạp MNIST thật, FER-2013, depth_hand.zip, car_data.zip), trainer.py, metrics.py, visualize.py.",
         "Hệ thống Callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau) và 11 biểu đồ 300 DPI.",
         "Thiếu dữ liệu MNIST cục bộ khi chạy offline các bài MLP/CNN/LSTM.",
         "Tải và cache data/mnist.npz (60k/10k ảnh thật) để dùng chung cho code và notebook.", "Hoàn thành 100%"),
        (5, "25/11/2026", "GĐ 5: Chạy 13 Notebooks & Kiểm chứng", "Cả nhóm (Sơn, Phương, Hiếu)",
         "Tạo và thực thi toàn bộ 13 notebook trong converted_pytorch/ qua IPython kernel, chạy lại benchmark dữ liệu thật.",
         "13/13 notebook có sẵn output & biểu đồ inline; log benchmark chuẩn xác trong logs/benchmark_keras_vs_pytorch.csv.",
         "1_MNIST_CNN PCA ban đầu cắt cố định 1000 mẫu vượt quá tập test CPU (800).",
         "Cập nhật n_vis = min(1000, len(x_test)) và kiểm chứng chạy thành công 13/13 notebook.", "Hoàn thành 100%"),
    ]

    for row_idx, rec in enumerate(daily_records, 4):
        fill_curr = fill_zebra if row_idx % 2 == 0 else PatternFill(fill_type=None)
        for col_idx, val in enumerate(rec, 1):
            cell = ws_daily.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            cell.fill = fill_curr
            cell.border = thin_border
            align_h = "center" if col_idx in [1, 2, 9] else "left"
            cell.alignment = Alignment(horizontal=align_h, vertical="center", wrap_text=True)
            if col_idx == 9:
                cell.fill = fill_success
                cell.font = font_body_bold

    # =========================================================================
    # SHEET 2: PHÂN CÔNG NHIỆM VỤ
    # =========================================================================
    ws_tasks = wb.create_sheet(title="Phân công nhiệm vụ (Tasks)")
    ws_tasks.views.sheetView[0].showGridLines = True

    ws_tasks.merge_cells("A1:G1")
    ws_tasks["A1"] = "BẢNG PHÂN CÔNG NHIỆM VỤ VÀ TỶ LỆ ĐÓNG GÓP THÀNH VIÊN — BTL03"
    ws_tasks["A1"].font = font_title
    ws_tasks["A1"].fill = fill_title
    ws_tasks["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_tasks.row_dimensions[1].height = 40

    headers_tasks = ["STT", "Họ và tên thành viên", "Mã số sinh viên (MSSV)", "Vai trò", "Phân công nhiệm vụ chi tiết", "Tỷ lệ đóng góp", "Đánh giá"]
    for col_idx, h in enumerate(headers_tasks, 1):
        cell = ws_tasks.cell(row=3, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws_tasks.row_dimensions[3].height = 28

    for row_idx, mem in enumerate(GROUP_INFO['members'], 4):
        vals = (row_idx - 3, mem['name'], mem['student_id'], mem['role'], mem['tasks'], mem['contribution'], "Xuất sắc")
        for col_idx, val in enumerate(vals, 1):
            cell = ws_tasks.cell(row=row_idx, column=col_idx, value=val)
            cell.font = font_body
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center" if col_idx in [1, 3, 4, 6, 7] else "left", vertical="center", wrap_text=True)
            if col_idx == 6:
                cell.font = font_body_bold
                cell.fill = fill_success

    # =========================================================================
    # SHEET 3: KẾT QUẢ THỰC NGHIỆM THỰC ĐO
    # =========================================================================
    ws_bench = wb.create_sheet(title="Kết quả Benchmark PyTorch")
    ws_bench.views.sheetView[0].showGridLines = True

    ws_bench.merge_cells("A1:G1")
    ws_bench["A1"] = "KẾT QUẢ KIỂM CHỨNG THỰC ĐO CÁC MÔ HÌNH PYTORCH (DỮ LIỆU THẬT)"
    ws_bench["A1"].font = font_title
    ws_bench["A1"].fill = fill_title
    ws_bench["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_bench.row_dimensions[1].height = 40

    csv_path = LOGS_DIR / "benchmark_keras_vs_pytorch.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        headers_b = ["Mô hình", "Số tham số", "Chỉ số đánh giá", "Kết quả (%)", "Val Loss", "Thời gian (s)", "Nguồn dữ liệu"]
        for col_idx, h in enumerate(headers_b, 1):
            cell = ws_bench.cell(row=3, column=col_idx, value=h)
            cell.font = font_header
            cell.fill = fill_header
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
        ws_bench.row_dimensions[3].height = 26

        for r_idx, (_, r) in enumerate(df.iterrows(), 4):
            row_vals = [r['model'], int(r['params']), r['metric_name'], float(r['score_pct']), float(r['val_loss']), float(r['train_time_s']), r['dataset_source']]
            for col_idx, val in enumerate(row_vals, 1):
                cell = ws_bench.cell(row=r_idx, column=col_idx, value=val)
                cell.font = font_body_bold if col_idx == 4 else font_body
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center" if col_idx in [2, 3, 4, 5, 6] else "left", vertical="center")
                if col_idx == 4:
                    cell.fill = fill_success

    for ws in [ws_daily, ws_tasks, ws_bench]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or '')
                if len(val_str) > max_len and '\n' not in val_str:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 14), 48)

    out_file = BASE_DIR / output_filename
    wb.save(out_file)
    print(f"Đã xuất file Excel: {out_file.name}")
    return out_file


if __name__ == "__main__":
    generate_excel_task_and_log()
