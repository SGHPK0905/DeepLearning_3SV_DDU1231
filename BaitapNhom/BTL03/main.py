"""
BTL03: CLI Điều phối Toàn diện Đồ án Chuyển đổi Keras sang PyTorch
Môn học: Học sâu (Deep Learning) — GVHD: TS. Đỗ Như Tài
Nhóm 05: Nguyễn Trương Cao Sơn (NT), Đặng Thanh Phương, Phạm Trung Hiếu

Cách sử dụng:
    python main.py --all
    python main.py --convert_notebooks
    python main.py --execute_notebooks
    python main.py --run_benchmarks
    python main.py --generate_excel
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import GROUP_INFO
from src.convert_all_notebooks import convert_all
from experiments.exp_benchmark_conversion import run_all_benchmarks
from src.generate_excel_log import generate_excel_task_and_log


def step_convert_notebooks(execute: bool = False):
    print("\n" + "=" * 70)
    print("BƯỚC 1: CHUYỂN ĐỔI TOÀN BỘ 13 FILE NOTEBOOK KERAS SANG PYTORCH")
    print("=" * 70)
    convert_all(execute=execute)
    print(">>> Bước 1 hoàn thành thành công!")


def step_run_benchmarks():
    print("\n" + "=" * 70)
    print("BƯỚC 2: THỰC THI CHUỖI THỰC NGHIỆM KIỂM CHỨNG CÁC MÔ HÌNH PYTORCH")
    print("=" * 70)
    run_all_benchmarks()
    print(">>> Bước 2 hoàn thành thành công!")


def step_generate_excel():
    print("\n" + "=" * 70)
    print("BƯỚC 3: TẠO BẢNG TÍNH EXCEL PHÂN CÔNG & NHẬT KÝ THỰC NGHIỆM")
    print("=" * 70)
    generate_excel_task_and_log()
    print(">>> Bước 3 hoàn thành thành công!")


def main():
    parser = argparse.ArgumentParser(description="BTL03: Keras to PyTorch Deep Learning Pipeline")
    parser.add_argument('--convert_notebooks', action='store_true', help="Chuyển đổi 13 notebooks sang PyTorch")
    parser.add_argument('--execute_notebooks', action='store_true', help="Chuyển đổi và chạy toàn bộ 13 notebooks")
    parser.add_argument('--run_benchmarks', action='store_true', help="Chạy chuỗi đo lường thực nghiệm 7 mô hình")
    parser.add_argument('--generate_excel', action='store_true', help="Tạo file Excel phân công và nhật ký")
    parser.add_argument('--all', action='store_true', help="Chạy toàn bộ quy trình tự động từ đầu đến cuối")

    args = parser.parse_args()

    start_time = time.time()
    print("\n" + "#" * 70)
    print(f"HỌC PHẦN: {GROUP_INFO['subject']} — GVHD: {GROUP_INFO['instructor']}")
    print(f"ĐỒ ÁN: {GROUP_INFO['project_title']}")
    print(f"NHÓM THỰC HIỆN: NHÓM 05 — LỚP: {GROUP_INFO['class_name']}")
    print("#" * 70)

    if len(sys.argv) == 1 or args.all:
        step_convert_notebooks(execute=True)
        step_run_benchmarks()
        step_generate_excel()
    else:
        if args.execute_notebooks:
            step_convert_notebooks(execute=True)
        elif args.convert_notebooks:
            step_convert_notebooks(execute=False)
        if args.run_benchmarks:
            step_run_benchmarks()
        if args.generate_excel:
            step_generate_excel()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TOÀN BỘ QUY TRÌNH BTL03 ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
