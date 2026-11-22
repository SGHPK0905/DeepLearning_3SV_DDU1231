"""
BTL02: Main Orchestrator CLI for Facial Emotion Recognition (FER2013)
Unified pipeline for training, evaluation, experiments, report and slide generation, and submission packaging.

Usage:
    python main.py --all
    python main.py --prepare_data
    python main.py --train_champion
    python main.py --run_experiments
    python main.py --generate_reports
    python main.py --pack_zip
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    GROUP_INFO,
    RAW_CSV_PATH,
    TRAIN_NPZ_PATH,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    FIGURES_DIR,
    REPORT_DIR,
    SLIDE_DIR
)
from src.data import extract_and_cache_fer2013, load_data
from src.visualize import plot_dataset_distribution, plot_image_samples
from src.train_champion import train_champion_pipeline
from experiments.exp_A_architectures import run_experiment_A
from experiments.exp_B_optimizers import run_experiment_B
from experiments.exp_C_augmentations import run_experiment_C
from experiments.exp_D_regularization import run_experiment_D


def step_prepare_data():
    print("\n" + "=" * 60)
    print("BƯỚC 1: TIỀN XỬ LÝ & TRỰC QUAN HÓA BỘ DỮ LIỆU FER-2013")
    print("=" * 60)
    extract_and_cache_fer2013()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    plot_dataset_distribution(y_train, y_val, y_test)
    plot_image_samples(x_train[:32], y_train[:32], title='Training Samples (4x8 Grid)', save_name='trainining_images.png')
    plot_image_samples(x_val[:32], y_val[:32], title='Validating Samples (4x8 Grid)', save_name='validating_images.png')
    print("Bước 1 hoàn thành thành công!")


def step_run_experiments():
    print("\n" + "=" * 60)
    print("BƯỚC 2: THỰC THI 4 CHUỖI THỰC NGHIỆM KIỂM SOÁT A / B / C / D")
    print("=" * 60)
    run_experiment_A(epochs=6, batch_size=128, subset_size=6000)
    run_experiment_B(epochs=6, batch_size=128, subset_size=6000)
    run_experiment_C(epochs=6, batch_size=128, subset_size=6000)
    run_experiment_D(epochs=6, batch_size=128, subset_size=6000)
    print("Bước 2 hoàn thành thành công!")


def step_train_champion():
    print("\n" + "=" * 60)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH QUÁN QUÂN & TRỰC QUAN HÓA CHI TIẾT")
    print("=" * 60)
    train_champion_pipeline(epochs=12, batch_size=128, subset_size=12000)
    print("Bước 3 hoàn thành thành công!")


def step_generate_reports():
    print("\n" + "=" * 60)
    print("BƯỚC 4: TẠO BÁO CÁO (WORD/PDF), SLIDE (PPTX/PDF) VÀ NHẬT KÝ EXCEL")
    print("=" * 60)
    from src.generate_excel_log import generate_excel_task_and_log
    from src.generate_word_report import build_word_report
    from src.generate_slide_deck import build_presentation_deck
    from src.export_pdf import export_docx_to_pdf, export_pptx_to_pdf

    generate_excel_task_and_log()
    build_word_report()
    build_presentation_deck()

    # Xuất file PDF bằng Word/PowerPoint COM
    docx_path = REPORT_DIR / "BTL02_BaoCao.docx"
    pdf_report_path = REPORT_DIR / "BTL02_BaoCao.pdf"
    pptx_path = SLIDE_DIR / "BTL02_ThuyetTrinh.pptx"
    pdf_slide_path = SLIDE_DIR / "Slide.pdf"

    export_docx_to_pdf(docx_path, pdf_report_path)
    export_pptx_to_pdf(pptx_path, pdf_slide_path)
    print("Bước 4 hoàn thành thành công!")


def step_pack_zip():
    print("\n" + "=" * 60)
    print("BƯỚC 5: ĐÓNG GÓI SẢN PHẨM NỘP BÀI (NHÓM TRƯỞNG NỘP)")
    print("=" * 60)
    import zipfile
    leader_name = "NguyenTruongCaoSon"
    leader_id = "3123580040"
    zip_filename = f"BTL02_Emotion_{leader_name}_{leader_id}.zip"
    zip_path = Path(zip_filename)

    include_patterns = [
        'report/',
        'Slide/',
        'figures/',
        'logs/',
        'checkpoints/best_model.pt',
        'src/',
        'experiments/',
        'data/preprocessing/',
        'main.py',
        'README.md',
        'requirements.txt',
        'Phan_cong_nhiem_vu_va_Nhat_ky_BTL02.xlsx'
    ]

    print(f"Đang tạo gói nén: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip hidden and cache directories
            if any(p in root for p in ['.git', '__pycache__', '.ipynb_checkpoints', '.system_generated']):
                continue
            for f in files:
                if f.startswith('~$') or f.endswith(('.pyc', '.zip')) or f == 'fer2013.csv':
                    continue
                # Keep only best_model.pt in checkpoints for clean size
                if 'checkpoints' in root and f != 'best_model.pt':
                    continue
                file_path = Path(root) / f
                arcname = file_path.as_posix()
                try:
                    zipf.write(file_path, arcname)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Đóng gói hoàn tất: {zip_path.name} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="BTL02: Facial Emotion Recognition (FER2013) Pipeline")
    parser.add_argument('--prepare_data', action='store_true', help="Preprocess FER2013 and cache npz")
    parser.add_argument('--run_experiments', action='store_true', help="Run controlled experiments A/B/C/D")
    parser.add_argument('--train_champion', action='store_true', help="Train Champion Model and evaluate")
    parser.add_argument('--generate_reports', action='store_true', help="Build Word, Slide, Excel, and PDFs")
    parser.add_argument('--pack_zip', action='store_true', help="Create final submission zip archive")
    parser.add_argument('--all', action='store_true', help="Run entire pipeline from end to end")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.all:
        step_prepare_data()
        step_run_experiments()
        step_train_champion()
        step_generate_reports()
        step_pack_zip()
    else:
        if args.prepare_data:
            step_prepare_data()
        if args.run_experiments:
            step_run_experiments()
        if args.train_champion:
            step_train_champion()
        if args.generate_reports:
            step_generate_reports()
        if args.pack_zip:
            step_pack_zip()


if __name__ == "__main__":
    main()
