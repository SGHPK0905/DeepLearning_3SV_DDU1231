"""
BTL02: Professional Slide Deck (.pptx) Generator
Builds Slide/BTL02_ThuyetTrinh.pptx with 16 slides in 16:9 widescreen,
featuring SGU Navy palette, clear layout cards, embedded high-res figures, and structured bullet points.
"""

import os
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from .config import (
    GROUP_INFO,
    SLIDE_DIR,
    FIGURES_DIR
)


def create_card_shape(slide, left, top, width, height, fill_color, border_color=None):
    """Helper to create a background card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def build_presentation_deck(output_filename="BTL02_ThuyetTrinh.pptx"):
    prs = pptx.Presentation()
    # 16:9 Widescreen: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # Color Palette
    NAVY = RGBColor(27, 54, 93)      # SGU Navy
    STEEL = RGBColor(42, 85, 140)
    DARK = RGBColor(30, 41, 59)
    LIGHT_BG = RGBColor(248, 250, 252)
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_CLR = RGBColor(226, 232, 240)
    ACCENT_RED = RGBColor(225, 29, 72)
    ACCENT_GREEN = RGBColor(16, 185, 129)
    ACCENT_BLUE = RGBColor(14, 165, 233)
    WHITE = RGBColor(255, 255, 255)

    def add_header(slide, title_text, category_text="BÀI TẬP LỚN 02 — HỌC SÂU (DEEP LEARNING)"):
        # Header banner
        create_card_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.15), NAVY)

        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.95))
        tf = tx_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = WHITE

        # SGU Logo
        logo_path = FIGURES_DIR / "sgu_logo.png"
        if logo_path.exists():
            slide.shapes.add_picture(str(logo_path), Inches(12.0), Inches(0.15), height=Inches(0.85))

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    create_card_shape(s1, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)

    # Decorative accent bar
    create_card_shape(s1, Inches(0), Inches(7.2), Inches(13.333), Inches(0.3), ACCENT_BLUE)

    # Logo
    logo_path = FIGURES_DIR / "sgu_logo.png"
    if logo_path.exists():
        s1.shapes.add_picture(str(logo_path), Inches(1.0), Inches(1.0), height=Inches(1.3))

    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11.3), Inches(2.2))
    tf1 = tb_title.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "TRƯỜNG ĐẠI HỌC SÀI GÒN — KHOA CÔNG NGHỆ THÔNG TIN"
    p0.font.name = "Calibri"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE

    p1 = tf1.add_paragraph()
    p1.text = "NHẬN DIỆN CẢM XÚC KHUÔN MẶT BẰNG MẠNG NƠ-RON TÍCH CHẬP (CNN) TRONG PYTORCH"
    p1.font.name = "Calibri"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_before = Pt(8)

    p2 = tf1.add_paragraph()
    p2.text = "Khảo sát và thực nghiệm chuyên sâu trên bộ dữ liệu chuẩn Kaggle FER-2013"
    p2.font.name = "Calibri"
    p2.font.size = Pt(15)
    p2.font.italic = True
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(6)

    # Meta card
    create_card_shape(s1, Inches(1.0), Inches(4.9), Inches(11.333), Inches(1.8), RGBColor(15, 23, 42))
    tb_meta = s1.shapes.add_textbox(Inches(1.3), Inches(5.0), Inches(10.7), Inches(1.6))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True

    pm1 = tf_meta.paragraphs[0]
    pm1.text = "Giảng viên hướng dẫn: TS. ĐỖ NHƯ TÀI   |   Học phần: Học sâu (Deep Learning) - Lớp DDU1231"
    pm1.font.name = "Calibri"
    pm1.font.size = Pt(13)
    pm1.font.bold = True
    pm1.font.color.rgb = ACCENT_BLUE

    pm2 = tf_meta.add_paragraph()
    pm2.text = "Thành viên nhóm 05 thực hiện: (Đóng góp: 100% - 100% - 100%)"
    pm2.font.name = "Calibri"
    pm2.font.size = Pt(12)
    pm2.font.bold = True
    pm2.font.color.rgb = WHITE
    pm2.space_before = Pt(4)

    pm3 = tf_meta.add_paragraph()
    pm3.text = "• Nguyễn Trương Cao Sơn (3123580040) — Nhóm trưởng   • Đặng Thanh Phương (3123580036)   • Phạm Trung Hiếu (3123580014)"
    pm3.font.name = "Calibri"
    pm3.font.size = Pt(12)
    pm3.font.color.rgb = RGBColor(241, 245, 249)
    pm3.space_before = Pt(2)

    # =========================================================================
    # SLIDE 2: PHÂN CÔNG NHIỆM VỤ & ĐÓNG GÓP THÀNH VIÊN
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Bảng Phân Công Nhiệm Vụ & Tỷ Lệ Đóng Góp Thành Viên")

    # 3 Member Cards
    card_w = Inches(3.64)
    card_h = Inches(5.6)
    card_top = Inches(1.45)
    members_info = [
        ("Nguyễn Trương Cao Sơn", "3123580040", "Nhóm trưởng", "100%", ACCENT_BLUE, [
            "Chủ trì thiết kế kiến trúc CNN VGG16_FER & ResNet_FER.",
            "Điều phối và thực thi 4 chuỗi thực nghiệm kiểm soát A/B/C/D.",
            "Lập trình module trích xuất bộ lọc Conv tầng cuối (Slide 5) & Feature Maps.",
            "Chủ trì biên soạn Báo cáo học thuật (.docx/.pdf) và Slide (.pptx/.pdf).",
            "Đóng gói sản phẩm nộp bài chuẩn quy cách BTL02."
        ]),
        ("Đặng Thanh Phương", "3123580036", "Thành viên", "100%", ACCENT_GREEN, [
            "Thiết kế kiến trúc mạng SmallCNN và MediumCNN (theo gợi ý của Thầy).",
            "Xây dựng bộ khung huấn luyện PyTorch chuẩn (src/trainer.py).",
            "Thiết lập cơ chế Early Stopping, LR Scheduler và Gradient Clipping.",
            "Thực hiện đo đạc và phân tích thực nghiệm B (Optimizers) & thực nghiệm D.",
            "Đồng biên soạn nội dung Báo cáo Word chương 3 và 5."
        ]),
        ("Phạm Trung Hiếu", "3123580014", "Thành viên", "100%", RGBColor(245, 158, 11), [
            "Xây dựng pipeline nạp dữ liệu fer2013.csv và đóng gói cache .npz.",
            "Lập trình module Data Augmentation (Flip, Rotate, Shift, Cutout).",
            "Xây dựng module tính toán bộ chỉ số đánh giá đa lớp (src/metrics.py).",
            "Phụ trách thực nghiệm C (Data Augmentation) và khảo sát Generalization Gap.",
            "Xây dựng Nhật ký thực nghiệm chi tiết trong file Excel."
        ])
    ]

    for idx, (name, sid, role, contrib, clr, tasks) in enumerate(members_info):
        c_left = Inches(0.8 + idx * 4.0)
        create_card_shape(s2, c_left, card_top, card_w, card_h, CARD_BG, BORDER_CLR)
        # Top banner for card
        create_card_shape(s2, c_left, card_top, card_w, Inches(1.0), clr)

        tb = s2.shapes.add_textbox(c_left + Inches(0.2), card_top + Inches(0.1), card_w - Inches(0.4), Inches(0.8))
        tf = tb.text_frame
        tf.word_wrap = True
        p_n = tf.paragraphs[0]
        p_n.text = f"{name} ({role})"
        p_n.font.name = "Calibri"
        p_n.font.size = Pt(13)
        p_n.font.bold = True
        p_n.font.color.rgb = WHITE

        p_s = tf.add_paragraph()
        p_s.text = f"MSSV: {sid}   |   Đóng góp: {contrib}"
        p_s.font.name = "Calibri"
        p_s.font.size = Pt(11)
        p_s.font.bold = True
        p_s.font.color.rgb = RGBColor(241, 245, 249)

        # Tasks list
        tb_t = s2.shapes.add_textbox(c_left + Inches(0.2), card_top + Inches(1.15), card_w - Inches(0.4), Inches(4.2))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True

        p_th = tf_t.paragraphs[0]
        p_th.text = "Trách nhiệm chính:"
        p_th.font.name = "Calibri"
        p_th.font.size = Pt(11)
        p_th.font.bold = True
        p_th.font.color.rgb = DARK

        for task in tasks:
            pt = tf_t.add_paragraph()
            pt.text = f"• {task}"
            pt.font.name = "Calibri"
            pt.font.size = Pt(10)
            pt.font.color.rgb = RGBColor(51, 65, 85)
            pt.space_before = Pt(4)

    # =========================================================================
    # SLIDE 3: BÀI TOÁN FER2013 & THÁCH THỨC
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Bài Toán Nhận Diện Cảm Xúc Khuôn Mặt & Thách Thức FER-2013")

    # Left card: Problem Specs
    create_card_shape(s3, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_prob = s3.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(5.2))
    tf_p = tb_prob.text_frame
    tf_p.word_wrap = True

    p = tf_p.paragraphs[0]
    p.text = "1. Đặc tả bài toán (Problem Specification)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    items_prob = [
        ("Input:", "Ảnh chân dung khuôn mặt kích thước 48x48 pixel mức xám (1 kênh)."),
        ("Output:", "Vector 7 chiều tương ứng phân phối xác suất của 7 lớp cảm xúc chuẩn:"),
        ("", "0: Angry (Tức giận)       4: Sad (Buồn bã)\n1: Disgust (Ghê tởm)     5: Surprise (Ngạc nhiên)\n2: Fear (Sợ hãi)            6: Neutral (Trung tính)\n3: Happy (Vui vẻ)"),
        ("Mục tiêu:", "Tối đa hóa độ chính xác tổng quát hóa trên tập PrivateTest (tiệm cận mức kỷ lục thế giới 69% - 71% như Slide 3).")
    ]
    for k, v in items_prob:
        p = tf_p.add_paragraph()
        p.text = f"{k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    p_c = tf_p.add_paragraph()
    p_c.text = "2. Những thách thức cốt lõi (Core Challenges)"
    p_c.font.size = Pt(14)
    p_c.font.bold = True
    p_c.font.color.rgb = NAVY
    p_c.space_before = Pt(12)

    challenges = [
        "Mất cân bằng lớp trầm trọng: Lớp Happy có 8,989 mẫu trong khi Disgust chỉ có 547 mẫu (chênh lệch hơn 16 lần).",
        "Độ phân giải siêu nhỏ (48x48): Rất ít thông tin điểm ảnh, dễ tiêu biến đặc trưng nếu lạm dụng pooling.",
        "Biểu cảm mơ hồ (Ambiguity): Ranh giới giữa Fear vs Surprise, Angry vs Sad rất mong manh."
    ]
    for ch in challenges:
        p = tf_p.add_paragraph()
        p.text = f"• {ch}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = RGBColor(71, 85, 105)
        p.space_before = Pt(4)

    # Right Card: Distribution Histogram
    create_card_shape(s3, Inches(6.7), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    dist_img = FIGURES_DIR / "dataset_distribution_histogram.png"
    if dist_img.exists():
        s3.shapes.add_picture(str(dist_img), Inches(6.9), Inches(1.8), width=Inches(5.4))

    tb_cap = s3.shapes.add_textbox(Inches(6.9), Inches(6.5), Inches(5.4), Inches(0.4))
    p = tb_cap.text_frame.paragraphs[0]
    p.text = "Hình 1: Phân bố số lượng mẫu 7 lớp trên Training, PublicTest và PrivateTest"
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = RGBColor(100, 116, 139)
    p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 4: TIỀN XỬ LÝ & DATA AUGMENTATION
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Quy Trình Tiền Xử Lý Dữ Liệu & Data Augmentation")

    create_card_shape(s4, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.6), CARD_BG, BORDER_CLR)
    train_img = FIGURES_DIR / "trainining_images.png"
    if train_img.exists():
        s4.shapes.add_picture(str(train_img), Inches(0.9), Inches(1.6), width=Inches(5.4))

    tb_tcap = s4.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(5.4), Inches(0.4))
    p = tb_tcap.text_frame.paragraphs[0]
    p.text = "Hình 2: Trực quan hóa 32 mẫu ảnh huấn luyện FER-2013 thực tế"
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = RGBColor(100, 116, 139)
    p.alignment = PP_ALIGN.CENTER

    create_card_shape(s4, Inches(6.7), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_aug = s4.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.2), Inches(5.2))
    tf_aug = tb_aug.text_frame
    tf_aug.word_wrap = True

    p = tf_aug.paragraphs[0]
    p.text = "Chiến lược Tiền xử lý & Tăng cường dữ liệu:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    aug_steps = [
        ("Tách phân vùng độc lập:", "Trích xuất từ fer2013.csv thành train.npz (28,709), valid.npz (3,589) và test.npz (3,589) để nạp tức thì mà không rò rỉ dữ liệu."),
        ("Chuẩn hóa dải giá trị:", "Quy đổi pixel uint8 [0, 255] về tensor [-1.0, 1.0] bằng công thức: x_norm = (x / 255.0 - 0.5) / 0.5."),
        ("Lật ngang ngẫu nhiên (p=0.5):", "Tận dụng tính đối xứng tự nhiên của khuôn mặt người để nhân đôi không gian đặc trưng."),
        ("Xoay ngẫu nhiên [-10°, +10°]:", "Khớp với tham số rotation_range của Thầy trong bài mẫu, mô phỏng các góc nghiêng đầu thực tế."),
        ("Dịch chuyển ngẫu nhiên [-4px, +4px]:", "Mô phỏng sai lệch căn chỉnh vị trí khuôn mặt trong khung hình (Shift Range)."),
        ("Cơ chế Cutout (Random Erasing 8x8):", "Che ngẫu nhiên một vùng nhỏ, ép mạng phải học đặc trưng toàn cục thay vì phụ thuộc vào một điểm đơn lẻ.")
    ]
    for title, desc in aug_steps:
        p = tf_aug.add_paragraph()
        p.text = f"• {title} {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = DARK
        p.space_before = Pt(5)

    # =========================================================================
    # SLIDE 5: KIẾN TRÚC CÁC MÔ HÌNH CNN ĐỀ XUẤT
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Thiết Kế Các Họ Kiến Trúc CNN Khảo Sát Trong PyTorch")

    arch_cards = [
        ("SmallCNN (2 Blocks)", "4.8M Params", ACCENT_BLUE, [
            "Block 1: Conv(1->32) + Conv(32->64) + MaxPool(2x2) + Dropout(0.25) -> 24x24",
            "Block 2: Conv(64->128) + Conv(128->128) + MaxPool(2x2) + Dropout(0.25) -> 12x12",
            "Classifier: Flatten -> Dense(512) -> BN -> ReLU -> Dropout(0.5) -> Dense(7)",
            "Ưu điểm: Tốc độ train nhanh (8.5s/epoch). Hạn chế: Khó nắm bắt vi biểu cảm."
        ]),
        ("MediumCNN (3 Blocks - Đề xuất của Thầy)", "13.7M Params", ACCENT_GREEN, [
            "Block 1: 2 Conv(64, 3x3) + MaxPool(2x2) + Dropout(0.25) -> 24x24",
            "Block 2: 2 Conv(128, 3x3) + MaxPool(2x2) + Dropout(0.25) -> 12x12",
            "Block 3: 2 Conv(256, 3x3) + MaxPool(2x2) + Dropout(0.25) -> 6x6",
            "Classifier: Flatten -> Dense(1024) -> Dense(256) -> Dense(7, Softmax)",
            "Đặc điểm: Cân bằng lý tưởng giữa số tham số và độ sâu biểu diễn."
        ]),
        ("VGG16_FER (4 Blocks)", "18.5M Params", RGBColor(147, 51, 234), [
            "Gồm 4 giai đoạn tích chập sâu (64 -> 128 -> 256 -> 512 kênh).",
            "Sử dụng 4 tầng MaxPooling liên tiếp, giảm không gian 48x48 xuống 3x3.",
            "Classifier: Dense(1024) -> Dense(512) -> Dense(7)",
            "Hiệu năng: Đạt Test Acc cao nhưng tiêu tốn tài nguyên tính toán (21.6s/epoch)."
        ]),
        ("ResNet_FER (Residual Blocks)", "11.2M Params", RGBColor(234, 88, 12), [
            "4 khối Residual Blocks với kết nối tắt x_out = F(x) + x chống tiêu biến gradient.",
            "Thay thế tầng FC cồng kềnh bằng Global Average Pooling (GAP 1x1).",
            "Số tham số gọn gàng (11.2M), truyền dẫn gradient cực kỳ ổn định.",
            "Kiểm chứng: Với ảnh 48x48, các khối tích chập chuẩn phẳng vẫn chiếm ưu thế."
        ])
    ]

    for i, (title, params, col_clr, details) in enumerate(arch_cards):
        left_pos = Inches(0.8 + i * 2.95)
        create_card_shape(s5, left_pos, Inches(1.45), Inches(2.8), Inches(5.6), CARD_BG, BORDER_CLR)
        create_card_shape(s5, left_pos, Inches(1.45), Inches(2.8), Inches(0.8), col_clr)

        tb = s5.shapes.add_textbox(left_pos + Inches(0.1), Inches(1.5), Inches(2.6), Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE

        p2 = tb.text_frame.add_paragraph()
        p2.text = params
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = RGBColor(241, 245, 249)

        tb_body = s5.shapes.add_textbox(left_pos + Inches(0.15), Inches(2.35), Inches(2.5), Inches(4.5))
        tf_b = tb_body.text_frame
        tf_b.word_wrap = True

        for dt in details:
            p = tf_b.add_paragraph()
            p.text = f"• {dt}"
            p.font.size = Pt(9.5)
            p.font.color.rgb = DARK
            p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 6: THỰC NGHIỆM A — SO SÁNH KIẾN TRÚC
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Thực Nghiệm A: Khảo Sát & So Sánh 4 Họ Kiến Trúc CNN")

    create_card_shape(s6, Inches(0.8), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    expA_img = FIGURES_DIR / "exp_A_architecture_comparison.png"
    if expA_img.exists():
        s6.shapes.add_picture(str(expA_img), Inches(0.9), Inches(1.8), width=Inches(5.6))

    tb_capA = s6.shapes.add_textbox(Inches(0.9), Inches(6.5), Inches(5.6), Inches(0.4))
    p = tb_capA.text_frame.paragraphs[0]
    p.text = "Hình 3: Độ chính xác Test Acc (%) và Thời gian huấn luyện mỗi epoch"
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = RGBColor(100, 116, 139)
    p.alignment = PP_ALIGN.CENTER

    create_card_shape(s6, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_txtA = s6.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(5.2))
    tf_A = tb_txtA.text_frame
    tf_A.word_wrap = True

    p = tf_A.paragraphs[0]
    p.text = "Bảng kết quả và Kết luận Thực nghiệm A:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    conclusions_A = [
        ("SmallCNN:", "Test Acc: 55.45% | Time: 8.5s/epoch. Mạng quá nông, trường tiếp nhận chưa đủ bao quát toàn bộ gương mặt, giới hạn ở mức 55%."),
        ("MediumCNN:", "Test Acc: 60.88% | Time: 14.2s/epoch. Đạt độ chính xác cao vượt trội (+5.4%), duy trì được độ phân giải 6x6 trước classifier."),
        ("VGG16_FER:", "Test Acc: 62.45% | Time: 21.6s/epoch. Cho độ chính xác đỉnh nhất nhóm A, nhưng thời gian huấn luyện dài hơn 1.5 lần."),
        ("ResNet_FER:", "Test Acc: 59.35% | Time: 17.8s/epoch. Khối GAP làm mất đi một số thông tin tọa độ không gian nhỏ (Spatial Information) cần thiết cho biểu cảm 48x48."),
        ("Lựa chọn tối ưu:", "MediumCNN và VGG16_FER là 2 ứng viên xuất sắc nhất để xây dựng Mô hình Quán quân.")
    ]
    for k, v in conclusions_A:
        p = tf_A.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 7: THỰC NGHIỆM B — TỐI ƯU HÓA (OPTIMIZERS)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Thực Nghiệm B: Khảo Sát Thuật Toán Tối Ưu (SGD vs Adam vs AdamW)")

    create_card_shape(s7, Inches(0.8), Inches(1.45), Inches(6.0), Inches(5.6), CARD_BG, BORDER_CLR)
    expB_img = FIGURES_DIR / "exp_B_optimizer_convergence.png"
    if expB_img.exists():
        s7.shapes.add_picture(str(expB_img), Inches(0.9), Inches(1.8), width=Inches(5.8))

    create_card_shape(s7, Inches(7.0), Inches(1.45), Inches(5.5), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_txtB = s7.shapes.add_textbox(Inches(7.2), Inches(1.6), Inches(5.1), Inches(5.2))
    tf_B = tb_txtB.text_frame
    tf_B.word_wrap = True

    p = tf_B.paragraphs[0]
    p.text = "Phân tích Quỹ đạo Hội tụ của các Optimizer:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    conclusions_B = [
        ("SGD với Momentum (0.9):", "Hội tụ chậm hơn trong 3 epoch đầu do cần thời gian tích lũy động lượng. Test Acc đạt 56.80%."),
        ("Adam (lr=1e-3):", "Hội tụ thần tốc ngay từ epoch 1 nhờ ước lượng moment bậc 1 và bậc 2 riêng biệt cho từng trọng số. Đạt Test Acc 60.88%."),
        ("AdamW (lr=1e-3, wd=1e-3):", "Đạt Test Acc cao nhất: 61.70% (+0.82% so với Adam). Tách rời Weight Decay khỏi bước tính moment giúp các trọng số tích chập không bị co cụm bất thường."),
        ("Chiến lược Learning Rate:", "Kết hợp ReduceLROnPlateau (factor=0.5, patience=2) giúp hạ tốc độ học khi loss đi vào thung lũng hẹp, tránh hiện tượng dao động quanh cực tiểu.")
    ]
    for k, v in conclusions_B:
        p = tf_B.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 8: THỰC NGHIỆM C — DATA AUGMENTATION
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Thực Nghiệm C: Khảo Sát Tác Động Thu Hẹp Generalization Gap")

    create_card_shape(s8, Inches(0.8), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    expC_img = FIGURES_DIR / "exp_C_augmentation_impact.png"
    if expC_img.exists():
        s8.shapes.add_picture(str(expC_img), Inches(0.9), Inches(1.8), width=Inches(5.6))

    create_card_shape(s8, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_txtC = s8.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(5.2))
    tf_C = tb_txtC.text_frame
    tf_C.word_wrap = True

    p = tf_C.paragraphs[0]
    p.text = "Chứng minh Thực nghiệm về Chống Overfitting:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    conclusions_C = [
        ("Không Augmentation (None):", "Train Acc vọt lên 74.2% nhưng Val Acc chỉ 56.4% -> Khoảng cách Gap khổng lồ: 17.80%. Mô hình bị quá khớp cục bộ."),
        ("Tăng cường Cơ bản (Flip):", "Khoảng cách Gap giảm xuống 8.15%, Test Acc tăng lên 60.88% (+5.08%)."),
        ("Tăng cường Nâng cao (Advanced):", "Gap co lại chỉ còn 5.00%, trong khi Test Acc chạm mốc 62.90% (+7.1% so với không tăng cường)."),
        ("Ý nghĩa lý thuyết:", "Các phép biến đổi ngẫu nhiên đóng vai trò như một bộ điều hòa không gian (Spatial Regularizer), ngăn chặn các nơ-ron học vẹt các chi tiết phông nền.")
    ]
    for k, v in conclusions_C:
        p = tf_C.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 9: THỰC NGHIỆM D — REGULARIZATION & BATCHNORM ABLATION
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Thực Nghiệm D: Khảo Sát Regularization & Triệt Tiêu BatchNorm")

    create_card_shape(s9, Inches(0.8), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    expD_img = FIGURES_DIR / "exp_D_regularization_ablation.png"
    if expD_img.exists():
        s9.shapes.add_picture(str(expD_img), Inches(0.9), Inches(1.8), width=Inches(5.6))

    create_card_shape(s9, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_txtD = s9.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(5.2))
    tf_D = tb_txtD.text_frame
    tf_D.word_wrap = True

    p = tf_D.paragraphs[0]
    p.text = "Kết quả Nghiên Cứu Triệt Tiêu (Ablation Study):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    conclusions_D = [
        ("Dropout=0.10 (Quá thấp):", "Test Acc: 58.40%. Các liên kết nơ-ron vẫn xảy ra hiện tượng co-adaptation (đồng thích nghi dư thừa)."),
        ("Dropout=0.25 (Tiêu chuẩn):", "Test Acc: 60.88%. Tỷ lệ tối ưu nhất bảo toàn thông tin biểu cảm khuôn mặt."),
        ("Dropout=0.40 (Quá cao):", "Test Acc: 57.30%. Gây thiếu khớp (Underfitting), làm chậm quá trình hội tụ của mạng."),
        ("Ablation: Loại bỏ BatchNorm:", "Test Acc sụt giảm nghiêm trọng xuống 51.40% (-9.48%). Khẳng định Batch Normalization là thành phần 'xương sống' để ổn định dòng phân phối kích hoạt qua các tầng tích chập sâu.")
    ]
    for k, v in conclusions_D:
        p = tf_D.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 10: MÔ HÌNH QUÁN QUÂN & ĐƯỜNG CONG HỌC TẬP
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Mô Hình Quán Quân (Champion Model) — Đường Cong Học Tập")

    create_card_shape(s10, Inches(0.8), Inches(1.45), Inches(6.2), Inches(5.6), CARD_BG, BORDER_CLR)
    champ_lc = FIGURES_DIR / "champion_learning_curves.png"
    if champ_lc.exists():
        s10.shapes.add_picture(str(champ_lc), Inches(0.9), Inches(1.8), width=Inches(6.0))

    create_card_shape(s10, Inches(7.2), Inches(1.45), Inches(5.3), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_c_txt = s10.shapes.add_textbox(Inches(7.4), Inches(1.6), Inches(4.9), Inches(5.2))
    tf_c = tb_c_txt.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Cấu hình & Thành tích Mô hình Quán quân:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    c_points = [
        ("Cấu hình hội tụ:", "MediumCNN + AdamW (lr=0.001) + Advanced Augmentation + ReduceLROnPlateau."),
        ("Validation Accuracy:", "Đạt đỉnh 66.45% tại epoch 11."),
        ("PrivateTest Accuracy:", "Đạt 65.85% trên 3,589 ảnh kiểm thử riêng tư hoàn toàn độc lập."),
        ("Macro F1-Score:", "Đạt 63.70% thể hiện sự cân bằng tốt trên toàn bộ 7 lớp cảm xúc."),
        ("So sánh thế giới:", "Kết quả 65.85% tiệm cận rất sát với mốc kỷ lục tốt nhất thế giới trên Kaggle (69% public, 71% private như nêu trong Slide 3).")
    ]
    for k, v in c_points:
        p = tf_c.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 11: MA TRẬN NHẦM LẪN 7X7
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Đánh Giá Chi Tiết Bằng Ma Trận Nhầm Lẫn (Confusion Matrix)")

    create_card_shape(s11, Inches(0.8), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    cm_img = FIGURES_DIR / "champion_confusion_matrix.png"
    if cm_img.exists():
        s11.shapes.add_picture(str(cm_img), Inches(0.9), Inches(1.5), height=Inches(5.4))

    create_card_shape(s11, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_cm_txt = s11.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(5.2))
    tf_cm = tb_cm_txt.text_frame
    tf_cm.word_wrap = True

    p = tf_cm.paragraphs[0]
    p.text = "Phân tích Độ chính xác Từng Lớp Cảm Xúc:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    emotions_eval = [
        ("Happy (Vui vẻ):", "87.2% chính xác. Lớp dễ nhận diện nhất do đặc trưng nụ cười và khóe môi mở rộng rõ nét."),
        ("Surprise (Ngạc nhiên):", "76.5% chính xác. Đặc trưng mắt mở to và miệng tròn chữ O tạo biên cạnh sắc sảo."),
        ("Neutral (Trung tính):", "64.8% chính xác. Nhận diện vững chắc các trạng thái bình thản."),
        ("Angry (Tức giận) vs Sad (Buồn):", "Nhầm lẫn qua lại ~16.5% do cùng có đặc điểm khóe môi hạ thấp và chau mày."),
        ("Fear (Sợ hãi):", "43.8% chính xác, thường bị nhầm sang Sad (18.2%) do các biến dạng cơ mặt phức tạp."),
        ("Disgust (Ghê tởm):", "48.2% chính xác, lớp khó nhất do kích thước mẫu quá ít (547 mẫu), hay nhầm sang Angry do nét nhăn mũi tương đồng.")
    ]
    for k, v in emotions_eval:
        p = tf_cm.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = DARK
        p.space_before = Pt(4)

    # =========================================================================
    # SLIDE 12: ĐÁNH GIÁ ĐỊNH TÍNH: TRUE CASES VS WRONG CASES
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Đánh Giá Định Tính: Phân Tích Ca Đoán Đúng & Ca Đoán Sai")

    create_card_shape(s12, Inches(0.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tc_img = FIGURES_DIR / "champion_true_cases.png"
    if tc_img.exists():
        s12.shapes.add_picture(str(tc_img), Inches(0.9), Inches(1.5), width=Inches(5.5))

    create_card_shape(s12, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    wc_img = FIGURES_DIR / "champion_wrong_cases.png"
    if wc_img.exists():
        s12.shapes.add_picture(str(wc_img), Inches(6.9), Inches(1.5), width=Inches(5.5))

    # =========================================================================
    # SLIDE 13: BỘ LỌC TÍCH CHẬP TẦNG CUỐI (YÊU CẦU SLIDE 5)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Trực Quan Hóa Bộ Lọc Tích Chập Tầng Cuối (Convolution Filters)")

    create_card_shape(s13, Inches(0.8), Inches(1.45), Inches(5.8), Inches(5.6), CARD_BG, BORDER_CLR)
    filt_img = FIGURES_DIR / "champion_conv_filters.png"
    if filt_img.exists():
        s13.shapes.add_picture(str(filt_img), Inches(1.0), Inches(1.5), height=Inches(5.4))

    create_card_shape(s13, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_f_txt = s13.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(5.2))
    tf_f = tb_f_txt.text_frame
    tf_f.word_wrap = True

    p = tf_f.paragraphs[0]
    p.text = "Đáp ứng Yêu Cầu của Thầy tại Slide 5:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = NAVY

    filter_analysis = [
        ("Trực quan hóa bộ lọc:", "Trích xuất toàn bộ 64 kernel tích chập kích thước 3x3 tại tầng Conv cuối cùng của Block 3."),
        ("Bản chất hình học:", "Các hạt nhân hiển thị các cấu trúc tần số không gian cao (vệt sáng tối xen kẽ, các góc bo tròn)."),
        ("Đầu dò biểu cảm (Feature Detectors):", "Bộ lọc tầng cuối không còn nhìn các pixel đơn lẻ mà phản ứng mạnh mẽ với các tổ hợp đặc trưng phức tạp: nếp nhăn trán, độ cong của rãnh cười và khóe mắt nhíu lại."),
        ("Tính đa dạng:", "64 bộ lọc thể hiện tính trực giao cao (Orthogonality), không có hạt nhân nào bị chết (dead filter), minh chứng hiệu quả điều hòa của Dropout 0.25.")
    ]
    for k, v in filter_analysis:
        p = tf_f.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK
        p.space_before = Pt(6)

    # =========================================================================
    # SLIDE 14: BẢN ĐỒ KÍCH HOẠT FEATURE MAPS
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Bản Đồ Kích Hoạt Đặc Trưng (Feature Activation Maps)")

    create_card_shape(s14, Inches(0.8), Inches(1.45), Inches(11.733), Inches(5.6), CARD_BG, BORDER_CLR)
    fmap_img = FIGURES_DIR / "champion_feature_maps.png"
    if fmap_img.exists():
        s14.shapes.add_picture(str(fmap_img), Inches(1.0), Inches(1.8), width=Inches(11.3))

    tb_fmap = s14.shapes.add_textbox(Inches(1.0), Inches(5.8), Inches(11.3), Inches(1.1))
    tf_fm = tb_fmap.text_frame
    tf_fm.word_wrap = True
    p = tf_fm.paragraphs[0]
    p.text = "Cơ chế phân tầng biểu diễn: Block 1 trích xuất đường viền biên cạnh (Edges); Block 2 kết hợp thành các bộ phận khuôn mặt (Eyes, Lips); Block 3 trừu tượng hóa thành biểu cảm tổng thể."
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 15: PHÂN TÍCH CA THẤT BẠI KHI FINE-TUNE VGG16 (5 POOLING)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Phân Tích Ca Thất Bại: Vì Sao Fine-Tune VGG16 Bị Suy Thoái?")

    create_card_shape(s15, Inches(0.8), Inches(1.45), Inches(11.733), Inches(5.6), CARD_BG, BORDER_CLR)
    tb_fail = s15.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11.0), Inches(5.0))
    tf_fail = tb_fail.text_frame
    tf_fail.word_wrap = True

    p = tf_fail.paragraphs[0]
    p.text = "Giải Thích Câu Hỏi trong Báo Cáo Mẫu của Thầy (Mục 5.1 demo_report.pdf):"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ACCENT_RED

    fail_points = [
        ("Hiện tượng Thầy đã nêu:", "'Because the network is too deep with 5 MaxPooling. So, the images should be large and the data needs to be large for training.'"),
        ("1. Tiêu biến kích thước không gian (Extreme Spatial Collapse):", "VGG-16 gốc thiết kế cho ảnh ImageNet 224x224. Với ảnh 48x48 của FER-2013, qua 5 tầng MaxPooling: 48 -> 24 -> 12 -> 6 -> 3 -> 1x1 pixel! Bản đồ đặc trưng bị nén về 1 điểm duy nhất, làm mất toàn bộ cấu trúc hình học tương đối giữa mắt, mũi, miệng."),
        ("2. Hiện tượng Domain Mismatch nghiêm trọng:", "Trọng số tiền huấn luyện ImageNet được học trên 1.2M ảnh màu tự nhiên (RGB), trong khi FER-2013 là ảnh xám 1 kênh độ phân giải thấp và nhiều nhiễu hạt."),
        ("3. Kết luận kiến trúc thực tế:", "Mô hình thiết kế riêng với 3 tầng pooling (như MediumCNN) giữ lại bản đồ 6x6 pixel là kiến trúc tối ưu nhất cho bài toán FER mức xám 48x48.")
    ]
    for k, v in fail_points:
        p = tf_fail.add_paragraph()
        p.text = f"• {k} {v}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = DARK
        p.space_before = Pt(8)

    # =========================================================================
    # SLIDE 16: TỔNG KẾT & KẾT LUẬN
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    create_card_shape(s16, Inches(0), Inches(0), Inches(13.333), Inches(7.5), NAVY)

    tb_end = s16.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5))
    tf_end = tb_end.text_frame
    tf_end.word_wrap = True

    p = tf_end.paragraphs[0]
    p.text = "TỔNG KẾT BÀI TẬP LỚN 02"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = ACCENT_BLUE

    p = tf_end.add_paragraph()
    p.text = "1. Xây dựng hoàn chỉnh pipeline dữ liệu FER-2013 bằng PyTorch thuần."
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.space_before = Pt(12)

    p = tf_end.add_paragraph()
    p.text = "2. Thực thi 4 chuỗi thực nghiệm A/B/C/D kiểm soát nghiêm ngặt, minh chứng tác động của kiến trúc, tối ưu và data augmentation."
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.space_before = Pt(8)

    p = tf_end.add_paragraph()
    p.text = "3. Huấn luyện Mô hình Quán quân đạt 65.85% Test Acc, tiệm cận kỷ lục thế giới (69-71%)."
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.space_before = Pt(8)

    p = tf_end.add_paragraph()
    p.text = "4. Trực quan hóa đầy đủ bộ lọc tích chập tầng cuối (Slide 5), feature maps, confusion matrix và phân tích lỗi."
    p.font.size = Pt(14)
    p.font.color.rgb = WHITE
    p.space_before = Pt(8)

    p_thanks = tf_end.add_paragraph()
    p_thanks.text = "NHÓM 05 XIN CHÂN THÀNH CẢM ƠN THẦY TS. ĐỖ NHƯ TÀI & CÁC BẠN!"
    p_thanks.font.size = Pt(18)
    p_thanks.font.bold = True
    p_thanks.font.color.rgb = RGBColor(250, 204, 21)
    p_thanks.space_before = Pt(24)

    out_path = SLIDE_DIR / output_filename
    prs.save(out_path)
    print(f"Slide thuyết trình PowerPoint đã tạo thành công: {out_path}")
    return out_path


if __name__ == "__main__":
    build_presentation_deck()
