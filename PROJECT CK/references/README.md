# Tài liệu Tham khảo Gốc (Reference Papers)

Thư mục lưu trữ toàn văn các bài báo khoa học nền tảng làm cơ sở thiết kế kiến trúc, lựa chọn bộ dữ liệu, độ đo và quy trình 2 giai đoạn của đề tài (**Nhóm 05 – Học sâu DDU1231**).

| STT | File PDF | Tác giả & Năm | Nội dung kế thừa / Vai trò trong đồ án | Mã nguồn gốc / Link |
| :---: | :--- | :--- | :--- | :--- |
| 01 | `01_Vinyals_2015_ShowAndTell.pdf` | O. Vinyals et al. (CVPR 2015) | Kiến trúc nền tảng **Model 1 — Baseline** (`ResNet-50 + LSTM`), trích xuất **Hình 2.1**. | [arXiv:1411.4555](https://arxiv.org/abs/1411.4555) |
| 02 | `02_Xu_2015_ShowAttendAndTell.pdf` | K. Xu et al. (ICML 2015) | Kiến trúc nền tảng **Model 2 — Bahdanau Spatial Attention** (trả lời **RQ1**), trích xuất **Hình 2.2** & phương pháp trực quan hóa Attention Heatmap. | [arXiv:1502.03044](https://arxiv.org/abs/1502.03044) |
| 03 | `03_He_2016_ResNet.pdf` | K. He et al. (CVPR 2016) | Kiến trúc Visual Encoder **ResNet-50** (global 2048-d & spatial 7×7×2048). | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| 04 | `04_Vaswani_2017_AttentionIsAllYouNeed.pdf` | A. Vaswani et al. (NeurIPS 2017) | Cơ chế Multi-Head Self/Cross-Attention cho **Transformer Decoder**, trích xuất **Hình 2.3**. | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 05 | `05_Dosovitskiy_2021_VisionTransformer_ViT.pdf` | A. Dosovitskiy et al. (ICLR 2021) | Kiến trúc Visual Encoder **ViT-B/16** (196 patch tokens) cho **Model 3** (trả lời **RQ2**), trích xuất **Hình 2.3**. | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| 06 | `06_Ngo_2022_MTet_EnViT5.pdf` | C. Ngo et al. (VietAI, 2022) | Kiến trúc mô hình dịch máy **EnViT5 (`VietAI/envit5-translation`)** dùng cho **Giai đoạn 2 (Chuyển ngữ Anh–Việt)**. | [arXiv:2210.05610](https://arxiv.org/abs/2210.05610) |
| 07 | `07_Hodosh_2013_Flickr8k_JAIR.pdf` | M. Hodosh et al. (JAIR 2013) | Bài báo gốc giới thiệu tập dữ liệu **Flickr8k** (8.091 ảnh × 5 captions). | [JAIR Vol. 47](https://www.jair.org/index.php/jair/article/view/10842) |
| 08 | `08_Karpathy_2015_VisualSemanticAlignments.pdf` | A. Karpathy & L. Fei-Fei (CVPR 2015) | Giao thức chia tập chuẩn **Karpathy Split** (6.000 train / 1.000 val / 1.000 test) chống Data Leakage. | [arXiv:1412.2306](https://arxiv.org/abs/1412.2306) |
| 09 | `09_Stefanini_2023_ImageCaptioningSurvey.pdf` | M. Stefanini et al. (IEEE TPAMI 2023) | Tổng quan toàn diện các phương pháp, độ đo (`BLEU`, `METEOR`, `ROUGE-L`, `CIDEr`) và hướng nghiên cứu Image Captioning. | [arXiv:2107.06912](https://arxiv.org/abs/2107.06912) |

> **Lưu ý khi đưa hình vào Báo cáo / Slide:** Khi trích lại sơ đồ kiến trúc từ các file PDF trên (Ví dụ: Hình 2.1, 2.2, 2.3), phải ghi rõ nguồn trích dẫn trong chú thích hình và cập nhật vào Tab `Nguon_Pipeline_HinhAnh` trong [NhatKy_ThucNghiem_CK.xlsx](../NhatKy_ThucNghiem_CK.xlsx).
