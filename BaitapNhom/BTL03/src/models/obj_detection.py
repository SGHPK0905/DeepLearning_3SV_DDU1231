"""
BTL03: Chuyển đổi mô hình phát hiện xe ô tô (Car Object Detection) sang PyTorch
Kế thừa chính xác cấu trúc từ sample/examples/5. Others/2_obj_detection.ipynb:
    Input (x in [0, 255]) -> Lambda(x / 127.5 - 1.)
    -> Conv2d(3, 10, 3, same) -> ReLU -> Conv2d(10, 10, 3, same) -> ReLU
    -> MaxPool2d(8, 8) -> Dropout2d(0.25)
    -> Conv2d(10, 128, 8, valid) -> ReLU -> Dropout2d(0.5)
    -> Conv2d(128, 1, 1, valid) -> Tanh
"""

import numpy as np
import torch
import torch.nn as nn


class CarDetectorCNN(nn.Module):
    """
    Mô hình Fully Convolutional Network phát hiện xe hơi (Heatmap Localization).
    Đầu vào chuẩn x có giá trị pixel [0, 255] hoặc đã chuẩn hóa [-1, 1] nếu normalize_input=False.
    """
    def __init__(self, in_channels: int = 3, normalize_input: bool = True):
        super().__init__()
        self.normalize_input = normalize_input
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(10, 128, kernel_size=8, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = x / 127.5 - 1.0
        return self.net(x)

    def locate_cars(self, image_rgb_uint8: np.ndarray, threshold: float = 0.99):
        """
        Tái hiện hàm locate() từ 2_obj_detection.ipynb:
        Trả về (heatmap, sel_heatmap, annotated_image_rgb).
        """
        self.eval()
        device = next(self.parameters()).device
        annotated = image_rgb_uint8.copy()
        tensor_img = torch.tensor(
            image_rgb_uint8.transpose(2, 0, 1), dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            hm = self.forward(tensor_img).squeeze(0).squeeze(0).cpu().numpy()

        sel_hm = hm > threshold
        yy, xx = np.where(sel_hm)
        h_img, w_img, _ = annotated.shape
        for j, i in zip(yy, xx):
            y0, y1 = min(j * 8, h_img - 1), min(j * 8 + 64, h_img - 1)
            x0, x1 = min(i * 8, w_img - 1), min(i * 8 + 64, w_img - 1)
            annotated[y0:y1, x0] = (0, 255, 0)
            annotated[y0:y1, x1] = (0, 255, 0)
            annotated[y0, x0:x1] = (0, 255, 0)
            annotated[y1, x0:x1] = (0, 255, 0)

        return hm, sel_hm, annotated
