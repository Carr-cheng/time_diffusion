# imgToTime.py

import torch
import torch.nn as nn
from typing import Literal

class ImageToTimeSeries(nn.Module):
    """
    将生成图像解码为时间序列，支持 mean-pooling 和 CNN decoder 两种方式，
    并自适应输入通道数。

    输入：
        img: Tensor of shape (B, C, H, W)
    输出：
        ts: Tensor of shape (B, T, D)
    参数：
        in_channels: 输入图像的通道数 C
        out_len: 预测序列长度 T
        out_dim: 序列维度 D
        mode: 'mean' 或 'cnn'
    """
    def __init__(
        self,
        in_channels: int,
        out_len: int = 20,
        out_dim: int = 1,
        mode: Literal['mean','cnn'] = 'cnn'
    ):
        super().__init__()
        self.mode = mode
        self.out_len = out_len
        self.out_dim = out_dim

        if mode == 'cnn':
            # 自适应 in_channels
            self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                # 自适应池化到 (out_len, out_dim)
                nn.AdaptiveAvgPool2d((out_len, out_dim))
            )
            # 将 64 维特征映射到单通道点
            self.linear = nn.Linear(64, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W)
        Returns:
            ts: (B, T, D)
        """
        if self.mode == 'mean':
            B, C, H, W = img.shape
            ts_point = img.mean(dim=[2,3])              # (B, C)
            ts = ts_point.unsqueeze(1).repeat(1, self.out_len, 1)  # (B, T, C)
            return ts

        elif self.mode == 'cnn':
            x = self.decoder(img)                       # (B, 64, T, D)
            x = x.permute(0, 2, 3, 1)                   # -> (B, T, D, 64)
            ts = self.linear(x).squeeze(-1)             # -> (B, T, D)
            return ts

        else:
            raise NotImplementedError(f"Unknown mode {self.mode}")
