# timeToimg.py

import numpy as np
import torch
from typing import Literal

def min_max_scale(x: np.ndarray) -> np.ndarray:
    min_x, max_x = np.min(x), np.max(x)
    return (x - min_x) / (max_x - min_x + 1e-8)

def gaf(ts: np.ndarray) -> np.ndarray:
    ts = min_max_scale(ts) * 2 - 1
    ts = np.clip(ts, -1, 1)
    phi = np.arccos(ts)
    return np.cos(phi[:, None] + phi[None, :])

def mtf(ts: np.ndarray, n_bins: int = 8) -> np.ndarray:
    ts = min_max_scale(ts)
    bins = np.linspace(0, 1, n_bins + 1)
    digitized = np.digitize(ts, bins) - 1
    P = np.zeros((n_bins, n_bins))
    for i in range(len(digitized) - 1):
        P[digitized[i], digitized[i + 1]] += 1
    P_sum = P.sum(axis=1, keepdims=True)
    P_sum[P_sum == 0] = 1
    P = P / P_sum
    return P[digitized][:, digitized]

class TimeToImage:
    """
    将批量时间序列转换为图像：
      - 输入: ts_batch Tensor of shape (B, T, D)
      - 输出: img Tensor of shape (B, C, H, W)
    方法 method 支持 'gaf' 或 'mtf'，对于 D>1 时会对各维度生成多通道图像。
    """
    def __init__(self, method: Literal['gaf','mtf']='gaf', n_bins: int=8):
        self.method = method
        self.n_bins = n_bins

    def __call__(self, ts_batch: torch.Tensor) -> torch.Tensor:
        # ts_batch: (B, T, D)
        B, T, D = ts_batch.shape
        ts_np = ts_batch.detach().cpu().numpy()
        imgs = []
        for b in range(B):
            channels = []
            for d in range(D):
                ts = ts_np[b, :, d]
                if self.method == 'gaf':
                    img = gaf(ts)
                elif self.method == 'mtf':
                    img = mtf(ts, n_bins=self.n_bins)
                else:
                    raise ValueError(f"Unknown method {self.method}")
                channels.append(img)  # (H, W)
            # stack channels and add batch dim
            img_b = np.stack(channels, axis=0)      # (D, H, W)
            imgs.append(img_b)
        imgs = np.stack(imgs, axis=0)               # (B, D, H, W)
        return torch.tensor(imgs, dtype=torch.float32)
