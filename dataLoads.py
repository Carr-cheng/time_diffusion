# dataLoads.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from DiT_Graph.timeToImg import TimeToImage  # 用于将序列转图像
import torchvision.transforms as T  # 图像端增强

def normalize_series(data: np.ndarray, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError
    data_norm = scaler.fit_transform(data)
    return data_norm, scaler

def smooth_series(data: np.ndarray, window=5):
    df = pd.DataFrame(data)
    return df.rolling(window=window, min_periods=1, center=True).mean().values

def slide_windows(data: np.ndarray, hist_len, fut_len, stride=1):
    samples = []
    T = len(data)
    for start in range(0, T - hist_len - fut_len + 1, stride):
        hist = data[start : start + hist_len]
        fut  = data[start + hist_len : start + hist_len + fut_len]
        samples.append((hist, fut))
    return samples

class AugmentedSMDataset(Dataset):
    """
    将时序->滑窗->预处理->图像转换->图像增强 整合到一个 Dataset
    """
    def __init__(self,
                 raw: np.ndarray,
                 hist_len: int,
                 fut_len: int,
                 stride: int,
                 ts_norm: str = 'minmax',
                 ts_smooth: int = 5,
                 img_method: str = 'mtf',
                 img_bins: int = 8,
                 img_augment: bool = True):
        # 1. 时序端预处理
        data = smooth_series(raw, window=ts_smooth)
        data, self.scaler = normalize_series(data, method=ts_norm)
        # 2. 滑窗生成样本
        self.samples = slide_windows(data, hist_len, fut_len, stride=stride)
        # 3. 图像转换器
        self.to_image = TimeToImage(method=img_method, n_bins=img_bins)
        # 4. 图像增强
        if img_augment:
            self.augment = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(degrees=5),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor()
            ])
        else:
            self.augment = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist, fut = self.samples[idx]
        # 转为图像
        cond_img = self.to_image(torch.tensor(hist).unsqueeze(0))  # (1,H,W) or (D,H,W)
        fut_img  = self.to_image(torch.tensor(fut).unsqueeze(0))
        # 增强
        if self.augment:
            cond_img = self.augment(cond_img)
            fut_img  = self.augment(fut_img)
        return {
            'cond_img': cond_img,                         # (C,H,W)
            'tgt_img': fut_img,                           # (C,H,W)
            'future_ts': torch.tensor(fut, dtype=torch.float32)  # (fut_len, D)
        }

def load_smd_dataset(data_dir: str,
                     hist_len=100,
                     fut_len=20,
                     stride=5,
                     ts_norm='minmax',
                     ts_smooth=5,
                     img_method='mtf',
                     img_bins=8,
                     img_augment=True,
                     split_ratio=0.8,
                     seed=42):
    # 读取所有机器
    machines = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt') or fname.endswith('.csv') or fname.endswith('.npy'):
            path = os.path.join(data_dir, fname)
            raw = np.loadtxt(path, delimiter=',') if fname.endswith(('.txt','.csv')) else np.load(path)
            machines.append(raw)
    # 构造每台机器的 Dataset
    all_ds = []
    for raw in machines:
        ds = AugmentedSMDataset(raw, hist_len, fut_len, stride,
                                ts_norm, ts_smooth, img_method, img_bins, img_augment)
        all_ds.append(ds)
    # 按机器划分 train/test
    torch.manual_seed(seed)
    n_train = int(len(all_ds) * split_ratio)
    train_list, test_list = random_split(all_ds, [n_train, len(all_ds) - n_train])
    train_ds = ConcatDataset(train_list)
    test_ds  = ConcatDataset(test_list)
    return train_ds, test_ds
