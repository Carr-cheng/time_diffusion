# evaluate.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataLoads import load_smd_dataset
from diffusion_transformer import ConditionalDiffusion
from imgToTime import ImageToTimeSeries
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

# === 配置 ===
DATA_DIR = './ServerMachineDataset/train'
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'saved_model.pth'

# === 加载数据集（与训练时一致）===
train_ds, test_ds = load_smd_dataset(
    data_dir=DATA_DIR,
    hist_len=100,
    fut_len=20,
    stride=5,
    ts_norm='minmax',
    ts_smooth=5,
    img_method='mtf',
    img_bins=8,
    img_augment=False,  # 评估时不加随机增强
    split_ratio=0.8,
)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# === 创建模型结构（形状参考样本）===
sample = test_ds[0]
C, H, W = sample['cond_img'].shape
future_dim = sample['future_ts'].shape[1]

model = ConditionalDiffusion(img_shape=(C, H, W)).to(DEVICE)
decoder = ImageToTimeSeries(in_channels=C, out_len=20, out_dim=future_dim, mode='cnn').to(DEVICE)

# === 加载保存的模型参数 ===
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['diffusion_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

model.eval()
decoder.eval()

# === 执行推理和评估 ===
preds, gts = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        cond = batch['cond_img'].to(DEVICE)
        fut_ts = batch['future_ts']
        pred_img = model(cond, inference=True).cpu()
        pred_ts = decoder(pred_img)
        preds.append(pred_ts.squeeze(0))
        gts.append(fut_ts)

preds = torch.stack(preds)
gts = torch.stack(gts)
mse = ((preds - gts) ** 2).mean().item()

print(f"\n✅ 模型评估完成 - Test MSE: {mse:.6f}")

# === 保存预测和真实值为 CSV ===
SAVE_DIR = './eval_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# 还原维度：[samples, future_len, features]
preds_np = preds.numpy().reshape(preds.shape[0], -1)
gts_np   = gts.numpy().reshape(gts.shape[0], -1)

pred_df = pd.DataFrame(preds_np)
gt_df   = pd.DataFrame(gts_np)

pred_df.to_csv(os.path.join(SAVE_DIR, 'predictions.csv'), index=False)
gt_df.to_csv(os.path.join(SAVE_DIR, 'ground_truth.csv'), index=False)

print(f"✅ 预测结果已保存到 {SAVE_DIR}/predictions.csv 和 ground_truth.csv")

VIS_DIR = os.path.join(SAVE_DIR, 'plots_all_features')
os.makedirs(VIS_DIR, exist_ok=True)

num_features = preds.shape[2]
fut_len = preds.shape[1]

for i in range(len(preds)):
    fig, axs = plt.subplots(nrows=math.ceil(num_features / 4), ncols=4, figsize=(20, 2.5 * math.ceil(num_features / 4)))
    axs = axs.flatten()

    for j in range(num_features):
        pred_seq = preds[i][:, j].numpy()
        gt_seq   = gts[i][:, j].numpy()

        axs[j].plot(gt_seq, label='Ground Truth', color='blue', linewidth=1.5)
        axs[j].plot(pred_seq, label='Prediction', color='orange', linestyle='--', linewidth=1.5)
        axs[j].set_title(f"Feature {j}")
        axs[j].set_xlabel("Time Step")
        axs[j].set_ylabel("Value")
        axs[j].legend(loc="upper right", fontsize=8)

    # 移除多余子图
    for k in range(num_features, len(axs)):
        fig.delaxes(axs[k])

    fig.suptitle(f"Sample {i} - All 38 Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(VIS_DIR, f"sample_{i:03d}_all_features.png"))
    plt.close()

print(f"✅ 所有特征图像已保存到：{VIS_DIR}")