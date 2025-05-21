# __init__.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_transformer import ConditionalDiffusion
from dataLoads import load_smd_dataset
from imgToTime import ImageToTimeSeries

# 配置
DATA_DIR = './ServerMachineDataset/train'
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载并增强数据
train_ds, test_ds = load_smd_dataset(
    data_dir=DATA_DIR,
    hist_len=100,
    fut_len=20,
    stride=5,
    ts_norm='minmax',
    ts_smooth=5,
    img_method='mtf',
    img_bins=8,
    img_augment=True,
    split_ratio=0.8,
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# 2. 准备模型
# 从 train_ds 中拿一个样本确认图像 shape
sample = train_ds[0]
C, H, W = sample['cond_img'].shape
model = ConditionalDiffusion(img_shape=(C, H, W)).to(DEVICE)
decoder = ImageToTimeSeries(in_channels=C, out_len=20, out_dim=sample['future_ts'].shape[1], mode='cnn')
optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-4)

# 3. 训练
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        cond = batch['cond_img'].to(DEVICE)
        tgt  = batch['tgt_img'].to(DEVICE)
        fut_ts = batch['future_ts'].to(DEVICE)

        loss = model(cond, tgt)  # MSE in diffusion module
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

torch.save({
    'diffusion_state_dict': model.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'saved_model.pth')
print("模型已保存为 saved_model.pth")

