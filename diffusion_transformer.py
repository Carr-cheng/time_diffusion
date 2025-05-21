# diffusio_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Transformer Condition Encoder
# ----------------------------
class TransformerConditionEncoder(nn.Module):
    def __init__(self, img_shape, condition_dim=128):
        super().__init__()
        C, H, W = img_shape
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(C * H * W, condition_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=condition_dim, nhead=4),
            num_layers=2
        )

    def forward(self, x):  # x: (B, C, H, W)
        b = x.size(0)
        x = self.flatten(x).unsqueeze(0)  # (1, B, C*H*W)
        x = self.linear(x)                # (1, B, condition_dim)
        out = self.transformer(x)         # (1, B, condition_dim)
        return out.squeeze(0)             # (B, condition_dim)

# ----------------------------
# UNet 模拟网络（简化版）
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Linear(cond_dim, 64)
        self.up = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x, t, cond_embed):  # x: (B,C,H,W), cond_embed: (B, cond_dim)
        h = self.down(x)  # (B,64,H,W)
        c = self.middle(cond_embed).view(-1, 64, 1, 1)
        h = h + c  # 简单加和
        return self.up(h)

# ----------------------------
# Scheduler（时间调度器）
# ----------------------------
class LinearScheduler:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise):
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(x_start.device)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise

    def p_sample(self, z, eps_pred, t):
        beta = self.betas[t].view(-1, 1, 1, 1).to(z.device)
        alpha = self.alphas[t].view(-1, 1, 1, 1).to(z.device)
        return (z - beta / torch.sqrt(1 - self.alpha_bars[t].view(-1,1,1,1)) * eps_pred) / torch.sqrt(alpha)

# ----------------------------
# Conditional Diffusion Model
# ----------------------------
class ConditionalDiffusion(nn.Module):
    def __init__(self, img_shape, condition_dim=128, num_steps=1000):
        super().__init__()
        self.condition_encoder = TransformerConditionEncoder(img_shape, condition_dim)
        self.unet = SimpleUNet(in_channels=img_shape[0], cond_dim=condition_dim)
        self.scheduler = LinearScheduler(num_steps)

    def forward(self, cond_img, tgt_img=None, inference=False):
        c = self.condition_encoder(cond_img)

        if inference:
            # 推理阶段：从 z_T 开始生成
            z = torch.randn_like(cond_img)
            for t in reversed(range(self.scheduler.num_steps)):
                t_tensor = torch.full((z.size(0),), t, device=z.device, dtype=torch.long)
                eps_pred = self.unet(z, t_tensor, c)
                z = self.scheduler.p_sample(z, eps_pred, t_tensor)
            return z
        else:
            # 训练阶段：学习预测噪声
            t = torch.randint(0, self.scheduler.num_steps, (cond_img.size(0),), device=cond_img.device)
            noise = torch.randn_like(tgt_img)
            z_t = self.scheduler.q_sample(tgt_img, t, noise)
            eps_pred = self.unet(z_t, t, c)
            loss = F.mse_loss(eps_pred, noise)
            return loss
