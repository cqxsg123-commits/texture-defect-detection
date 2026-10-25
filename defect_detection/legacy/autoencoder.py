"""卷积自编码器（重建基线，计划书模型 2）。

原理：仅用正常样本训练，编码器压缩为低维隐向量，解码器重建原图；
缺陷区域未被模型学习，重建残差大，残差大小即异常程度。

训练：Denoising MSE 损失（输入加高斯噪声，强迫解码器学习平滑先验，
对破坏纹理连续性的缺陷更敏感）+ Adam(1e-4) + Dropout + 早停（计划书 4.3）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAnomalyModel
from src.utils import smooth_map


class ConvAutoEncoder(nn.Module):
    """对称卷积自编码器（256×256 -> 8×8×256 隐空间 -> 256×256）。"""

    def __init__(self, latent_channels: int = 256, dropout: float = 0.2):
        super().__init__()
        # 编码器：每层 stride=2 下采样
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, latent_channels, 4, 2, 1), nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2),
        )
        # Dropout 抑制过拟合（计划书：网络增加 Dropout 层）；
        # dropout=0 时退化为无正则基线（实验 4 基线实验用）
        self.dropout = nn.Dropout2d(dropout)
        # 解码器：转置卷积对称上采样
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.encoder(x))
        return self.decoder(z)


class AutoEncoderModel(BaseAnomalyModel):
    """AutoEncoder 异常检测模型（统一接口封装）。"""

    name = "autoencoder"

    def __init__(
        self,
        device="auto",
        image_size: int = 256,
        lr: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        latent_channels: int = 256,
        noise: float = 0.05,  # Denoising AE 的输入噪声强度（仅训练时）
        dropout: float = 0.2,  # 隐空间 Dropout 概率（0 = 无正则基线）
    ):
        super().__init__(device, image_size)
        self.net = ConvAutoEncoder(latent_channels, dropout).to(self.device)
        self.lr, self.epochs, self.patience, self.noise = lr, epochs, patience, noise
        self.dropout = dropout

    def train(self, train_loader) -> None:
        """训练 + 早停：按 9:1 从训练集切出验证集监控 val loss。"""
        net = self.net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # 从正常训练样本中切出验证集（仅用于早停，不参与梯度更新）
        all_images = []
        for batch in train_loader:
            all_images.append(batch["image"])
        images = torch.cat(all_images, dim=0)
        n_val = max(1, len(images) // 10)
        val_images = images[:n_val].to(self.device)
        train_images = images[n_val:]

        best_val, bad_epochs = float("inf"), 0
        n_train = len(train_images)
        print(f"[{self.name}] 训练集 {n_train} 张，验证集 {len(val_images)} 张，epochs={self.epochs}")
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n_train)
            total_loss = 0.0
            for i in range(0, n_train, 32):
                xb = train_images[perm[i : i + 32]].to(self.device)
                # Denoising：输入加高斯噪声，重建目标仍是干净图
                noisy = xb + torch.randn_like(xb) * self.noise
                rec = net(noisy)
                loss = criterion(rec, xb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            train_loss = total_loss / n_train

            with torch.no_grad():
                val_loss = criterion(net(val_images), val_images).item()

            if val_loss < best_val - 1e-5:
                best_val, bad_epochs = val_loss, 0
                self._best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
            if epoch % 10 == 0 or bad_epochs == 0:
                print(f"  epoch {epoch:3d} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")

            if bad_epochs >= self.patience:
                print(f"[{self.name}] 早停于 epoch {epoch}（{self.patience} 轮无改善）")
                break

        net.load_state_dict(self._best_state)
        net.eval()
        print(f"[{self.name}] 训练完成，最佳 val loss = {best_val:.5f}")

    @torch.no_grad()
    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        """重建残差 -> 通道平均 -> 高斯平滑，得到异常热力图。"""
        self.net.eval()
        x = image.to(self.device)
        rec = self.net(x)
        residual = (x - rec).pow(2).mean(dim=1)  # (1,H,W) 逐像素残差
        amap = residual[0].cpu().numpy()
        # sigma=8：更大平滑核抑制单像素噪声，图像级得分更稳定
        return smooth_map(amap, sigma=8)

    def _state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "config": {"lr": self.lr, "epochs": self.epochs}}

    def _load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
