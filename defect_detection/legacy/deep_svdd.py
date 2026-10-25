"""DeepSVDD 深度单分类模型（计划书模型 3）。

原理：将正常样本映射到特征空间紧凑超球体内；推理时计算样本特征到
球心距离，距离越大越可能异常。
- 防坍塌：球心 c 固定（AE 预训练隐码均值）+ 全连接层无偏置 + L2 正则
- 初始化：先训练自编码器，用其编码器权重初始化（计划书 4.3）
- 像素级定位：滑窗 Patch 逐一送入编码器，按到球心距离生成热力图
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAnomalyModel
from src.utils import resize_map, smooth_map


class SVDDEncoder(nn.Module):
    """编码器：全卷积 + 全局平均池化 + 无偏置全连接（输入尺寸无关）。"""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, latent_dim, bias=False)  # 无偏置，防止超球面坍塌

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gap(self.convs(x)).flatten(1)
        return self.fc(z)


class SVDDDecoder(nn.Module):
    """解码器（仅 AE 预训练阶段使用）。"""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconvs(self.fc(z).view(z.shape[0], 256, 8, 8))


class DeepSVDDModel(BaseAnomalyModel):
    """DeepSVDD：AE 预训练初始化 -> 固定球心 -> 距离最小化训练。"""

    name = "deep_svdd"

    def __init__(
        self,
        device="auto",
        image_size: int = 256,
        latent_dim: int = 128,
        lr: float = 1e-4,
        epochs: int = 100,
        ae_pretrain_epochs: int = 50,
        patch_size: int = 64,
        patch_stride: int = 16,
    ):
        super().__init__(device, image_size)
        self.latent_dim = latent_dim
        self.lr, self.epochs = lr, epochs
        self.ae_pretrain_epochs = ae_pretrain_epochs
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.encoder = SVDDEncoder(latent_dim).to(self.device)
        self.center: torch.Tensor | None = None  # 超球面球心

    # ------------------------------------------------------------------ #
    # 训练：AE 预训练 -> 固定球心 -> SVDD 距离最小化
    # ------------------------------------------------------------------ #
    def train(self, train_loader) -> None:
        # 1) 收集全部正常样本
        images = torch.cat([b["image"] for b in train_loader], dim=0)

        # 2) AE 预训练初始化编码器（计划书 4.3）；
        #    基线实验（实验 4）ae_pretrain_epochs=0 时跳过，退化为随机初始化
        if self.ae_pretrain_epochs > 0:
            self._pretrain_ae(images)

        # 3) 固定球心：正常样本隐码均值
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(images.to(self.device))
        self.center = z.mean(dim=0, keepdim=True)  # (1,latent)

        # 4) SVDD 训练：最小化到球心距离
        self.encoder.train()
        optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.lr, weight_decay=1e-6
        )
        n = len(images)
        print(f"[{self.name}] SVDD 训练：{n} 张正常样本，epochs={self.epochs}")
        best_loss, best_state = float("inf"), None
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, 32):
                xb = images[perm[i : i + 32]].to(self.device)
                dist = (self.encoder(xb) - self.center).pow(2).sum(dim=1).mean()
                optimizer.zero_grad()
                dist.backward()
                optimizer.step()
                total += dist.item() * len(xb)
            avg = total / n
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | 平均球心距离 {avg:.5f}")
        self.encoder.load_state_dict(best_state)
        self.encoder.eval()
        print(f"[{self.name}] 训练完成，最佳平均球心距离 = {best_loss:.5f}")

    def _pretrain_ae(self, images: torch.Tensor) -> None:
        """用自编码器预训练编码器权重。"""
        decoder = SVDDDecoder(self.latent_dim).to(self.device)
        params = list(self.encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        criterion = nn.MSELoss()
        n = len(images)
        print(f"[{self.name}] AE 预训练初始化（{self.ae_pretrain_epochs} epochs）...")
        self.encoder.train()
        decoder.train()
        for epoch in range(1, self.ae_pretrain_epochs + 1):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, 32):
                xb = images[perm[i : i + 32]].to(self.device)
                rec = decoder(self.encoder(xb))
                loss = criterion(rec, xb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            if epoch % 10 == 0:
                print(f"  AE 预训练 epoch {epoch:3d} | MSE {total / n:.5f}")

    # ------------------------------------------------------------------ #
    # 推理：滑窗 Patch 到球心距离 -> 热力图
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        assert self.center is not None, "请先调用 train()"
        self.encoder.eval()
        x = image.to(self.device)  # (1,3,H,W)
        h, w = x.shape[-2], x.shape[-1]
        ps, st = self.patch_size, self.patch_stride

        # 滑窗裁剪全部 Patch，批量过编码器
        patches = []
        for i in range(0, h - ps + 1, st):
            for j in range(0, w - ps + 1, st):
                patches.append(x[:, :, i : i + ps, j : j + ps])
        grid_h = (h - ps) // st + 1
        grid_w = (w - ps) // st + 1
        patches_t = torch.cat(patches, dim=0)  # (P,3,ps,ps)
        z = self.encoder(patches_t)
        dist = (z - self.center).pow(2).sum(dim=1).cpu().numpy()  # (P,)
        amap = dist.reshape(grid_h, grid_w)
        amap = resize_map(amap, (h, w))
        return smooth_map(amap, sigma=4)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "center": self.center,
            "latent_dim": self.latent_dim,
        }

    def _load_state_dict(self, state: dict) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.center = state["center"].to(self.device)
