"""CutPaste 自监督异常检测模型（计划书模型 5）。

原理：无需真实缺陷样本；
- 训练：对正常图像随机剪切粘贴生成伪异常（CutPaste / CutPaste-Scar），
  与正常样本做二分类自监督训练（ResNet18）
- 推理：补丁级滑窗分类（原论文定位方案）或异常类 GradCAM 生成热力图
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base import BaseAnomalyModel
from src.utils import resize_map, smooth_map


# ---------------------------------------------------------------------- #
# CutPaste 伪异常增广（GPU 向量化 tensor 实现，直接在归一化张量上操作）
# ---------------------------------------------------------------------- #
def cutpaste_single(x: torch.Tensor, variant: str = "normal") -> torch.Tensor:
    """对单张归一化张量 (3,H,W) 执行 CutPaste 剪切粘贴，返回新图像。

    注意：使用模块级 np.random（受 set_seed 控制），保证实验可复现。
    """
    h, w = x.shape[1], x.shape[2]
    area_ratio = float(np.random.uniform(0.02, 0.15))

    if variant == "scar":
        # 细长划痕状补丁（CutPaste-Scar）
        patch_h = int(np.random.uniform(2, 4))
        patch_w = max(2, int(area_ratio * h * w / patch_h))
    else:
        # 常规矩形补丁
        ratio = float(np.random.uniform(0.3, 3.3))
        patch_h = int(np.sqrt(area_ratio * ratio) * h)
        patch_w = int(np.sqrt(area_ratio / ratio) * w)
    patch_h = min(max(patch_h, 2), h - 1)
    patch_w = min(max(patch_w, 2), w - 1)

    x1 = int(np.random.randint(0, w - patch_w + 1))
    y1 = int(np.random.randint(0, h - patch_h + 1))
    x2 = int(np.random.randint(0, w - patch_w + 1))
    y2 = int(np.random.randint(0, h - patch_h + 1))

    patch = x[:, y1 : y1 + patch_h, x1 : x1 + patch_w].clone()
    # 模糊（GPU avg_pool 近似高斯）：避免分类器走「硬边缘」捷径（原论文策略）；
    # 模糊核按补丁尺寸自适应（划痕补丁可能只有 2~3 像素高）
    k = patch_h if patch_h % 2 == 1 else patch_h - 1
    k = min(k, 15)
    if k > 1:
        patch = F.avg_pool2d(patch.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2)[0]
    # 补丁亮度扰动（归一化空间小幅度加性扰动）
    patch = patch + float(np.random.uniform(-0.05, 0.05))

    out = x.clone()
    out[:, y2 : y2 + patch_h, x2 : x2 + patch_w] = patch
    # 全图亮度/对比度抖动（原论文：迫使分类器学习纹理一致性而非局部伪影）
    out = out * float(np.random.uniform(0.92, 1.08))
    out = out + float(np.random.uniform(-0.03, 0.03))
    return out


def cutpaste_batch(x: torch.Tensor) -> torch.Tensor:
    """批量生成伪异常（一半 CutPaste，一半 CutPaste-Scar）。"""
    out = []
    for i in range(x.shape[0]):
        variant = "normal" if i % 2 == 0 else "scar"
        out.append(cutpaste_single(x[i], variant))
    return torch.stack(out)


# ---------------------------------------------------------------------- #
# GradCAM
# ---------------------------------------------------------------------- #
class _GradCAM:
    """ResNet 最后一层卷积的 GradCAM。"""

    def __init__(self, model: nn.Module, target: nn.Module):
        self.activations, self.gradients = None, None
        self.fwd = target.register_forward_hook(self._fwd)
        # 用 full_backward_hook：BasicBlock 前向含多个 autograd 节点，
        # 旧版非完整钩子可能拿不到梯度
        self.bwd = target.register_full_backward_hook(self._bwd)

    def _fwd(self, module, i, o):
        self.activations = o.detach()

    def _bwd(self, module, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, score: torch.Tensor):
        self.gradients = None
        score.backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)  # 梯度权重
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        return cam

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()


# ---------------------------------------------------------------------- #
# CutPaste 模型
# ---------------------------------------------------------------------- #
class CutPasteModel(BaseAnomalyModel):
    """CutPaste：伪异常自监督二分类 + GradCAM 定位。"""

    name = "cutpaste"

    def __init__(
        self,
        device="auto",
        image_size: int = 256,
        epochs: int = 50,
        lr: float = 0.01,
        patience: int = 8,
        min_epochs: int = 10,  # 早停保护：epoch 数不足不保存模型（防止欠拟合最优被选中）
        loc_mode: str = "gradcam",  # 定位方式：gradcam=异常类GradCAM（默认）/ patch=补丁级分类
        patch_size: int = 32,
    ):
        super().__init__(device, image_size)
        self.epochs, self.lr, self.patience, self.min_epochs = epochs, lr, patience, min_epochs
        self.loc_mode = loc_mode
        self.patch_size = patch_size
        self.net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)  # 正常 / 伪异常
        self.net = self.net.to(self.device)
        self._cam: _GradCAM | None = None

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def train(self, train_loader) -> None:
        net = self.net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 切 10% 验证集做早停
        images = torch.cat([b["image"] for b in train_loader], dim=0)
        n_val = max(1, len(images) // 10)
        val_images, train_images = images[:n_val], images[n_val:]

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        n = len(train_images)
        print(f"[{self.name}] 二分类训练：正常 {n} + 伪异常 {n}，epochs={self.epochs}")
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, 32):
                xb = train_images[perm[i : i + 32]].to(self.device)
                pseudo = self._make_pseudo(xb)
                inputs = torch.cat([xb, pseudo], dim=0)
                labels = torch.cat(
                    [torch.zeros(len(xb)), torch.ones(len(pseudo))]
                ).long().to(self.device)
                logits = net(inputs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(inputs)
            train_loss = total / (2 * n)

            with torch.no_grad():
                vp = self._make_pseudo(val_images.to(self.device))
                v_inputs = torch.cat([val_images.to(self.device), vp], dim=0)
                v_labels = torch.cat(
                    [torch.zeros(len(val_images)), torch.ones(len(vp))]
                ).long().to(self.device)
                val_loss = criterion(net(v_inputs), v_labels).item()

            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                # min_epochs 之前不保存，避免选中欠拟合的最优
                if epoch >= self.min_epochs:
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
            if epoch % 10 == 0:
                print(f"  epoch {epoch:3d} | train loss {train_loss:.5f} | val loss {val_loss:.5f}")
            if best_state is not None and bad_epochs >= self.patience:
                print(f"[{self.name}] 早停于 epoch {epoch}")
                break

        if best_state is None:
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        print(f"[{self.name}] 训练完成，最佳 val loss = {best_loss:.5f}")

    def _make_pseudo(self, xb: torch.Tensor) -> torch.Tensor:
        """直接在归一化张量上生成伪异常（GPU 向量化，无 uint8 往返）。"""
        return cutpaste_batch(xb)

    # ------------------------------------------------------------------ #
    # 推理：异常热力图
    # ------------------------------------------------------------------ #
    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        self.net.eval()
        x = image.to(self.device)
        if self.loc_mode == "gradcam":
            return self._predict_gradcam(x)
        return self._predict_patch(x)

    def _predict_patch(self, x: torch.Tensor) -> np.ndarray:
        """补丁级定位（原论文做法）：滑窗切块逐块二分类。

        异常得分 = 1 - P(正常)。分类器在整图正常纹理上训练，正常区域补丁
        得分高、缺陷区域补丁得分低，拼成网格图即为异常热力图。
        """
        ps = self.patch_size
        stride = max(ps // 2, 4)
        unfold = F.unfold(x, kernel_size=ps, stride=stride)  # (1, 3*ps*ps, L)
        L = unfold.shape[-1]
        patches = unfold[0].t().reshape(L, 3, ps, ps)
        p_normal = []
        with torch.no_grad():
            for i in range(0, L, 64):
                logits = self.net(patches[i : i + 64])
                p_normal.append(F.softmax(logits, dim=1)[:, 0])
        p_normal = torch.cat(p_normal).cpu().numpy()  # (L,)
        gh = (self.image_size - ps) // stride + 1  # 网格边长（输入为方形）
        amap = (1.0 - p_normal).reshape(gh, gh)
        amap = resize_map(amap, (self.image_size, self.image_size))
        return smooth_map(amap, sigma=4)

    def _predict_gradcam(self, x: torch.Tensor) -> np.ndarray:
        """GradCAM 定位 + 分类概率校准（原论文 CutPaste+GradCAM 方案）。

        - 定位：对异常类得分反向传播，突出疑似伪异常（不连续纹理）区域
        - 校准：cam 逐图除以自身最大值再乘以 P(异常)，使热力图峰值等于
          分类器异常概率——图像级得分（取最大值）与论文的分类概率一致，
          同时保留 GradCAM 的空间分布用于像素级定位
        """
        if self._cam is None:
            # 必须在 forward 之前注册钩子，否则捕获不到激活与梯度
            self._cam = _GradCAM(self.net, self.net.layer4[-1])
        x = x.detach().requires_grad_(True)
        # GradCAM 依赖反向传播，须显式开启梯度（app.detect 外层统一
        # torch.no_grad()，此处 enable_grad 优先于外层生效）
        with torch.enable_grad():
            logits = self.net(x)
            p_anomaly = F.softmax(logits, dim=1)[0, 1].item()
            cam = self._cam(logits[:, 1].sum())  # 异常类得分反向传播
            cam = cam.detach()
        cam = F.interpolate(cam, size=(self.image_size, self.image_size), mode="bilinear")[0, 0]
        cam = smooth_map(cam.detach().cpu().numpy(), sigma=4)
        cam_max = cam.max()
        if cam_max < 1e-8:
            # 无定位信号（如自信判为正常的图）：均匀图，峰值仍为 P(异常)，
            # 保证图像级得分与分类概率完全一致
            return np.full_like(cam, p_anomaly)
        # 先平滑后缩放：峰值精确等于 P(异常)，图像级得分与分类概率一致
        return cam / cam_max * p_anomaly

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "config": {"epochs": self.epochs, "lr": self.lr}}

    def _load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
