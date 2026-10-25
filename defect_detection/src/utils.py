"""通用工具：随机种子、设备选择、热力图可视化。"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于服务器/脚本运行
import matplotlib.pyplot as plt
import numpy as np
import torch

# 中文字体：热力图标题含中文，默认 DejaVu Sans 缺 CJK 字形会渲染成方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ImageNet 均值 / 标准差（与计划书预处理流水线一致）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """返回可用设备：优先 CUDA，否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(img: np.ndarray) -> np.ndarray:
    """反归一化：ImageNet 归一化图像 -> [0,1] 原始图像 (H,W,3)。"""
    img = img * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    return np.clip(img, 0.0, 1.0)


def min_max_norm(x: np.ndarray) -> np.ndarray:
    """逐张图 min-max 归一化到 [0,1]。"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def smooth_map(anomaly_map: np.ndarray, sigma: int = 4) -> np.ndarray:
    """高斯平滑异常热力图，去除散点噪声。"""
    kernel = int(sigma * 4) | 1  # 保证奇数
    return cv2.GaussianBlur(anomaly_map, (kernel, kernel), sigma)


def resize_map(anomaly_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """双线性放大异常图到目标尺寸 (H,W)。"""
    return cv2.resize(anomaly_map, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def save_triplet(
    save_path: Path,
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: np.ndarray | None = None,
    title: str = "",
) -> None:
    """保存「原图 + 热力图 + 缺陷掩码」三联图（报告素材）。

    Args:
        save_path: 输出文件路径。
        image: 原始图像 (H,W,3)，范围 [0,1]。
        anomaly_map: 异常热力图 (H,W)，任意范围（内部自动 min-max 归一化）。
        mask: 真实缺陷掩码 (H,W)，0/1；为 None 时只画两联。
        title: 图片标题（含图像级得分等信息）。
    """
    n_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2))
    # n_cols 恒 >= 2，subplots 返回的 axes 本身就是一维数组，直接使用
    axes = np.atleast_1d(axes)

    axes[0].imshow(image)
    axes[0].set_title("原图")
    im = axes[1].imshow(min_max_norm(anomaly_map), cmap="jet")
    axes[1].set_title("异常热力图")
    if mask is not None:
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("真实缺陷掩码")

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def heatmap_overlay(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """将热力图以半透明方式叠加到原图上 (H,W,3) uint8，用于 Gradio 展示。"""
    heat = cv2.applyColorMap(
        (min_max_norm(anomaly_map) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    base = (image * 255).astype(np.uint8)[..., ::-1]  # RGB -> BGR
    return cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)
