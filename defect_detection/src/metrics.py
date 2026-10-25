"""评估指标：图像级 AUROC、像素级 AUROC（计划书 2. 任务定义）。

协议（与 Anomalib 官方一致，保证 3 个模型对比公平）：
- 图像级异常得分 = 原始（未归一化）异常热力图的最大值
- 像素级 AUROC：每张异常图先做逐图 min-max 归一化，再全局展平计算
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """图像级 AUROC。

    Args:
        scores: 每张测试图的异常得分 (N,)，越大越异常。
        labels: 真实标签 (N,)，0=正常 1=缺陷。
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(anomaly_maps: np.ndarray, masks: np.ndarray) -> float:
    """像素级 AUROC。

    Args:
        anomaly_maps: 原始异常热力图 (N,H,W)，任意范围。
        masks: 真实缺陷掩码 (N,H,W)，0/1。
    """
    maps = anomaly_maps.reshape(len(anomaly_maps), -1)
    masks = masks.reshape(len(masks), -1)

    # 逐图 min-max 归一化（异常图最大值位置不固定，需先归一化再跨图比较）
    normalized = []
    for m in maps:
        m_min, m_max = m.min(), m.max()
        if m_max - m_min < 1e-8:
            normalized.append(np.zeros_like(m))
        else:
            normalized.append((m - m_min) / (m_max - m_min))

    flat_maps = np.concatenate(normalized).ravel()
    flat_masks = masks.ravel()
    return float(roc_auc_score(flat_masks, flat_maps))


def anomaly_map_to_score(anomaly_map: np.ndarray) -> float:
    """原始异常图 -> 图像级异常得分（取最大值，Anomalib 协议）。"""
    return float(anomaly_map.max())
