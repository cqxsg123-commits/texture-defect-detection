"""统一评估流程：图像级 AUROC、PixelAUROC、单张推理耗时、对比表与热力图样例。

3 个模型共用本模块，保证实验条件严格一致（计划书 4.3 对比实验设置）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.metrics import compute_image_auroc, compute_pixel_auroc
from src.models.base import BaseAnomalyModel
from src.models.padim_anomalib import PadimAnomalibModel
from src.models.patchcore import PatchCoreModel
from src.utils import denormalize, save_triplet

TIMING_RUNS = 20  # 推理耗时测量的重复次数


def _norm_path(p) -> str:
    return os.path.normpath(os.path.abspath(str(p)))


def evaluate_model(
    model: BaseAnomalyModel,
    test_loader,
    heatmap_dir: str | Path | None = None,
    n_heatmap_samples: int = 3,
) -> dict:
    """评估单个模型。

    Returns:
        {"model", "image_auroc", "pixel_auroc", "mean_inference_ms", "n_test"}
    """
    print(f"[eval] 评估模型：{model.name}")
    ds = test_loader.dataset

    # ---- 收集全部测试图的原始异常热力图（保持数据集顺序） ----
    if isinstance(model, PadimAnomalibModel):
        # Anomalib 适配层：整测试集预测后按路径对齐
        maps_by_path = {_norm_path(k): v for k, v in model.predict_dataset().items()}
        raw_maps = np.stack([maps_by_path[_norm_path(ds.image_paths[i])] for i in range(len(ds))])
    elif isinstance(model, PatchCoreModel):
        # 官方评估协议：FAISS CPU 检索生成热力图（指标与优化前完全一致；
        # GPU 批量检索仅用于部署与耗时测试，两者地图差异为 fp32 求和次序噪声）
        maps = []
        for batch in test_loader:
            for i in range(batch["image"].shape[0]):
                maps.append(model.predict_image_faiss(batch["image"][i : i + 1]))
        raw_maps = np.stack(maps)
    else:
        maps = []
        model_predict = model.predict
        for batch in test_loader:
            maps.append(model_predict(batch["image"].to(model.device)))
        raw_maps = np.concatenate(maps, axis=0)

    labels = np.array(ds.labels)
    masks = np.stack([np.asarray(ds[i]["mask"][0]) for i in range(len(ds))])

    # ---- 指标 ----
    scores = raw_maps.max(axis=(1, 2))  # 图像级得分 = 原始热力图最大值
    image_auroc = compute_image_auroc(scores, labels)
    pixel_auroc = compute_pixel_auroc(raw_maps, masks)

    # ---- 推理耗时 ----
    mean_ms = _measure_time(model)

    result = {
        "model": model.name,
        "image_auroc": round(image_auroc, 4),
        "pixel_auroc": round(pixel_auroc, 4),
        "mean_inference_ms": round(mean_ms, 2),
        "n_test": len(ds),
    }
    print(
        f"[eval] {model.name}: Image AUROC={result['image_auroc']:.4f} | "
        f"Pixel AUROC={result['pixel_auroc']:.4f} | 推理耗时 {result['mean_inference_ms']:.2f} ms/张"
    )

    # ---- 热力图样例（报告素材） ----
    if heatmap_dir is not None:
        heatmap_dir = Path(heatmap_dir) / model.name
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        defect_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]
        for tag, idxs in (("defect", defect_idx), ("normal", normal_idx)):
            if len(idxs) == 0:
                continue
            for k in range(min(n_heatmap_samples, len(idxs))):
                i = idxs[k]
                img = denormalize(ds[i]["image"].numpy().transpose(1, 2, 0))
                save_triplet(
                    heatmap_dir / f"{tag}_{k}.png",
                    img,
                    raw_maps[i],
                    np.asarray(ds[i]["mask"][0]) if labels[i] == 1 else None,
                    title=f"{model.name} | score={scores[i]:.4f} | label={labels[i]}",
                )
        print(f"[eval] 热力图样例已保存 -> {heatmap_dir}")

    return result


def _measure_time(model: BaseAnomalyModel) -> float:
    """测量单张推理耗时（ms）：单图 predict_image 重复 20 次取均值（含 GPU 同步）。

    三模型统一口径，测的是实际部署的单张推理路径：
    - PatchCore：批量前向提取 + GPU 分块 GEMM 记忆库检索（predict_image）
    - PaDiM：直接前向 + GPU 马氏距离（绕开 Engine 调度，predict_image）
    - CutPaste：单次前向 + GradCAM
    """
    device = model.device
    sample = torch.randn(1, 3, model.image_size, model.image_size).to(device)
    model.predict_image(sample)  # 预热（含首次权重搬移）
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMING_RUNS):
        model.predict_image(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / TIMING_RUNS * 1000.0


def make_comparison_table(results: list[dict], save_dir: str | Path) -> pd.DataFrame:
    """生成对比表：保存 CSV + Markdown，返回 DataFrame。"""
    df = pd.DataFrame(results).sort_values("pixel_auroc", ascending=False)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    with open(save_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("[eval] 对比表已保存 ->", save_dir)
    return df
