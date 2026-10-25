# -*- coding: utf-8 -*-
"""实验 5：PatchCore 主方案优化搜索（全量记忆库下 k 值与特征层组合）。

背景：实验 4 证明朴素基线（k=1 + 全量记忆库，0.9081/0.9013）远优于
优化配置（coreset 0.1 + k=5，0.6994/0.5395）——220 张小训练集上
coreset 压缩损失代表性、k=5 平均稀释最近邻信号。

本实验在全量记忆库（coreset_ratio=1.0）、无数据增强的已验证最优状态下，
继续搜索 k ∈ {1,3,5} 与特征层（单层 conv4_3 / 多层融合）组合，
确定主方案的最终配置。

输出：results/exp5_patchcore_optimize.csv/md
用法：python experiments/exp5_patchcore_optimize.py [--retrain]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从项目根或 experiments/ 目录运行都能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import RESULTS_DIR
from src.data import get_test_loader, get_train_loader
from src.experiment_utils import get_or_train, load_config, release_memory, resolve
from src.metrics import compute_image_auroc, compute_pixel_auroc
from src.models import PatchCoreModel
from src.utils import set_seed

# (标签, 特征层, k, 复用已有权重文件名或 None)
SWEEP_GROUPS = [
    ("conv4_3_k1", ["conv4_3"], 1, "patchcore_baseline.pth"),  # 复用实验 4 基线权重
    ("conv4_3_k3", ["conv4_3"], 3, None),
    ("conv4_3_k5", ["conv4_3"], 5, None),
    ("多层融合_k1", ["conv3_3", "conv4_3"], 1, None),
    ("多层融合_k3", ["conv3_3", "conv4_3"], 3, None),
    ("多层融合_k5", ["conv3_3", "conv4_3"], 5, None),
]


def evaluate_auroc_only(model: PatchCoreModel, test_loader) -> tuple[float, float]:
    """仅计算图像级 / 像素级 AUROC（不做 20 次重复耗时测量，加速搜索）。

    协议与 src/evaluation.py 完全一致：图像级得分 = 原始热力图最大值；
    像素级 AUROC = 逐图 min-max 归一化后全局展平。
    """
    raw_maps, masks, labels = [], [], []
    for batch in test_loader:
        maps = model.predict(batch["image"].to(model.device))  # (B,H,W)
        raw_maps.append(maps)
        masks.append(batch["mask"].squeeze(1).numpy())
        labels.append(batch["label"].numpy())
    raw_maps = np.concatenate(raw_maps, axis=0)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    scores = raw_maps.max(axis=(1, 2))
    return (compute_image_auroc(scores, labels),
            compute_pixel_auroc(raw_maps, masks))


def run_experiment(retrain: bool = False) -> pd.DataFrame:
    cfg = load_config()
    set_seed(cfg["dataset"]["seed"])
    data_root = resolve(cfg["paths"]["data_root"])
    results_dir = resolve(cfg["paths"]["results_dir"])

    test_loader = get_test_loader(
        data_root,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        image_size=cfg["dataset"]["image_size"],
    )

    results = []
    for tag, layers, k, reuse_ckpt in SWEEP_GROUPS:
        print(f"\n{'=' * 60}\n>>> PatchCore 优化搜索：{tag}\n{'=' * 60}")
        train_loader = get_train_loader(
            data_root,
            batch_size=cfg["dataset"]["batch_size"],
            num_workers=cfg["dataset"]["num_workers"],
            image_size=cfg["dataset"]["image_size"],
            use_augmentation=False,   # 沿用实验 4 已验证的最优训练状态（无增强）
        )
        model = PatchCoreModel(
            device="auto",
            image_size=cfg["dataset"]["image_size"],
            layers=layers,
            k=k,
            coreset_ratio=1.0,        # 全量记忆库：无 coreset 压缩
            stride=cfg["models"]["patchcore"]["stride"],
        )
        ckpt = reuse_ckpt or f"patchcore_opt_{tag}.pth"
        model = get_or_train(model, ckpt, train_loader, retrain=retrain)
        img_auc, pix_auc = evaluate_auroc_only(model, test_loader)
        results.append({"model": tag, "image_auroc": img_auc,
                        "pixel_auroc": pix_auc, "n_test": 115})
        print(f"[exp5] {tag}: 图像级 {img_auc:.4f} / 像素级 {pix_auc:.4f}")
        release_memory(model)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "exp5_patchcore_optimize.csv", index=False,
              encoding="utf-8-sig")
    with open(results_dir / "exp5_patchcore_optimize.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("\n实验 5 结果：\n", df.to_string(index=False))
    print("\n实验 5 完成 ✔")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="忽略已有权重强制重训")
    args = parser.parse_args()
    run_experiment(retrain=args.retrain)


if __name__ == "__main__":
    main()
