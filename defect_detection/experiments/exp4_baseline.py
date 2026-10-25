"""实验 4：三模型基线实验。

将每个模型还原为最朴素形态（去除本项目加入的优化手段），在完全相同的
数据与评估协议下训练、评估，与实验 1 的最终配置结果对比，
量化各项优化手段的收益。

基线定义：
- PatchCore  ：k=1 最近邻 + 单层 conv4_3 + 无 coreset 压缩（全量记忆库）+ 无数据增强
- PaDiM      ：原论文默认配置（ResNet18 + layer1~3 + 100 维随机投影）
- CutPaste   ：无数据增强（与实验 3 的「无增强」配置一致）

输出：results/exp4_baseline.csv/md（基线结果）、
      results/baseline_vs_optimized.csv/md（基线 vs 优化对比）、
      results/heatmaps_baseline/（基线模型热力图样例）。

用法：
    python experiments/exp4_baseline.py [--retrain]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 保证从项目根或 experiments/ 目录运行都能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import RESULTS_DIR
from src.data import get_test_loader, get_train_loader
from src.evaluation import evaluate_model
from src.experiment_utils import get_or_train, load_config, release_memory, resolve
from src.models import build_model
from src.utils import set_seed

# 模型名 -> (中文名, 基线构造参数覆盖, 基线配置说明)
BASELINE_GROUPS = [
    ("patchcore", "PatchCore", dict(layers=["conv4_3"], k=1, coreset_ratio=1.0),
     "单层 conv4_3 + k=1 最近邻 + 无 coreset 压缩 + 无增强"),
    ("padim", "PaDiM", dict(backbone="resnet18",
                            layers=["layer1", "layer2", "layer3"], n_features=None),
     "原论文默认配置（ResNet18 + layer1~3 + 100 维投影）"),
    ("cutpaste", "CutPaste", dict(),
     "无数据增强"),
]

CN_MAP = {"patchcore": "PatchCore", "padim": "PaDiM", "cutpaste": "CutPaste"}


def run_experiment(retrain: bool = False) -> pd.DataFrame:
    """实验 4 主逻辑，返回「基线 vs 优化」对比 DataFrame。"""
    cfg = load_config()
    set_seed(cfg["dataset"]["seed"])
    data_root = resolve(cfg["paths"]["data_root"])
    results_dir = resolve(cfg["paths"]["results_dir"])
    ckpt_dir = resolve(cfg["paths"]["checkpoint_dir"])

    test_loader = get_test_loader(
        data_root,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=cfg["dataset"]["num_workers"],
        image_size=cfg["dataset"]["image_size"],
    )

    results = []
    for name, cn, overrides, desc in BASELINE_GROUPS:
        print(f"\n{'=' * 60}\n>>> 基线模型：{cn}（{desc}）\n{'=' * 60}")
        # CutPaste 基线配置与实验 3「无增强」完全一致，直接复用其权重避免重复训练
        ckpt = ckpt_dir / f"{name}_baseline.pth"
        if name == "cutpaste" and not retrain and not ckpt.exists():
            src = ckpt_dir / "cutpaste_无增强.pth"
            if src.exists():
                shutil.copyfile(src, ckpt)
                print(f"[exp4] 复用实验 3 的 cutpaste 无增强权重 -> {ckpt.name}")

        # 基线统一：训练集不做数据增强
        train_loader = get_train_loader(
            data_root,
            batch_size=cfg["dataset"]["batch_size"],
            num_workers=cfg["dataset"]["num_workers"],
            image_size=cfg["dataset"]["image_size"],
            use_augmentation=False,
        )
        kwargs = dict(cfg["models"].get(name, {}))
        kwargs.update(overrides)  # 基线参数覆盖 yaml 优化配置
        model = build_model(name, device="auto",
                            image_size=cfg["dataset"]["image_size"], **kwargs)
        model = get_or_train(model, f"{name}_baseline.pth", train_loader, retrain=retrain)
        res = evaluate_model(model, test_loader,
                             heatmap_dir=results_dir / "heatmaps_baseline",
                             n_heatmap_samples=3)
        res["model"] = f"{cn}（基线）"
        results.append(res)
        release_memory(model)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "exp4_baseline.csv", index=False, encoding="utf-8-sig")
    with open(results_dir / "exp4_baseline.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    # ---- 与实验 1 优化结果合并对比 ----
    opt = pd.read_csv(results_dir / "comparison.csv")
    # CutPaste 的优化配置「有增强」在实验 3 中复训（0.8925/0.8548），
    # 实验 1 的早期权重像素级偏低（0.6013），对比时取复训结果
    exp3 = pd.read_csv(results_dir / "exp3_augmentation.csv")
    row3 = exp3[exp3["model"] == "cutpaste(有增强)"].iloc[0]
    opt.loc[opt["model"] == "cutpaste",
            ["image_auroc", "pixel_auroc", "mean_inference_ms", "n_test"]] = \
        row3[["image_auroc", "pixel_auroc", "mean_inference_ms", "n_test"]].values
    opt["model"] = opt["model"].map(lambda m: f"{CN_MAP[m]}（优化）")
    merged = pd.concat([opt, df[["model", "image_auroc", "pixel_auroc",
                                 "mean_inference_ms", "n_test"]]], ignore_index=True)
    merged.to_csv(results_dir / "baseline_vs_optimized.csv", index=False, encoding="utf-8-sig")
    with open(results_dir / "baseline_vs_optimized.md", "w", encoding="utf-8") as f:
        f.write(merged.to_markdown(index=False))

    # ---- 打印每个模型的提升 ----
    print("\n" + "=" * 60 + "\n基线 vs 优化 提升汇总\n" + "=" * 60)
    for name, cn, _, _ in BASELINE_GROUPS:
        b = df[df["model"] == f"{cn}（基线）"].iloc[0]
        o = opt[opt["model"] == f"{cn}（优化）"].iloc[0]
        print(f"{cn:>12}: 基线 {b['image_auroc']:.4f}/{b['pixel_auroc']:.4f}"
              f" -> 优化 {o['image_auroc']:.4f}/{o['pixel_auroc']:.4f}"
              f"  (Δ{o['image_auroc'] - b['image_auroc']:+.4f}/"
              f"{o['pixel_auroc'] - b['pixel_auroc']:+.4f})")
    print("\n实验 4 完成 ✔")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="忽略已有权重强制重训")
    args = parser.parse_args()
    run_experiment(retrain=args.retrain)


if __name__ == "__main__":
    main()
