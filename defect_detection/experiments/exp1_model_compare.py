"""实验 1：3 模型横向对比（计划书 4.3 对比实验）。

按准确率筛选后的最终模型集合：PatchCore（主方案）/ PaDiM / CutPaste
指标：图像级 AUROC、PixelAUROC、单张平均推理耗时。
输出：results/comparison.csv + comparison.md + results/heatmaps/ 热力图样例。

用法：
    python experiments/exp1_model_compare.py [--models patchcore,padim] [--retrain]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从项目根或 experiments/ 目录运行都能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import RESULTS_DIR
from src.data import get_test_loader, get_train_loader
from src.evaluation import evaluate_model, make_comparison_table
from src.experiment_utils import get_or_train, load_config, release_memory, resolve
from src.models import build_model
from src.utils import set_seed


def run_experiment(models: list[str] | None = None, retrain: bool = False):
    """实验 1 主逻辑（Notebook 可直接调用），返回对比表 DataFrame。"""
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

    model_names = models or ["patchcore", "padim", "cutpaste"]

    results = []
    for name in model_names:
        print(f"\n{'=' * 60}\n>>> 模型：{name}\n{'=' * 60}")
        # 每个模型单独创建训练加载器（DataLoader 为迭代器，不可复用）
        train_loader = get_train_loader(
            data_root,
            batch_size=cfg["dataset"]["batch_size"],
            num_workers=cfg["dataset"]["num_workers"],
            image_size=cfg["dataset"]["image_size"],
            use_augmentation=True,
        )
        model = build_model(name, device="auto", image_size=cfg["dataset"]["image_size"],
                            **cfg["models"].get(name, {}))
        model = get_or_train(model, f"{name}.pth", train_loader, retrain=retrain)
        res = evaluate_model(
            model, test_loader, heatmap_dir=results_dir / "heatmaps", n_heatmap_samples=3
        )
        results.append(res)
        release_memory(model)  # 串行训练，及时释放显存

    df = make_comparison_table(results, results_dir)
    print("\n实验 1 完成 ✔")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None, help="逗号分隔，默认全部 3 个")
    parser.add_argument("--retrain", action="store_true", help="忽略已有权重强制重训")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_experiment(models=models, retrain=args.retrain)


if __name__ == "__main__":
    main()
