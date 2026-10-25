"""实验 3：数据增强消融实验（计划书 5. 实验设计 - 实验 3）。

CutPaste 在「有/无数据增强」两种条件下训练对比（指标 AUROC）。
有增强：随机水平翻转 + ±5°旋转 + 亮度扰动（仅训练集）。

用法：
    python experiments/exp3_augmentation.py [--retrain]
"""

from __future__ import annotations

import argparse
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


def run_experiment(retrain: bool = False) -> pd.DataFrame:
    """实验 3 主逻辑（Notebook 可直接调用），返回消融结果 DataFrame。"""
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
    for model_name in ("cutpaste",):
        for use_aug, tag in ((True, "有增强"), (False, "无增强")):
            print(f"\n{'=' * 60}\n>>> {model_name} {tag}\n{'=' * 60}")
            train_loader = get_train_loader(
                data_root,
                batch_size=cfg["dataset"]["batch_size"],
                num_workers=cfg["dataset"]["num_workers"],
                image_size=cfg["dataset"]["image_size"],
                use_augmentation=use_aug,  # 实验开关
            )
            model = build_model(model_name, device="auto",
                                image_size=cfg["dataset"]["image_size"],
                                **cfg["models"].get(model_name, {}))
            model = get_or_train(model, f"{model_name}_{tag}.pth", train_loader,
                                 retrain=retrain)
            res = evaluate_model(model, test_loader)
            res["model"] = f"{model_name}({tag})"
            results.append(res)
            release_memory(model)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "exp3_augmentation.csv", index=False, encoding="utf-8-sig")
    with open(results_dir / "exp3_augmentation.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("\n实验 3 结果：\n", df.to_string(index=False))
    print("实验 3 完成 ✔")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="忽略已有权重强制重训")
    args = parser.parse_args()
    run_experiment(retrain=args.retrain)


if __name__ == "__main__":
    main()
