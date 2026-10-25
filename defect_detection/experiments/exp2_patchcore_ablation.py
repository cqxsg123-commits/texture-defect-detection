"""实验 2：PatchCore 消融实验（计划书 5. 实验设计 - 实验 2）。

A. 单层 VGG 特征 vs 多层特征融合（评价指标 PixelAUROC）
B. 近邻数量 k 小范围网格搜索（计划书 4.3：平衡检测精度与推理速度）

用法：
    python experiments/exp2_patchcore_ablation.py [--retrain]
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
from src.models import PatchCoreModel
from src.utils import set_seed

# 实验 2 的消融配置组：(标签, layers, k)
ABLATION_GROUPS = [
    ("单层conv4_3_k5", ["conv4_3"], 5),
    ("多层融合_k5", ["conv3_3", "conv4_3"], 5),
    ("多层融合_k1", ["conv3_3", "conv4_3"], 1),
    ("多层融合_k3", ["conv3_3", "conv4_3"], 3),
    ("多层融合_k9", ["conv3_3", "conv4_3"], 9),
]


def run_experiment(retrain: bool = False) -> pd.DataFrame:
    """实验 2 主逻辑（Notebook 可直接调用），返回消融结果 DataFrame。"""
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
    for tag, layers, k in ABLATION_GROUPS:
        print(f"\n{'=' * 60}\n>>> PatchCore 消融：{tag}\n{'=' * 60}")
        # 每组消融单独创建训练加载器（DataLoader 为迭代器，不可复用）
        train_loader = get_train_loader(
            data_root,
            batch_size=cfg["dataset"]["batch_size"],
            num_workers=cfg["dataset"]["num_workers"],
            image_size=cfg["dataset"]["image_size"],
        )
        model = PatchCoreModel(
            device="auto",
            image_size=cfg["dataset"]["image_size"],
            layers=layers,
            k=k,
            coreset_ratio=cfg["models"]["patchcore"]["coreset_ratio"],
            stride=cfg["models"]["patchcore"]["stride"],
        )
        model = get_or_train(model, f"patchcore_ablation_{tag}.pth", train_loader,
                             retrain=retrain)
        res = evaluate_model(model, test_loader)
        res["model"] = tag
        results.append(res)
        release_memory(model)

    df = pd.DataFrame(results)
    df.to_csv(results_dir / "exp2_patchcore_ablation.csv", index=False, encoding="utf-8-sig")
    with open(results_dir / "exp2_patchcore_ablation.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("\n实验 2 结果：\n", df.to_string(index=False))
    print("实验 2 完成 ✔")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="忽略已有权重强制重训")
    args = parser.parse_args()
    run_experiment(retrain=args.retrain)


if __name__ == "__main__":
    main()
