# -*- coding: utf-8 -*-
"""一键运行全部实验（实验 1 ~ 实验 4）。

用法（VS Code / 终端，需在 defect_det 环境）：
    python run_all_experiments.py

用法（Jupyter）：
    %run run_all_experiments.py

说明：
- checkpoints/ 下已有对应权重时自动跳过训练、仅重新评估（约 30~40 分钟）；
  需从头训练加 --retrain（耗时数小时，慎用）。
- 结果写入 results/：CSV + Markdown 对比表 + 热力图样例（实验报告 / PPT 素材）。
- 实验 4 的 CutPaste 对比依赖实验 3 的复训权重与结果表，因此固定按 1→2→3→4 顺序执行。
"""
import os
import sys
from pathlib import Path

# ---- 必须先于 numpy/torch 导入设置（Windows：防 OpenBLAS 内存报错、GBK 编码报错）----
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))   # 保证在任意目录 / Jupyter 中都能 import experiments 与 src

# 输出重定向到文件时防止 GBK 编码报错（Notebook 内核下跳过，避免干扰其输出流）
if "IPython" not in sys.modules:
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

import time

import pandas as pd

from experiments.exp1_model_compare import run_experiment as exp1
from experiments.exp2_patchcore_ablation import run_experiment as exp2
from experiments.exp3_augmentation import run_experiment as exp3
from experiments.exp4_baseline import run_experiment as exp4


def main() -> None:
    retrain = "--retrain" in sys.argv
    results_dir = PROJECT_ROOT / "results"
    print("=" * 72)
    print("基于纹理特征的工业缺陷检测系统 · 全部实验（实验 1 ~ 实验 4）")
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"训练模式：{'强制重训（--retrain）' if retrain else '权重存在则跳过训练，仅评估'}")
    print("=" * 72)

    t0 = time.time()
    print("\n>>> 实验 1：3 模型横向对比（图像级 AUROC / PixelAUROC / 推理耗时）")
    exp1(retrain=retrain)

    print("\n>>> 实验 2：PatchCore 特征与 k 值消融")
    exp2(retrain=retrain)

    print("\n>>> 实验 3：数据增强消融（CutPaste）")
    exp3(retrain=retrain)

    print("\n>>> 实验 4：三模型基线实验（基线 vs 最终配置）")
    exp4(retrain=retrain)

    print("\n" + "=" * 72)
    print(f"全部实验完成，总耗时 {time.time() - t0:.0f} s")
    print("=" * 72)
    for f in ["comparison.csv", "exp2_patchcore_ablation.csv", "exp3_augmentation.csv",
              "exp4_baseline.csv", "baseline_vs_optimized.csv"]:
        p = results_dir / f
        if p.exists():
            print(f"\n--- {f} ---")
            print(pd.read_csv(p).to_string(index=False))


if __name__ == "__main__":
    main()
