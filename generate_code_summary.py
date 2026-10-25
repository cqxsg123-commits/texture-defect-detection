# -*- coding: utf-8 -*-
"""生成《实验完整代码汇总.md》：把 defect_detection/ 当前源码按模块分节汇总。

运行：D:\\anaconda\\envs\\defect_det\\python.exe generate_code_summary.py
产物：defect_detection/实验完整代码汇总.md（课程设计报告附录引用文件）
"""
from pathlib import Path

ROOT = Path(r"D:\基于纹理特征的工业缺陷检测系统")
DD = ROOT / "defect_detection"
OUT = DD / "实验完整代码汇总.md"

HEADER = """# 基于纹理特征的工业缺陷检测系统 —— 实验完整代码汇总

人工智能综合实践课程设计 · 课题编号 2 · 陈宇 202305630103

> 本文件汇总了项目五个实验的**全部可运行代码**，与 `defect_detection/` 目录下的
> 实际文件一一对应，可直接作为课程设计报告附录。
>
> 依赖关系：每个实验脚本（`experiments/expN_*.py`）是入口，实际逻辑调用
> `src/` 公共模块（数据集、模型、评估、指标）。因此「实验完整代码」=
> 实验脚本 + 第 0 部分列出的全部公共模块。
>
> 模型筛选说明：初版实现并对比 5 种技术路线，按 metal_nut 测试集准确率保留前 3 名
> （PaDiM 0.9668 / PatchCore 0.9477 / CutPaste 0.9003）；AutoEncoder 与 DeepSVDD
> 源码归档于 `legacy/`（本文件附录备查），其权重与热力图已删除。
"""

# (小节标题, 相对 defect_detection 的文件路径, 语言, 一句话说明)
SECTIONS = [
    ("第 0 部分：实验公共模块（五个实验共用）", [
        ("0.1", "configs/experiment.yaml", "yaml", "实验配置"),
        ("0.2", "src/__init__.py", "python", "包入口与常用目录"),
        ("0.3", "src/utils.py", "python", "随机种子、设备、热力图可视化"),
        ("0.4", "src/metrics.py", "python", "评估指标"),
        ("0.5", "src/evaluation.py", "python", "统一评估流程"),
        ("0.6", "src/experiment_utils.py", "python", "实验公共工具"),
        ("0.7", "src/data/__init__.py", "python", ""),
        ("0.8", "src/data/dataset.py", "python", "数据集模块"),
        ("0.9", "src/data/augmentations.py", "python", "数据增强"),
        ("0.10", "src/data/extract_mvtec.py", "python", "数据集解压"),
        ("0.11", "src/models/__init__.py", "python", "模型注册表（3 模型）"),
        ("0.12", "src/models/base.py", "python", "模型统一接口"),
        ("0.13", "src/models/patchcore.py", "python", "模型 1：PatchCore 主方案"),
        ("0.14", "src/models/cutpaste.py", "python", "模型 2：CutPaste 自监督"),
        ("0.15", "src/models/padim_anomalib.py", "python", "模型 3：PaDiM / Anomalib 适配层"),
    ]),
    ("第 1 部分：实验 1 —— 3 模型横向对比", [
        ("1.1", "experiments/exp1_model_compare.py", "python", ""),
        ("1.2", "results/comparison.md", "markdown", "实验 1 结果"),
    ]),
    ("第 2 部分：实验 2 —— PatchCore 消融实验", [
        ("2.1", "experiments/exp2_patchcore_ablation.py", "python", ""),
        ("2.2", "results/exp2_patchcore_ablation.md", "markdown", "实验 2 结果"),
    ]),
    ("第 3 部分：实验 3 —— 数据增强消融实验（CutPaste）", [
        ("3.1", "experiments/exp3_augmentation.py", "python", ""),
        ("3.2", "results/exp3_augmentation.md", "markdown", "实验 3 结果"),
    ]),
    ("第 4 部分：实验 4 —— 三模型基线实验", [
        ("4.1", "experiments/exp4_baseline.py", "python", ""),
        ("4.2", "results/exp4_baseline.md", "markdown", "实验 4 结果"),
    ]),
    ("第 5 部分：实验 5 —— PatchCore 主方案优化搜索", [
        ("5.1", "experiments/exp5_patchcore_optimize.py", "python", ""),
        ("5.2", "results/exp5_patchcore_optimize.md", "markdown", "实验 5 结果"),
    ]),
    ("第 6 部分：Gradio 演示系统（三模型推理 + 质检判定）", [
        ("6.1", "app.py", "python", ""),
        ("6.2", "scripts/compute_auto_threshold.py", "python", "自动判定阈值（训练集统计 + 测试集验证）"),
    ]),
    ("第 7 部分：一键运行全部实验", [
        ("7.1", "run_all_experiments.py", "python", ""),
    ]),
    ("附录 A：Notebook 版实验入口", [
        ("A.1", "experiments/experiments.ipynb", "notebook", "实验 1~5 的 Notebook 版入口"),
    ]),
    ("附录 B：已移除模型源码归档（legacy/，备查）", [
        ("B.1", "legacy/autoencoder.py", "python", "AutoEncoder 重建基线（0.5293，已移除）"),
        ("B.2", "legacy/deep_svdd.py", "python", "DeepSVDD 深度单分类（0.4736，已移除）"),
    ]),
]

NOTEBOOK_PTR = """> Notebook 为生成式文件，代码与第 1~5 部分实验脚本一致，此处不重复展开。
> 另附四份分实验 Notebook：`实验1_三模型横向对比.ipynb`、`实验2_PatchCore消融.ipynb`、
> `实验3_数据增强消融.ipynb`、`实验4_基线实验.ipynb`（由 `generate_experiment_notebooks.py` 生成）。
"""


def _code_block(text: str, lang: str) -> str:
    return f"```{lang}\n{text.rstrip()}\n```\n"


def _ipynb_summary(path: Path) -> str:
    """Notebook 不展开代码，仅列单元格结构与关键入口。"""
    import json

    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    lines = [f"共 {len(cells)} 个单元格："]
    for i, c in enumerate(cells):
        src = "".join(c.get("source", [])).strip()
        first = src.splitlines()[0] if src else ""
        tag = "md" if c.get("cell_type") == "markdown" else "py"
        lines.append(f"{i + 1:>3}. [{tag}] {first[:80]}")
    return "\n".join(lines) + "\n"


parts = [HEADER]
for part_title, entries in SECTIONS:
    parts.append(f"---\n\n## {part_title}\n")
    for num, rel, lang, note in entries:
        path = DD / rel
        title = f"`{rel}`" + (f"（{note}）" if note else "")
        parts.append(f"### {num} 文件：{title}\n")
        if lang == "notebook":
            parts.append(_ipynb_summary(path))
            parts.append(NOTEBOOK_PTR)
        else:
            parts.append(_code_block(path.read_text(encoding="utf-8"), lang))

OUT.write_text("\n".join(parts), encoding="utf-8")
print("代码汇总已生成 ->", OUT)
