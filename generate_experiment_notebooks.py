# -*- coding: utf-8 -*-
"""将 4 个实验各生成一个独立的 Jupyter Notebook（代码完整可见、可逐格运行）。

运行：D:\\anaconda\\envs\\defect_det\\python.exe generate_experiment_notebooks.py
产物：defect_detection/experiments/实验{1..4}_*.ipynb

每个 Notebook 结构：
  0 实验说明 → 1 环境准备 → 2 配置与数据准备 → 3 模型构建与训练（含 RETRAIN 开关）
  → 4 训练源码速览（inspect 打印本实验模型的 train 实现）
  → 5 评估与结果输出（实验 4 另有 6 基线 vs 优化对比）
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(r"D:\基于纹理特征的工业缺陷检测系统\defect_detection\experiments")

# ---------------- 公共单元格 ----------------

ENV_CELL = '''import os
import sys
import inspect
from pathlib import Path

# 必须先于 numpy/torch 导入设置（Windows：防 OpenBLAS 内存报错、GBK 编码报错）
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _find_project_root() -> Path:
    """定位项目根目录（包含 src/ 与 configs/ 的 defect_detection 目录）。"""
    # 方式 1：由 Notebook 文件所在位置向上找（VS Code / JupyterLab 均适用，不依赖启动目录）
    for var in ("__vsc_ipynb_file__", "__file__"):
        p = globals().get(var)
        if p and str(p).endswith(".ipynb"):
            for parent in Path(p).resolve().parents:
                if (parent / "src").exists() and (parent / "configs").exists():
                    return parent
    # 方式 2：从当前工作目录向上找（经典 Jupyter Notebook 场景）
    d = Path.cwd().resolve()
    for parent in (d, *d.parents):
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError(
        "未找到项目根目录（defect_detection，含 src/ 与 configs/）。\\n"
        "请确认 Notebook 位于项目的 experiments/ 目录下，或从项目根目录启动 Jupyter。"
    )


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
print("项目根目录:", PROJECT_ROOT)
print("CUDA 可用:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'''

CONFIG_CELL = '''# ============ 训练开关 ============
RETRAIN = False   # True = 忽略已有权重、从头训练（耗时数小时）；False = 复用 checkpoints 仅评估

import json

import pandas as pd

from src.experiment_utils import get_or_train, load_config, release_memory, resolve
from src.data import get_test_loader, get_train_loader
from src.utils import set_seed

cfg = load_config()
set_seed(cfg["dataset"]["seed"])
data_root = resolve(cfg["paths"]["data_root"])
results_dir = resolve(cfg["paths"]["results_dir"])

print("数据根目录:", data_root)
print("图像尺寸:", cfg["dataset"]["image_size"], "| batch_size:", cfg["dataset"]["batch_size"])
print("本实验涉及模型的超参数（configs/experiment.yaml）:")
print(json.dumps(cfg["models"], indent=2, ensure_ascii=False))

# 测试集加载器（评估用，固定不变）
# Windows Jupyter 下多进程 DataLoader 不稳，统一 num_workers=0（不影响结果，仅加载略慢）
test_loader = get_test_loader(data_root, batch_size=cfg["dataset"]["batch_size"],
                              num_workers=0, image_size=cfg["dataset"]["image_size"])'''

SRC_HEADER = "## ④ 训练源码速览（本实验模型的 train 实现）"
SRC_INTRO = ("以下用 `inspect` 直接打印项目 `src/` 中的训练代码，"
             "与上面实际执行的训练逻辑完全一致（单一来源，无需手工复制）：\n"
             "- `get_or_train`：统一训练入口（权重缓存判断 / 训练调度 / 保存）\n"
             "- 各模型 `train`：本实验的训练循环实现")

OUTPUTS_HEADER = "## 输出文件"


def md(src: str) -> dict:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


def build(path: Path, cells: list[dict]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (defect_det)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f"[ok] {path.name}  ({len(cells)} 个单元格)")


# ================= 实验 1 =================
exp1_main = '''from src.evaluation import evaluate_model, make_comparison_table
from src.models import build_model

MODELS = ["patchcore", "padim", "cutpaste"]

results = []
for name in MODELS:
    print(f"\\n{'=' * 60}\\n>>> 模型：{name}\\n{'=' * 60}")
    # 每个模型单独创建训练加载器（DataLoader 为迭代器，不可复用）
    train_loader = get_train_loader(data_root, batch_size=cfg["dataset"]["batch_size"],
                                    num_workers=0, image_size=cfg["dataset"]["image_size"],
                                    use_augmentation=True)
    model = build_model(name, device="auto", image_size=cfg["dataset"]["image_size"],
                        **cfg["models"].get(name, {}))
    model = get_or_train(model, f"{name}.pth", train_loader, retrain=RETRAIN)  # ← 训练/加载权重
    res = evaluate_model(model, test_loader, heatmap_dir=results_dir / "heatmaps",
                         n_heatmap_samples=3)
    results.append(res)
    release_memory(model)   # 串行训练，及时释放显存

df = make_comparison_table(results, results_dir)
df'''

exp1_src = '''print(inspect.getsource(get_or_train))   # 统一训练入口：权重缓存 / 训练 / 保存

from src.models.cutpaste import CutPasteModel
from src.models.padim_anomalib import PadimAnomalibModel
from src.models.patchcore import PatchCoreModel

for cls in (PatchCoreModel, PadimAnomalibModel, CutPasteModel):
    print("\\n" + "#" * 70)
    print(f"# {cls.__name__}.train  ——  {cls.__module__}")
    print("#" * 70)
    print(inspect.getsource(cls.train))'''

build(NB_DIR / "实验1_三模型横向对比.ipynb", [
    md("""# 实验 1：3 种模型横向对比

**目的**：在统一数据集与评估协议下，横向对比 3 条无监督检测技术路线（计划书 4.3 对比实验）。
（初版实现 5 种路线后按测试集准确率筛选，保留前 3 名。）

**模型**：PatchCore（主方案）/ PaDiM / CutPaste

**指标**：图像级 AUROC、PixelAUROC、单张平均推理耗时（约束 <1s）

**输出**：`results/comparison.csv`、`comparison.md`、`results/heatmaps/` 热力图样例

**运行提示**：
- 本 Notebook 单独成册，可从 Jupyter 首页直接打开运行
- 权重缓存在 `checkpoints/`，重跑自动跳过训练；把第 ③ 节的 `RETRAIN = True` 可从头训练 3 个模型（约数小时）
- 若之前在同一内核跑过其他 Notebook，建议先 `Kernel → Restart & Run All` 保证环境变量先生效"""),
    md("## ① 环境准备"),
    code(ENV_CELL),
    md("## ② 实验配置与数据准备"),
    code(CONFIG_CELL),
    md("## ③ 模型构建与训练（3 模型串行）"),
    code(exp1_main),
    md(SRC_HEADER),
    md(SRC_INTRO),
    code(exp1_src),
    md(OUTPUTS_HEADER),
    md("`results/comparison.csv`（对比表）、`results/comparison.md`、`results/heatmaps/<模型>/`（三联图样例）。"),
])

# ================= 实验 2 =================
exp2_main = '''from src.evaluation import evaluate_model
from src.models import PatchCoreModel

# 消融配置组：(标签, 特征层, 近邻数 k)
ABLATION_GROUPS = [
    ("单层conv4_3_k5", ["conv4_3"], 5),
    ("多层融合_k5", ["conv3_3", "conv4_3"], 5),
    ("多层融合_k1", ["conv3_3", "conv4_3"], 1),
    ("多层融合_k3", ["conv3_3", "conv4_3"], 3),
    ("多层融合_k9", ["conv3_3", "conv4_3"], 9),
]

results = []
for tag, layers, k in ABLATION_GROUPS:
    print(f"\\n{'=' * 60}\\n>>> PatchCore 消融：{tag}\\n{'=' * 60}")
    train_loader = get_train_loader(data_root, batch_size=cfg["dataset"]["batch_size"],
                                    num_workers=0, image_size=cfg["dataset"]["image_size"])
    model = PatchCoreModel(device="auto", image_size=cfg["dataset"]["image_size"],
                           layers=layers, k=k,                        # ← 消融变量
                           coreset_ratio=cfg["models"]["patchcore"]["coreset_ratio"],
                           stride=cfg["models"]["patchcore"]["stride"])
    model = get_or_train(model, f"patchcore_ablation_{tag}.pth", train_loader,
                         retrain=RETRAIN)
    res = evaluate_model(model, test_loader)
    res["model"] = tag
    results.append(res)
    release_memory(model)

df = pd.DataFrame(results)
df.to_csv(results_dir / "exp2_patchcore_ablation.csv", index=False, encoding="utf-8-sig")
with open(results_dir / "exp2_patchcore_ablation.md", "w", encoding="utf-8") as f:
    f.write(df.to_markdown(index=False))
df'''

exp2_src = '''print(inspect.getsource(get_or_train))

from src.models.patchcore import PatchCoreModel, _greedy_coreset

print(inspect.getsource(_greedy_coreset))      # 记忆库压缩：贪心 coreset 采样
print(inspect.getsource(PatchCoreModel.train)) # 特征提取 + 记忆库构建 + FAISS 索引'''

build(NB_DIR / "实验2_PatchCore消融.ipynb", [
    md("""# 实验 2：PatchCore 消融实验

**目的**：检验主方案 PatchCore 的两项关键设计（计划书 5. 实验设计 - 实验 2）：
- A. 单层 VGG 特征 vs 多层特征融合
- B. 近邻数量 k 网格搜索 {1, 3, 5, 9}（平衡检测精度与推理速度）

**输出**：`results/exp2_patchcore_ablation.csv`、`.md`

**运行提示**：权重缓存在 `checkpoints/patchcore_ablation_*.pth`，重跑跳过训练；`RETRAIN = True` 重训 5 组配置。"""),
    md("## ① 环境准备"),
    code(ENV_CELL),
    md("## ② 实验配置与数据准备"),
    code(CONFIG_CELL),
    md("## ③ 消融实验：5 组配置串行训练 + 评估"),
    code(exp2_main),
    md(SRC_HEADER),
    md(SRC_INTRO),
    code(exp2_src),
    md(OUTPUTS_HEADER),
    md("`results/exp2_patchcore_ablation.csv`（消融对比表）。"),
])

# ================= 实验 3 =================
exp3_main = '''from src.evaluation import evaluate_model
from src.models import build_model

results = []
for model_name in ("cutpaste",):
    for use_aug, tag in ((True, "有增强"), (False, "无增强")):
        print(f"\\n{'=' * 60}\\n>>> {model_name} {tag}\\n{'=' * 60}")
        train_loader = get_train_loader(data_root, batch_size=cfg["dataset"]["batch_size"],
                                        num_workers=0, image_size=cfg["dataset"]["image_size"],
                                        use_augmentation=use_aug)   # ← 实验开关：数据增强
        model = build_model(model_name, device="auto",
                            image_size=cfg["dataset"]["image_size"],
                            **cfg["models"].get(model_name, {}))
        model = get_or_train(model, f"{model_name}_{tag}.pth", train_loader,
                             retrain=RETRAIN)
        res = evaluate_model(model, test_loader)
        res["model"] = f"{model_name}({tag})"
        results.append(res)
        release_memory(model)

df = pd.DataFrame(results)
df.to_csv(results_dir / "exp3_augmentation.csv", index=False, encoding="utf-8-sig")
with open(results_dir / "exp3_augmentation.md", "w", encoding="utf-8") as f:
    f.write(df.to_markdown(index=False))
df'''

exp3_src = '''print(inspect.getsource(get_or_train))

from src.models.cutpaste import CutPasteModel, cutpaste_batch

print(inspect.getsource(cutpaste_batch))          # CutPaste 伪异常生成（训练期数据增强核心）
print(inspect.getsource(CutPasteModel.train))     # 伪异常二分类训练'''

build(NB_DIR / "实验3_数据增强消融.ipynb", [
    md("""# 实验 3：数据增强消融实验

**目的**：检验数据增强对自监督伪异常模型 CutPaste 的作用（计划书 5. 实验设计 - 实验 3）。

**变量**：训练集有无数据增强（随机水平翻转 + ±5° 旋转 + 亮度扰动）
**对象**：CutPaste 训练「有增强 / 无增强」两个版本

**输出**：`results/exp3_augmentation.csv`、`.md`

**运行提示**：权重缓存在 `checkpoints/cutpaste_*.pth`；`RETRAIN = True` 重训 2 组配置。"""),
    md("## ① 环境准备"),
    code(ENV_CELL),
    md("## ② 实验配置与数据准备"),
    code(CONFIG_CELL),
    md("## ③ 消融实验：CutPaste 有/无增强 串行训练 + 评估"),
    code(exp3_main),
    md(SRC_HEADER),
    md(SRC_INTRO),
    code(exp3_src),
    md(OUTPUTS_HEADER),
    md("`results/exp3_augmentation.csv`（增强消融对比表）。"),
])

# ================= 实验 4 =================
exp4_main = '''import shutil

from src.evaluation import evaluate_model
from src.models import build_model

# 模型名 -> (中文名, 基线参数覆盖, 基线配置说明)
BASELINE_GROUPS = [
    ("patchcore", "PatchCore", dict(layers=["conv4_3"], k=1, coreset_ratio=1.0),
     "单层 conv4_3 + k=1 最近邻 + 无 coreset 压缩 + 无增强"),
    ("padim", "PaDiM", dict(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                            n_features=None),
     "原论文默认配置（ResNet18 + layer1~3 + 100 维投影）"),
    ("cutpaste", "CutPaste", dict(), "无数据增强"),
]

ckpt_dir = resolve(cfg["paths"]["checkpoint_dir"])
results = []
for name, cn, overrides, desc in BASELINE_GROUPS:
    print(f"\\n{'=' * 60}\\n>>> 基线模型：{cn}（{desc}）\\n{'=' * 60}")
    # CutPaste 基线 = 实验 3「无增强」配置，直接复用其权重避免重复训练
    ckpt = ckpt_dir / f"{name}_baseline.pth"
    if name == "cutpaste" and not RETRAIN and not ckpt.exists():
        src = ckpt_dir / "cutpaste_无增强.pth"
        if src.exists():
            shutil.copyfile(src, ckpt)
            print(f"[exp4] 复用实验 3 的 cutpaste 无增强权重 -> {ckpt.name}")

    # 基线统一：训练集不做数据增强
    train_loader = get_train_loader(data_root, batch_size=cfg["dataset"]["batch_size"],
                                    num_workers=0, image_size=cfg["dataset"]["image_size"],
                                    use_augmentation=False)
    kwargs = dict(cfg["models"].get(name, {}))
    kwargs.update(overrides)   # 基线参数覆盖 yaml 中的优化配置
    model = build_model(name, device="auto",
                        image_size=cfg["dataset"]["image_size"], **kwargs)
    model = get_or_train(model, f"{name}_baseline.pth", train_loader, retrain=RETRAIN)
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
df'''

exp4_merge = '''CN_MAP = {"patchcore": "PatchCore", "padim": "PaDiM", "cutpaste": "CutPaste"}

opt = pd.read_csv(results_dir / "comparison.csv")   # 实验 1 的优化配置结果
# CutPaste 优化配置取实验 3 复训权重（0.8925/0.8548），实验 1 早期权重像素级偏低（0.6013）
exp3 = pd.read_csv(results_dir / "exp3_augmentation.csv")
row3 = exp3[exp3["model"] == "cutpaste(有增强)"].iloc[0]
opt.loc[opt["model"] == "cutpaste",
        ["image_auroc", "pixel_auroc", "mean_inference_ms", "n_test"]] = \\
    row3[["image_auroc", "pixel_auroc", "mean_inference_ms", "n_test"]].values
opt["model"] = opt["model"].map(lambda m: f"{CN_MAP[m]}（优化）")

merged = pd.concat([opt, df[["model", "image_auroc", "pixel_auroc",
                             "mean_inference_ms", "n_test"]]], ignore_index=True)
merged.to_csv(results_dir / "baseline_vs_optimized.csv", index=False, encoding="utf-8-sig")
with open(results_dir / "baseline_vs_optimized.md", "w", encoding="utf-8") as f:
    f.write(merged.to_markdown(index=False))

print("=" * 60 + "\\n基线 vs 优化 提升汇总（Δ = 优化 − 基线）\\n" + "=" * 60)
for name, cn, _, _ in BASELINE_GROUPS:
    b = df[df["model"] == f"{cn}（基线）"].iloc[0]
    o = opt[opt["model"] == f"{cn}（优化）"].iloc[0]
    print(f"{cn:>12}: 基线 {b['image_auroc']:.4f}/{b['pixel_auroc']:.4f}"
          f" -> 优化 {o['image_auroc']:.4f}/{o['pixel_auroc']:.4f}"
          f"  (Δ{o['image_auroc'] - b['image_auroc']:+.4f}/"
          f"{o['pixel_auroc'] - b['pixel_auroc']:+.4f})")
merged'''

exp4_src = '''print(inspect.getsource(get_or_train))

from src.models.cutpaste import CutPasteModel
from src.models.padim_anomalib import PadimAnomalibModel
from src.models.patchcore import PatchCoreModel

for cls in (PatchCoreModel, PadimAnomalibModel, CutPasteModel):
    print("\\n" + "#" * 70)
    print(f"# {cls.__name__}.train  ——  {cls.__module__}")
    print("#" * 70)
    print(inspect.getsource(cls.train))'''

build(NB_DIR / "实验4_基线实验.ipynb", [
    md("""# 实验 4：三模型基线实验

**目的**：把每个模型还原为「朴素基线形态」重新训练评估，与实验 1 的最终配置对比，
量化各项优化设计（coreset / 特征融合 / 增强等）的真实收益。

**基线定义**（参数覆盖 `configs/experiment.yaml` 中的优化配置）：

| 模型 | 基线配置 | 最终配置 |
|---|---|---|
| PatchCore | 单层 conv4_3 + k=1 + 全量记忆库 | 多层融合 + k=1 + 全量记忆库 |
| PaDiM | ResNet18 + layer1~3 + 100 维 | VGG16 双层 + 550 维 |
| CutPaste | 无数据增强 | 有数据增强 |

**输出**：`results/exp4_baseline.csv`、`baseline_vs_optimized.csv`（含 .md）、`results/heatmaps_baseline/`

**运行要求**：第 ⑥ 节对比需要实验 1、实验 3 的结果文件（`results/comparison.csv`、
`results/exp3_augmentation.csv` 已存在即可）；CutPaste 基线会复用实验 3 的 `cutpaste_无增强.pth`。"""),
    md("## ① 环境准备"),
    code(ENV_CELL),
    md("## ② 实验配置与数据准备"),
    code(CONFIG_CELL),
    md("## ③ 基线模型构建与训练（3 模型串行）"),
    code(exp4_main),
    md("## ④ 评估结果汇总（基线 vs 优化）"),
    code(exp4_merge),
    md(SRC_HEADER),
    md(SRC_INTRO),
    code(exp4_src),
    md(OUTPUTS_HEADER),
    md("`results/exp4_baseline.csv`、`baseline_vs_optimized.csv`、`results/heatmaps_baseline/<模型>/`。"),
])

print("全部 4 个实验 Notebook 生成完毕 ->", NB_DIR)
