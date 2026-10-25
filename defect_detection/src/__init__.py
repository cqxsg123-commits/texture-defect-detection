"""基于纹理特征的工业缺陷检测系统 — 核心代码包。"""

from pathlib import Path

# 项目根目录（defect_detection/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 常用目录
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
