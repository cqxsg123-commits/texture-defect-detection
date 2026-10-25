"""实验脚本公共工具：配置加载、路径解析、模型训练/加载逻辑。"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src import CHECKPOINT_DIR, CONFIG_DIR, PROJECT_ROOT, RESULTS_DIR


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 实验配置。"""
    path = Path(path) if path else CONFIG_DIR / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: str | Path) -> Path:
    """相对路径 -> 项目根绝对路径。"""
    root = Path(root)
    return root if root.is_absolute() else PROJECT_ROOT / root


def get_or_train(model, checkpoint_name: str | None, train_loader, retrain: bool = False):
    """已有权重则直接加载，否则训练并保存（--retrain 强制重训）。"""
    if checkpoint_name and not retrain:
        ckpt = resolve(CHECKPOINT_DIR) / checkpoint_name
        if ckpt.exists():
            model.load(ckpt)
            return model
    model.train(train_loader)
    if checkpoint_name:
        model.save(resolve(CHECKPOINT_DIR) / checkpoint_name)
    return model


def release_memory(*objs) -> None:
    """释放模型显存（计划书风险应对：串行训练、及时释放）。"""
    for o in objs:
        del o
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
