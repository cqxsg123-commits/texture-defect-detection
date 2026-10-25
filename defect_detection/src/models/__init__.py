"""模型模块：3 种无监督异常检测模型（按准确率筛选后的最终模型集合）。

- PatchCore / CutPaste：自定义 PyTorch 实现
- PaDiM：Anomalib 2.6.2 封装（统一接口适配）

注：初版实验实现并对比了 5 种技术路线（含 AutoEncoder、DeepSVDD），
按测试集准确率筛选后仅保留前 3 名：PaDiM（0.9668）、PatchCore（0.9477）、
CutPaste（0.9003）。AutoEncoder 与 DeepSVDD 的源码移至 legacy/ 目录备查。
"""

from .base import BaseAnomalyModel
from .cutpaste import CutPasteModel
from .padim_anomalib import PadimAnomalibModel
from .patchcore import PatchCoreModel

MODEL_REGISTRY = {
    "patchcore": PatchCoreModel,
    "padim": PadimAnomalibModel,
    "cutpaste": CutPasteModel,
}


def build_model(name: str, **kwargs) -> BaseAnomalyModel:
    """按名称构建模型。"""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {name}，可选：{list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "BaseAnomalyModel",
    "PatchCoreModel",
    "CutPasteModel",
    "PadimAnomalibModel",
    "MODEL_REGISTRY",
    "build_model",
]
