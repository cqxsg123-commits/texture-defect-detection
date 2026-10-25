"""模型统一接口：所有模型实现 train / predict，输出统一的异常热力图与得分。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class BaseAnomalyModel(ABC):
    """无监督异常检测模型基类。

    子类约定：
    - train(train_loader)：仅使用正常样本训练
    - predict_image(image)：输入单张归一化图像 (1,3,H,W) -> 原始异常热力图 (H,W) numpy
    - save / load：权重持久化
    """

    name: str = "base"

    def __init__(self, device: torch.device | str = "auto", image_size: int = 256):
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = image_size

    @abstractmethod
    def train(self, train_loader) -> None:
        """训练模型（只允许正常样本）。"""

    @abstractmethod
    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        """单张图像推理。

        Args:
            image: (1,3,H,W) 归一化张量。

        Returns:
            原始异常热力图 (H,W) numpy，值越大越异常。
        """

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """批量推理 -> (B,H,W) 原始异常热力图。"""
        maps = []
        for i in range(images.shape[0]):
            maps.append(self.predict_image(images[i : i + 1].to(self.device)))
        return np.stack(maps)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state_dict()
        torch.save(state, path)
        print(f"[{self.name}] 权重已保存 -> {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"权重文件不存在：{path}")
        state = torch.load(path, map_location=self.device)
        self._load_state_dict(state)
        print(f"[{self.name}] 权重已加载 <- {path}")

    # ---- 子类可选覆写 ----
    def _state_dict(self) -> dict:
        raise NotImplementedError

    def _load_state_dict(self, state: dict) -> None:
        raise NotImplementedError
