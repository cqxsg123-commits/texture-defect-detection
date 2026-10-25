"""PaDiM 模型（计划书模型 4）— Anomalib 2.6.2 封装。

原理：冻结预训练骨干网络，统计正常样本多尺度特征的均值与协方差，
构建多维高斯分布；推理时计算马氏距离得到像素级异常图。

说明：本模块是 3 个模型中唯一依托 Anomalib 库实现的对照模型，
其预测结果通过 predict_dataset() 适配到本项目的统一评估流程。
"""

from __future__ import annotations

import os
from pathlib import Path

# timm 预训练权重托管在 HuggingFace Hub，国内需镜像（必须在 huggingface_hub 导入前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 缓解 8GB 笔记本 GPU 的显存碎片化（PaDiM 协方差计算峰值显存高）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from .base import BaseAnomalyModel

# Anomalib 的 validate_path 仅接受纯 ASCII 路径（^[\x20-\x7E]+$），
# 本项目路径含中文，故在用户目录建立 NTFS junction（目录链接）指向真实数据根
_ASCII_LINK = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "mvtec_anomaly_detection"


def _ensure_ascii_link(real_root: Path) -> Path:
    """确保存在指向 real_root 的 ASCII 目录 junction（幂等，无需管理员权限）。

    注意：必须用 ``mklink /J`` 创建 NTFS junction；CreateSymbolicLinkW 创建的是
    目录符号链接，普通 Windows 用户无权限，会静默得到普通目录。
    """
    link = _ASCII_LINK
    # 功能性检查：junction 指向的数据目录必须可见（Python 对中文目标的
    # junction is_symlink() 不可靠，故直接验证数据子目录）
    if (link / "metal_nut").is_dir():
        return link
    if link.exists():
        import shutil

        shutil.rmtree(link)  # 清理历史残留（普通目录 / 下载残留）
    link.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(real_root.resolve())],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not (link / "metal_nut").is_dir():
        raise OSError(f"创建目录 junction 失败：{link} -> {real_root} ({result.stdout})")
    print(f"[padim] 已创建 ASCII 目录链接 {link} -> {real_root}")
    return link


def _patch_gaussian_fit() -> None:
    """修复 MultiVariateGaussian 在 Windows + CUDA 上的性能问题。

    anomalib 2.6.2 原始实现：逐位置 Python 循环求协方差，随后
    ``torch.linalg.inv`` 对 (P,550,550) 批量求逆。cuSOLVER 批量求逆在本机
    （RTX 4060 Laptop + Windows）逐批串行执行，实测 20+ 分钟未完成
    （GPU 利用率 100% 但功耗仅 ~37W，纯内核启动开销）。
    本补丁：einsum 向量化协方差 + 转 CPU numpy 求逆（约 15~30 秒）。
    数学等价：ddof=1 与 0.01+1e-5 对角线正则均与原实现一致。
    """
    from anomalib.models.components import MultiVariateGaussian

    def _fast_forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        device = embedding.device
        b, c, h, w = embedding.size()
        vectors = embedding.view(b, c, h * w)
        mean = torch.mean(vectors, dim=0)  # (C, P)
        centered = vectors - mean.unsqueeze(0)  # (B, C, P)
        # 向量化协方差 (P, C, C)，ddof=1（与 anomalib _cov 默认一致）
        cov = torch.einsum("bcp,bdp->pcd", centered, centered) / (b - 1)
        identity = torch.eye(c, device=device)
        cov = cov + (0.01 + 1e-5) * identity  # 与原实现相同的对角线正则
        cov_np = cov.detach().cpu().numpy()
        del cov  # 立即释放 GPU 显存
        inv = np.stack([np.linalg.inv(cov_np[i]) for i in range(cov_np.shape[0])])
        self.mean = mean
        self.inv_covariance = torch.from_numpy(inv).to(device)
        return [self.mean, self.inv_covariance]

    MultiVariateGaussian.forward = _fast_forward


class PadimAnomalibModel(BaseAnomalyModel):
    """Anomalib PaDiM 适配层。

    - train(train_loader)：内部使用 Anomalib 自带 MVTecAD 数据模块（与自定义
      数据集同样的 256×256 + ImageNet 归一化），完成高斯统计量拟合
    - predict_dataset()：对整个测试集推理，返回 {图像绝对路径: 原始异常热力图}
    - predict_image()：单张图像推理（Gradio 部署），复用 PadimModel 前向
    """

    name = "padim"

    def __init__(
        self,
        device="auto",
        image_size: int = 256,
        backbone: str = "vgg16",
        layers: list[str] | None = None,
        n_features: int | None = None,
    ):
        super().__init__(device, image_size)
        self.backbone = backbone
        # timm 特征阶段名（resnet: layer1-3；vgg16: features.N = conv4_3 / conv5_3 输出）
        # 注：选用 16×16 分辨率起点的两层融合——若从 features.15（32×32）开始，
        # 协方差矩阵 (550²×1024) 及其求逆临时矩阵在 8GB 笔记本 GPU 上会显存溢出
        if layers is None:
            layers = (
                ["layer1", "layer2", "layer3"]
                if "resnet" in backbone
                else ["features.22", "features.30"]
            )
        self.layers = layers
        # 非默认骨干必须显式给定特征通道数。随机投影降维到 550 维（PaDiM 原论文设置）
        if n_features is None and "vgg" in backbone:
            n_features = 550
        self.n_features = n_features
        self._model = None
        self._engine = None
        self._datamodule = None
        self._data_root = None
        self._category = None
        self._pred_cache: dict[str, np.ndarray] | None = None
        self._loaded_stats: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    # ------------------------------------------------------------------ #
    # 训练：Anomalib Engine.fit（PaDiM 仅统计量计算，无梯度）
    # ------------------------------------------------------------------ #
    def train(self, train_loader) -> None:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Padim

        # 高斯拟合性能补丁（必须在使用 MultiVariateGaussian 前打上）
        _patch_gaussian_fit()

        # 与自定义数据集保持完全一致的数据源与预处理
        self._data_root = train_loader.dataset.root
        category = train_loader.dataset.category
        self._category = category

        # Anomalib 内部校验拒绝非 ASCII 路径，改用 junction 链接
        self._anomalib_root = _ensure_ascii_link(self._data_root)

        datamodule = MVTecAD(
            root=str(self._anomalib_root),
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
            seed=42,
        )
        model = Padim(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=True,
            n_features=self.n_features,
        )
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        engine = Engine(accelerator=accelerator, devices=1, max_epochs=1)

        print(f"[{self.name}] Anomalib PaDiM 训练（backbone={self.backbone}, layers={self.layers}）")
        engine.fit(model, datamodule=datamodule)
        self._model, self._engine, self._datamodule = model, engine, datamodule
        print(f"[{self.name}] 高斯统计量拟合完成")

    # ------------------------------------------------------------------ #
    # 推理：整测试集预测 -> {path: anomaly_map}
    # ------------------------------------------------------------------ #
    def predict_dataset(self) -> dict[str, np.ndarray]:
        """对整个测试集推理，返回 {图像绝对路径: 原始异常热力图 (H,W)}。"""
        if self._pred_cache is not None:
            return self._pred_cache
        if self._model is None:
            if self._loaded_stats is None:
                raise RuntimeError("请先调用 train()")
            self._rebuild_from_checkpoint()
        preds = self._engine.predict(
            self._model, datamodule=self._datamodule, return_predictions=True
        )

        result: dict[str, np.ndarray] = {}
        batches = preds if isinstance(preds, (list, tuple)) else [preds]
        for batch in batches:
            for i in range(len(batch)):
                path = str(batch.image_path[i])
                # junction 路径映射回真实数据根，保证与自定义数据集路径对齐
                link = str(self._anomalib_root)
                if path.startswith(link):
                    path = str(self._data_root) + path[len(link):]
                amap = batch.anomaly_map[i]
                if isinstance(amap, torch.Tensor):
                    amap = amap.detach().cpu().numpy()
                result[path] = np.asarray(amap, dtype=np.float32)
        self._pred_cache = result
        return result

    def predict_dataset_fast(self, batch_size: int = 32) -> dict[str, np.ndarray]:
        """批量前向 + GPU 马氏距离推理（优化实验：绕开 Engine 调度与 dataloader 开销）。

        与 predict_dataset() 输出同源：同一 PadimModel、同一高斯统计量、
        同一预处理（read_image float32 → v2 抗锯齿 resize 256 → ImageNet
        归一化），仅省去 Engine/Lightning 的调度开销，单张耗时降至毫秒级。
        """
        if self._pred_cache is not None:
            return self._pred_cache
        if self._model is None:
            if self._loaded_stats is None:
                raise RuntimeError("请先调用 train()")
            self._rebuild_from_checkpoint()
        from anomalib.data.utils.image import read_image
        from torchvision.transforms import v2 as T

        torch_model = self._model.model.to(self.device)
        was_training = torch_model.training
        torch_model.eval()
        pp = T.Compose([
            T.Resize((256, 256), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # _data_root 是类别目录的父级（MVTec 根），需拼接 category
        test_root = Path(self._data_root) / self._category / "test"
        paths = sorted(
            p for p in test_root.glob("*/*")
            if p.suffix.lower() in (".png", ".jpg", ".bmp")
        )
        result: dict[str, np.ndarray] = {}
        try:
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                imgs = [read_image(str(p), as_tensor=True) for p in chunk]  # float32 [0,1]
                x = torch.stack([pp(t) for t in imgs]).to(self.device)
                with torch.no_grad():
                    out = torch_model(x)
                for p, amap in zip(chunk, out.anomaly_map):
                    result[str(p)] = np.asarray(amap.detach().cpu(), dtype=np.float32)
        finally:
            if was_training:
                torch_model.train()
        self._pred_cache = result
        return result

    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        """单张图像推理（Gradio 三模型部署）：返回原始异常热力图 (H, W)。

        复用 Anomalib PadimModel 的 eval 前向：特征提取 → 随机投影 →
        马氏距离 → 上采样求和。输入须按 Anomalib 默认 PreProcessor 预处理
        （torchvision v2 抗锯齿 resize 256×256 + ImageNet 归一化），
        Gradio 端 app.py 的 ``_preprocess_tv`` 即此实现，与整测试集评估同源。
        """
        if self._model is None:
            if self._loaded_stats is None:
                raise RuntimeError("请先调用 train()")
            self._rebuild_from_checkpoint()
        # 直接调用前向不走 Engine.predict，需手动把子模块搬到设备上
        # （重建时权重与统计量仍在 CPU，Engine 路径由 Lightning 自动搬移）；
        # 已在设备上时跳过 .to()（遍历全部参数张量有毫秒级开销，热路径每张图都会调用）
        torch_model = self._model.model
        if next(torch_model.parameters()).device.type != self.device.type:
            torch_model = torch_model.to(self.device)
        was_training = torch_model.training
        torch_model.eval()
        with torch.no_grad():
            out = torch_model(image.to(self.device))
        if was_training:
            torch_model.train()
        amap = out.anomaly_map
        if isinstance(amap, torch.Tensor):
            amap = amap.detach().cpu().numpy()
        return np.asarray(amap, dtype=np.float32).squeeze()

    # ------------------------------------------------------------------ #
    # 持久化：保存已拟合的高斯统计量（mean + 逆协方差），
    # 加载后无需重新拟合即可直接推理（PaDiM 的"权重"即统计量）
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> dict:
        state = {"backbone": self.backbone, "layers": self.layers, "n_features": self.n_features}
        if self._model is not None:
            gaussian = self._model.model.gaussian
            state["category"] = self._category
            state["data_root"] = str(self._data_root)
            state["mean"] = gaussian.mean.detach().cpu()
            state["inv_covariance"] = gaussian.inv_covariance.detach().cpu()
            # idx 随机投影与统计量必须成对保存，否则加载后投影不一致导致指标失真
            state["idx"] = self._model.model.idx.detach().cpu()
        return state

    def _load_state_dict(self, state: dict) -> None:
        self.backbone = state["backbone"]
        self.layers = state["layers"]
        self.n_features = state.get("n_features")
        self._loaded_stats = None
        if "mean" in state:
            self._loaded_stats = (
                state["data_root"],
                state["mean"],
                state["inv_covariance"],
                state["idx"],
            )
            self._category = state.get("category", "metal_nut")

    def _rebuild_from_checkpoint(self) -> None:
        """从 checkpoint 恢复模型与高斯统计量（等价于 train() 后的状态）。"""
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Padim

        data_root, mean, inv_cov, idx = self._loaded_stats
        self._data_root = Path(data_root)
        self._anomalib_root = _ensure_ascii_link(self._data_root)

        model = Padim(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=True,
            n_features=self.n_features,
        )
        gaussian = model.model.gaussian
        device = model.device
        gaussian.mean = mean.to(device)
        gaussian.inv_covariance = inv_cov.to(device)
        # 恢复训练时的随机投影索引（与统计量成对，缺一不可）
        model.model.idx.copy_(idx.to(model.model.idx.device))

        self._datamodule = MVTecAD(
            root=str(self._anomalib_root),
            category=self._category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
            seed=42,
        )
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self._engine = Engine(accelerator=accelerator, devices=1, max_epochs=1)
        self._model = model
        print(f"[{self.name}] 已从 checkpoint 恢复高斯统计量（跳过拟合）")
