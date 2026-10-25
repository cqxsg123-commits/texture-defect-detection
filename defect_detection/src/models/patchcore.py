"""PatchCore 模型（主方案，计划书模型 1）。

原理：冻结预训练 VGG16 仅作纹理特征提取器；
- 训练：提取正常样本局部 Patch 特征 -> 贪心 coreset 采样 -> FAISS 特征记忆库
- 推理：提取待测图 Patch 特征 -> k 近邻检索 -> 距离越大越偏离正常纹理 -> 像素级异常热力图

无需反向传播；支持单层 / 多层 VGG 特征融合（实验 2 消融）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .base import BaseAnomalyModel
from src.utils import resize_map, smooth_map

try:
    import faiss

    _FAISS_OK = True
except ImportError:  # pragma: no cover
    faiss = None
    _FAISS_OK = False

# VGG16 特征层名称 -> (features 序列下标, 通道数, 256 输入下的分辨率)
# 注意 torchvision vgg16.features 下标：14=conv3_3(256ch,32×32)、
# 24=conv4_3(512ch,16×16)、30=conv5_3(512ch,8×8)（下标 17 是 conv4_1，勿混淆）
VGG16_LAYERS = {
    "conv3_3": (14, 256, 32),
    "conv4_3": (24, 512, 16),
    "conv5_3": (30, 512, 8),
}


def _extract_layer_features(model: nn.Module, x: torch.Tensor, layer_name: str) -> torch.Tensor:
    """单次前向提取指定中间层特征图 (B,C,h,w)。"""
    idx, _, _ = VGG16_LAYERS[layer_name]
    out = {}
    hook = model.features[idx].register_forward_hook(
        lambda m, i, o: out.setdefault("feat", o)
    )
    try:
        with torch.no_grad():
            model(x)
    finally:
        hook.remove()
    return out["feat"]


def _extract_multilayer_features(model: nn.Module, x: torch.Tensor, layer_names: list[str]) -> list[torch.Tensor]:
    """一次前向同时提取多个中间层特征图 [(B,C,h,w), ...]（批量 GPU 推理用）。"""
    outs: dict[str, torch.Tensor] = {}
    handles = [
        model.features[VGG16_LAYERS[n][0]].register_forward_hook(
            lambda m, i, o, n=n: outs.setdefault(n, o)
        )
        for n in layer_names
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return [outs[n] for n in layer_names]


class PatchCoreModel(BaseAnomalyModel):
    """PatchCore：VGG16 特征 + coreset 记忆库 + FAISS kNN 检索。"""

    name = "patchcore"

    def __init__(
        self,
        device="auto",
        image_size: int = 256,
        layers: list[str] | str | None = None,
        k: int = 5,
        coreset_ratio: float = 0.1,
        stride: int = 1,
    ):
        super().__init__(device, image_size)
        if not _FAISS_OK:
            raise ImportError("PatchCore 需要 faiss-cpu，请先 pip install faiss-cpu")
        self.layers = (
            [layers] if isinstance(layers, str) else (layers or ["conv4_3"])
        )
        for name in self.layers:
            if name not in VGG16_LAYERS:
                raise ValueError(f"未知 VGG 特征层 {name}，可选 {list(VGG16_LAYERS)}")
        self.k, self.coreset_ratio, self.stride = k, coreset_ratio, stride

        # 冻结预训练 VGG16（仅特征提取，无梯度）
        self.backbone = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.memory_bank: np.ndarray | None = None  # (M,C) 记忆库
        self.index: faiss.IndexFlatL2 | None = None
        self._bank_t: torch.Tensor | None = None   # GPU 记忆库缓存（批量 GPU 检索用）

    # ------------------------------------------------------------------ #
    # 特征提取
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _extract_patches(self, images: torch.Tensor) -> np.ndarray:
        """提取多层融合 Patch 特征。

        Args:
            images: (B,3,H,W) 归一化图像。

        Returns:
            (B*P, C) 特征矩阵，P 为特征图 Patch 数（多层取最小分辨率并拼接）。
        """
        batch_feats = []
        for img in images:
            img = img.unsqueeze(0).to(self.device)
            layer_feats = []
            resolutions = [VGG16_LAYERS[n][2] for n in self.layers]
            target_res = min(resolutions)
            for name in self.layers:
                feat = _extract_layer_features(self.backbone, img, name)  # (1,C,h,w)
                if feat.shape[-1] != target_res:
                    feat = torch.nn.functional.avg_pool2d(
                        feat, feat.shape[-1] // target_res
                    )
                layer_feats.append(feat)
            fused = torch.cat(layer_feats, dim=1)  # (1,Cs,h,w) 多层特征融合
            # 步长采样（计划书：patch 采样步长超参数）
            fused = fused[:, :, :: self.stride, :: self.stride]
            batch_feats.append(fused.reshape(fused.shape[1], -1).T)  # (P,C)
        return torch.cat(batch_feats, dim=0).cpu().numpy()

    # ------------------------------------------------------------------ #
    # 训练：构建 coreset 记忆库
    # ------------------------------------------------------------------ #
    def train(self, train_loader) -> None:
        print(f"[{self.name}] 提取训练特征（layers={self.layers}, stride={self.stride}）...")
        all_feats = []
        for batch in train_loader:
            all_feats.append(self._extract_patches(batch["image"]))
        feats = np.concatenate(all_feats, axis=0)
        print(f"[{self.name}] 特征总数 {len(feats)} × {feats.shape[1]} 维")

        n_coreset = max(1, int(len(feats) * self.coreset_ratio))
        self.memory_bank = _greedy_coreset(feats, n_coreset)
        print(f"[{self.name}] coreset 采样 {len(self.memory_bank)} 个 Patch（比例 {self.coreset_ratio}）")

        # 构建 FAISS 最近邻索引
        self.index = faiss.IndexFlatL2(self.memory_bank.shape[1])
        self.index.add(self.memory_bank.astype(np.float32))
        self._bank_t = None  # 记忆库已变，失效 GPU 缓存
        print(f"[{self.name}] FAISS 记忆库构建完成")

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _extract_patches_batch(self, images: torch.Tensor) -> torch.Tensor:
        """批量提取多层融合 Patch 特征（一次前向，全部图像并行）。

        Args:
            images: (B,3,H,W) 归一化图像。

        Returns:
            (B,P,C) 特征，P 为特征图 Patch 数（多层取最小分辨率并拼接）。
        """
        images = images.to(self.device)
        layer_feats = _extract_multilayer_features(self.backbone, images, self.layers)
        resolutions = [VGG16_LAYERS[n][2] for n in self.layers]
        target_res = min(resolutions)
        for i, feat in enumerate(layer_feats):
            if feat.shape[-1] != target_res:
                feat = torch.nn.functional.avg_pool2d(feat, feat.shape[-1] // target_res)
                layer_feats[i] = feat
        fused = torch.cat(layer_feats, dim=1)  # (B,Cs,h,w)
        fused = fused[:, :, :: self.stride, :: self.stride]
        b, c, h, w = fused.shape
        return fused.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B,P,C)

    def _get_bank_tensor(self) -> torch.Tensor:
        """GPU 记忆库（惰性缓存）。

        注意：比较 device.type 而非整个 device——self.device 为
        ``torch.device("cuda")``（index=None），而实际张量在 ``cuda:0``，
        直接比较不相等会导致每次调用都重建记忆库（173MB 拷贝 + 传输）。
        """
        if self._bank_t is None or self._bank_t.device.type != self.device.type:
            self._bank_t = torch.from_numpy(self.memory_bank.astype(np.float32)).to(self.device)
        return self._bank_t

    def _knn_gpu(self, feats_t: torch.Tensor, k: int) -> torch.Tensor:
        """GPU 分块 L2 距离 + k 近邻（数学与 FAISS IndexFlatL2 一致）。

        Args:
            feats_t: (Q,C) 查询特征（GPU）。

        Returns:
            (Q,k) k 近邻距离（未开方，与 FAISS 返回的 squared L2 一致）。
        """
        bank = self._get_bank_tensor()  # (M,C)
        m = bank.shape[0]
        feats_norm = feats_t.pow(2).sum(1, keepdim=True)  # (Q,1)
        bank_norm = bank.pow(2).sum(1)  # (M,)，只读一遍记忆库
        # 查询数小时整库一次 GEMM（少分块 = 少 kernel 启动与归约，实测快 3 倍以上）；
        # 查询数大时按记忆库分块，控制 d² 中间矩阵显存峰值（约 256MB）
        max_elems = 1 << 26
        bank_chunk = m if feats_t.shape[0] * m <= max_elems else max_elems // feats_t.shape[0]
        cand_d = []
        for i in range(0, m, bank_chunk):
            b = bank[i : i + bank_chunk]  # (chunk,C)
            # d² = |x|² + |y|² − 2·x·yᵀ：矩阵乘法走 Tensor Core，
            # 比 cdist（逐输出元素重读输入）快一个数量级
            d2 = feats_norm + bank_norm[i : i + bank_chunk].unsqueeze(0) - 2.0 * (feats_t @ b.t())
            d2 = d2.clamp_min(0.0)  # 防展开式浮点误差出现微小负数
            if k == 1:
                cand_d.append(d2.min(dim=1).values)  # k=1 行最小即最近邻距离
            else:
                top_d, _ = torch.topk(-d2, min(k, d2.shape[1]), dim=1)
                cand_d.append(-top_d)
        if k == 1:
            return torch.stack(cand_d, dim=1).min(dim=1).values.unsqueeze(1)
        cand_d = torch.cat(cand_d, dim=1)  # (Q, 块数*k)
        final_d, _ = torch.topk(-cand_d, k, dim=1)  # 各块候选合并后取最终 k
        return -final_d

    def _amaps_from_feats(self, feats_t: torch.Tensor, n: int) -> np.ndarray:
        """(B,P,C) 特征 -> kNN 距离 -> (B,H,W) 异常热力图（GPU 检索）。"""
        q, _, c = feats_t.shape
        flat = feats_t.reshape(-1, c)
        scores = []
        for i in range(0, flat.shape[0], 2048):  # 查询分块，控制显存峰值
            scores.append(self._knn_gpu(flat[i : i + 2048], self.k).mean(dim=1))
        scores = torch.cat(scores)  # (B*P,)
        target_res = min(VGG16_LAYERS[n][2] for n in self.layers) // self.stride
        maps = scores.reshape(n, target_res, target_res).cpu().numpy()
        out = np.stack([
            smooth_map(resize_map(m, (self.image_size, self.image_size)), sigma=4)
            for m in maps
        ])
        return out

    def predict_image_faiss(self, image: torch.Tensor) -> np.ndarray:
        """FAISS CPU 检索单图推理（官方评估协议路径，跨设备确定性一致）。"""
        assert self.index is not None, "请先调用 train() 构建记忆库"
        feats = self._extract_patches(image)  # (P,C)
        distances, _ = self.index.search(feats.astype(np.float32), self.k)
        anomaly_scores = distances.mean(axis=1)  # (P,)

        # 还原成特征图空间分辨率
        target_res = min(VGG16_LAYERS[n][2] for n in self.layers) // self.stride
        amap = anomaly_scores.reshape(target_res, target_res)
        amap = resize_map(amap, (self.image_size, self.image_size))
        return smooth_map(amap, sigma=4)

    def predict_image(self, image: torch.Tensor) -> np.ndarray:
        """kNN 平均距离 -> 异常热力图（GPU 分块 GEMM 检索，与 FAISS 数学等价）。"""
        assert self.index is not None, "请先调用 train() 构建记忆库"
        if self.device.type == "cuda":
            feats_t = self._extract_patches_batch(image)  # (1,P,C) 一次前向
            return self._amaps_from_feats(feats_t, 1)[0]
        return self.predict_image_faiss(image)  # CPU 回退

    def predict(self, images: torch.Tensor) -> np.ndarray:
        """批量推理 -> (B,H,W) 异常热力图（批量特征提取 + GPU 检索）。"""
        assert self.index is not None, "请先调用 train() 构建记忆库"
        if self.device.type == "cuda":
            feats_t = self._extract_patches_batch(images)
            return self._amaps_from_feats(feats_t, images.shape[0])
        return super().predict(images)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _state_dict(self) -> dict:
        return {
            "memory_bank": self.memory_bank,
            "layers": self.layers,
            "k": self.k,
            "coreset_ratio": self.coreset_ratio,
            "stride": self.stride,
        }

    def _load_state_dict(self, state: dict) -> None:
        self.memory_bank = state["memory_bank"]
        self.layers = state["layers"]
        self.k = state["k"]
        self.coreset_ratio = state["coreset_ratio"]
        self.stride = state["stride"]
        self.index = faiss.IndexFlatL2(self.memory_bank.shape[1])
        self.index.add(self.memory_bank.astype(np.float32))
        self._bank_t = None  # 记忆库已变，失效 GPU 缓存


def _greedy_coreset(features: np.ndarray, n: int) -> np.ndarray:
    """贪心最远点采样（k-center-greedy 近似）压缩记忆库。"""
    feats = features.astype(np.float32)
    if n >= len(feats):
        return feats
    rng = np.random.default_rng(42)
    idx = faiss.IndexFlatL2(feats.shape[1])
    first = int(rng.integers(len(feats)))
    idx.add(feats[first : first + 1])
    selected = [first]
    remaining = n - 1
    # 每次迭代取离已选集最远的点加入（分批加速）
    batch = max(8, remaining // 100)
    while len(selected) < n:
        d, _ = idx.search(feats, 1)
        order = np.argsort(-d[:, 0])
        add = []
        for i in order:
            if len(selected) + len(add) >= n:
                break
            add.append(int(i))
        idx.add(feats[add])
        selected.extend(add)
        if len(add) == 0:
            break
    return feats[selected[:n]]
