"""计算三模型的自动判定阈值，并在测试集上验证判定效果（质检判定，非评估协议）。

参考 B2 组螺丝缺陷检测系统的做法：随模型权重保存一个由训练集统计
得到的默认判定阈值，部署时「异常得分 ≥ 阈值」即判不合格（阈值可经
灵敏度倍数滑杆微调，默认 1.0 = 自动阈值）。

流程（每模型一次加载）：
1. 对 220 张正常训练样本逐一推理（走与 Gradio 部署完全相同的
   predict_image 路径），取异常得分最大值作为自动阈值（另记录
   p99/p99.5 供参考；若热力图整体落在概率尺度 [0,1] 且最大值饱和
   ≥0.9995，则以 p99 兜底，防止 GradCAM 类热力图在正常样本上饱和
   导致阈值失效）。PatchCore 例外：训练样本自身在记忆库中（自匹配
   距离≈0），改用留一法——取第二近邻距离（第一近邻是自身），与
   部署 k=1 得分同尺度；
2. 在测试集（115 张 = 22 正常 + 93 缺陷）上统计自动阈值下的判定
   表现：缺陷样本检出率、正常样本误报率。

输出：
- checkpoints/thresholds.json   随权重文件一起分发（Gradio 读取）
- results/threshold_verdict.json 判定验证结果（报告素材）

用法：
    PYTHONIOENCODING=utf-8 python scripts/compute_auto_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 保证从项目根或 scripts/ 目录运行都能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src import CHECKPOINT_DIR, RESULTS_DIR
from src.data import get_test_loader, get_train_loader
from src.experiment_utils import load_config, resolve
from src.models import CutPasteModel, PadimAnomalibModel, PatchCoreModel
from src.models.patchcore import VGG16_LAYERS, resize_map, smooth_map
from src.utils import set_seed

# Gradio 界面显示名 -> (模型类, 权重文件名)
MODEL_CLASSES = {
    "PatchCore（主方案）": (PatchCoreModel, "patchcore.pth"),
    "PaDiM": (PadimAnomalibModel, "padim.pth"),
    "CutPaste": (CutPasteModel, "cutpaste.pth"),
}


def _scores_on_loader(model, loader) -> list[float]:
    """对整个 loader 逐图推理，返回图像级异常得分列表（max 聚合，与部署一致）。"""
    scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"]
            for i in range(x.shape[0]):
                scores.append(float(model.predict_image(x[i : i + 1]).max()))
    return scores


def _patchcore_loo_score(model: PatchCoreModel, image: torch.Tensor) -> float:
    """PatchCore 留一法图像得分：每 patch 取第二近邻距离（第一近邻是训练样本
    自身，距离≈0），再做与部署一致的 resize+smooth，取图像最大值。

    尺度与部署 k=1 一致：测试图 k=1 的第一近邻与训练图 k=2 的第二近邻
    都是「库外最近真实匹配」距离。
    """
    feats_t = model._extract_patches_batch(image)  # (1,P,C)
    d2 = model._knn_gpu(feats_t.reshape(-1, feats_t.shape[-1]), 2)  # (P,2) 升序
    target_res = min(VGG16_LAYERS[n][2] for n in model.layers) // model.stride
    amap = d2[:, 1].reshape(target_res, target_res).cpu().numpy()
    amap = resize_map(amap, (model.image_size, model.image_size))
    return float(smooth_map(amap, sigma=4).max())


def _scores_on_loader_patchcore(model: PatchCoreModel, loader) -> list[float]:
    """PatchCore 训练集留一法得分列表（训练样本在库中，必须剔除自身匹配）。"""
    scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"]
            for i in range(x.shape[0]):
                scores.append(_patchcore_loo_score(model, x[i : i + 1]))
    return scores


def main() -> None:
    cfg = load_config()
    set_seed(cfg["dataset"]["seed"])
    data_root = resolve(cfg["paths"]["data_root"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_train_loader(
        data_root,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=0,
        image_size=cfg["dataset"]["image_size"],
        use_augmentation=False,
    )
    test_loader = get_test_loader(
        data_root,
        batch_size=cfg["dataset"]["batch_size"],
        num_workers=0,
        image_size=cfg["dataset"]["image_size"],
    )

    thresholds: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    for name, (cls, ckpt_name) in MODEL_CLASSES.items():
        model = cls(device=device)
        model.load(Path(CHECKPOINT_DIR) / ckpt_name)

        # 1) 训练集（仅正常）自动阈值；PatchCore 用留一法（剔除自身匹配）
        if isinstance(model, PatchCoreModel):
            scores = sorted(_scores_on_loader_patchcore(model, train_loader))
            max_note = "留一法（第二近邻）训练集正常样本异常得分最大值"
        else:
            scores = sorted(_scores_on_loader(model, train_loader))
            max_note = "训练集正常样本异常得分最大值"
        n = len(scores)
        auto = scores[-1]
        # 仅当热力图整体落在概率尺度 [0,1] 且最大值饱和时最大法才失效，
        # 改用 p99 兜底；马氏距离类尺度（PaDiM ≈ 10²）不适用此规则
        if scores[0] >= 0.0 and scores[-1] <= 1.0 + 1e-3 and auto >= 0.9995:
            auto = scores[int(n * 0.99) - 1]
            note = "最大值饱和，取 p99 兜底"
        else:
            note = max_note
        thresholds[name] = {
            "auto": round(auto, 6),
            "max": round(scores[-1], 6),
            "p99": round(scores[int(n * 0.99) - 1], 6),
            "p99.5": round(scores[int(n * 0.995) - 1], 6),
            "n_train": n,
            "note": note,
        }

        # 2) 测试集判定验证（label: 0=正常 1=缺陷）
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for batch in test_loader:
                x, labels = batch["image"], batch["label"]
                for i in range(x.shape[0]):
                    flagged = float(model.predict_image(x[i : i + 1]).max()) >= auto
                    if labels[i].item() == 1:
                        tp += flagged; fn += not flagged
                    else:
                        fp += flagged; tn += not flagged
        recall = tp / (tp + fn)          # 缺陷检出率
        far = fp / (fp + tn)             # 正常误报率
        verdicts[name] = {
            "recall": round(recall, 4),
            "false_alarm": round(far, 4),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }
        print(f"{name}: auto={auto:.4f} ({note})\n"
              f"    测试集判定：检出率 {recall:.2%}（{tp}/{tp + fn}）、"
              f"误报率 {far:.2%}（{fp}/{fp + tn}）")

    out = Path(CHECKPOINT_DIR) / "thresholds.json"
    out.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    vout = Path(RESULTS_DIR) / "threshold_verdict.json"
    vout.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n自动阈值已保存 -> {out}")
    print(f"判定验证结果已保存 -> {vout}")


if __name__ == "__main__":
    main()
