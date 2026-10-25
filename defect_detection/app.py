"""基于纹理特征的工业缺陷检测系统 — Gradio 交互入口（计划书 4.4）。

模块划分：
1. 图像上传与预处理：本地图片上传 -> RGB -> resize 256×256 -> ImageNet 归一化
2. 三模型推理：PatchCore / PaDiM / CutPaste 可切换
3. 质检判定（参考 B2 组螺丝缺陷检测系统）：异常得分 ≥ 判定阈值判「不合格」，
   阈值 = 灵敏度倍数 × 自动阈值（训练集正常样本统计，随权重分发），
   默认 1.0 即自动阈值；倍数调小更严格（少漏检）、调大更宽松（少误报）
4. 可视化输出：原图、异常热力图（叠加）、缺陷掩码（≥ 判定阈值区域）、
   合格/不合格彩色判定与异常得分

启动：python app.py   然后浏览器打开 http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 保证在任意目录启动时都能导入 src
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import v2 as T

from src import CHECKPOINT_DIR
from src.models import CutPasteModel, PadimAnomalibModel, PatchCoreModel
from src.utils import IMAGENET_MEAN, IMAGENET_STD, heatmap_overlay, set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------- #
# 模型加载
# ---------------------------------------------------------------------- #
# 显示名 -> (模型类, 权重文件名)
MODEL_CLASSES = {
    "PatchCore（主方案）": (PatchCoreModel, "patchcore.pth"),
    "PaDiM": (PadimAnomalibModel, "padim.pth"),
    "CutPaste": (CutPasteModel, "cutpaste.pth"),
}

_MODELS: dict[str, object] = {}
_AUTO_THRESHOLDS: dict[str, float] = {}


def load_models() -> dict[str, str]:
    """启动时加载全部可用模型权重与自动判定阈值，返回提示信息。"""
    ckpt_dir = Path(CHECKPOINT_DIR)
    messages = []
    for name, (cls, ckpt_name) in MODEL_CLASSES.items():
        ckpt = ckpt_dir / ckpt_name
        try:
            model = cls(device=DEVICE)
            model.load(ckpt)
            _MODELS[name] = model
        except FileNotFoundError:
            messages.append(f"⚠ 未找到 {name} 权重（{ckpt.name}），"
                            f"请先运行：python experiments/exp1_model_compare.py")
    tj = ckpt_dir / "thresholds.json"
    if tj.exists():
        data = json.loads(tj.read_text(encoding="utf-8"))
        _AUTO_THRESHOLDS.update({k: float(v["auto"]) for k, v in data.items()})
    else:
        messages.append("⚠ 未找到 checkpoints/thresholds.json（自动判定阈值），"
                        "将按归一化热力图 0.5 兜底；可运行："
                        "python scripts/compute_auto_threshold.py")
    return messages


# ---------------------------------------------------------------------- #
# 预处理与推理
# ---------------------------------------------------------------------- #
def _preprocess(image: np.ndarray) -> torch.Tensor:
    """Gradio 图像 -> 归一化张量 (1,3,256,256)（PatchCore / CutPaste：cv2 双线性 resize，
    与训练/评估时的 MVTecDataset 预处理一致）。"""
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = image[..., :3]
    if image.ndim == 2 or image.shape[-1] == 1:  # 灰度 -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])(image)
    return t.unsqueeze(0)


# PaDiM 专用预处理：与 Anomalib 默认 PreProcessor 逐操作同源
# （uint8 -> float32[0,1] -> 抗锯齿 resize -> ImageNet 归一化；
#  Anomalib 评估管线为 read_image(float) + v2.Resize(antialias=True) + Normalize）
_PP_TV = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((256, 256), antialias=True),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


def _preprocess_tv(image: np.ndarray) -> torch.Tensor:
    """Gradio 图像 -> 归一化张量 (1,3,256,256)（PaDiM：float 上 torchvision 抗锯齿
    resize，与 Anomalib 评估管线逐像素一致）。"""
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = image[..., :3]
    if image.ndim == 2 or image.shape[-1] == 1:  # 灰度 -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    t = _PP_TV(Image.fromarray(image))
    return t.unsqueeze(0)


# 模型类 -> 其评估管线使用的预处理（保证 Gradio 输出与报告评估同源）
_PREPROCESS = {
    PatchCoreModel: _preprocess,
    CutPasteModel: _preprocess,
    PadimAnomalibModel: _preprocess_tv,
}


def detect(image: np.ndarray, model_name: str, threshold_mult: float) -> tuple:
    """三模型推理入口（质检判定：异常得分 ≥ 判定阈值 → 不合格）。

    Args:
        threshold_mult: 灵敏度倍数，判定阈值 = threshold_mult × 自动阈值；
            调小更严格（检出率↑、误报率↑）、调大更宽松（反之）。

    Returns:
        (原图, 热力图叠加图, 缺陷掩码, 合格/不合格判定 HTML, 检测详情文本)
    """
    if image is None:
        raise gr.Error("请先上传一张图片")
    if model_name not in _MODELS:
        raise gr.Error(f"模型未加载，请先训练并保存权重。\n{load_models()}")

    model = _MODELS[model_name]
    preprocess = _PREPROCESS.get(type(model), _preprocess)
    x = preprocess(image).to(model.device)
    with torch.no_grad():
        raw_map = model.predict_image(x)
    score = float(raw_map.max())

    # 判定阈值 = 灵敏度倍数 × 自动阈值（未生成 thresholds.json 时按
    # 归一化热力图 0.5 兜底，即原始热力图值域中点）
    auto = _AUTO_THRESHOLDS.get(model_name)
    if auto is None:
        auto = (raw_map.max() - raw_map.min()) * 0.5 + raw_map.min()
    thresh = float(threshold_mult) * auto
    is_defect = score >= thresh

    # 可视化输出（热力图/掩码上采样回原始图像尺寸，保证叠加与原图对齐）
    original = image
    h, w = image.shape[:2]
    display_map = cv2.resize(raw_map, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = heatmap_overlay(image.astype(np.float32) / 255.0, display_map)
    # 缺陷掩码 = 异常得分 ≥ 判定阈值的区域（与判定结论同源）
    mask = (raw_map >= thresh).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    verdict, color = ("不合格（检出缺陷）", "#dc2626") if is_defect else ("合格", "#16a34a")
    verdict_html = (
        f'<div style="font-size:30px;font-weight:bold;color:{color};'
        f'text-align:center;padding:14px;border:3px solid {color};'
        f'border-radius:10px;">{verdict}</div>'
    )
    auto_src = "自动阈值（训练集统计）" if model_name in _AUTO_THRESHOLDS else "兜底（归一化 0.5）"
    score_text = (f"异常得分：{score:.4f}\n"
                  f"判定阈值：{thresh:.4f} = {threshold_mult:.2f} × {auto:.4f}（{auto_src}）\n"
                  f"判定结果：{verdict}\n模型：{model_name}")
    return original, overlay, mask, verdict_html, score_text


# ---------------------------------------------------------------------- #
# 界面
# ---------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="基于纹理特征的工业缺陷检测系统") as demo:
        gr.Markdown(
            "# 基于纹理特征的工业缺陷检测系统\n"
            "无监督异常检测：仅用正常样本训练，输出图像级异常得分与像素级缺陷掩码。\n"
            "数据集：MVTec AD metal_nut | 指标：图像级 AUROC / PixelAUROC"
        )
        load_messages = load_models()
        for msg in load_messages:
            gr.Markdown(msg)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="上传工件图像", type="numpy", height=320)
                model_choice = gr.Dropdown(
                    choices=list(MODEL_CLASSES.keys()),
                    value=list(_MODELS.keys())[0] if _MODELS else None,
                    label="推理模型",
                )
                threshold_mult = gr.Slider(
                    0.2, 2.0, value=1.0, step=0.05,
                    label="灵敏度阈值倍数（× 自动阈值；调小更严格、调大更宽松）",
                )
                detect_btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_overlay = gr.Image(label="异常热力图（叠加显示）", height=300)
                    out_mask = gr.Image(label="缺陷掩码（≥ 判定阈值区域）", height=300)
                out_verdict = gr.HTML(label="判定结果")
                out_score = gr.Textbox(label="检测详情", lines=4)

        detect_btn.click(
            detect,
            inputs=[input_image, model_choice, threshold_mult],
            outputs=[input_image, out_overlay, out_mask, out_verdict, out_score],
        )

        # 示例图片（若已生成）
        examples_dir = Path(__file__).parent / "examples"
        example_paths = sorted(examples_dir.glob("*.png"))[:4] if examples_dir.exists() else []
        if example_paths:
            gr.Examples(examples=[[str(p)] for p in example_paths], inputs=[input_image])

        gr.Markdown(
            "> 说明：异常得分 = 原始异常热力图最大值（Anomalib 协议），热力图越亮表示"
            "该区域纹理偏离正常模式越严重；自动阈值由训练集正常样本统计得到"
            "（`scripts/compute_auto_threshold.py`，PatchCore 采用留一法），"
            "异常得分 ≥ 判定阈值判不合格，倍数滑杆可按产线要求调节灵敏度。"
        )
    return demo


if __name__ == "__main__":
    set_seed(42)
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
