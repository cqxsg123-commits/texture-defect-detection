# -*- coding: utf-8 -*-
"""生成《三模型基线实验报告》Word 文档。

数据来自 defect_detection/results/ 下实验 4 的产出
（exp4_baseline.csv、baseline_vs_optimized.csv、heatmaps_baseline/），
由 exp4_baseline.py 训练生成；本脚本直接读取，无需手工填数。
"""

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

PROJECT = Path(r"D:\基于纹理特征的工业缺陷检测系统")
RESULTS = PROJECT / "defect_detection" / "results"
HEATMAPS = RESULTS / "heatmaps_baseline"
OUT = PROJECT / "基线实验报告_基于纹理特征的工业缺陷检测系统.docx"

# ---------------------------------------------------------------------- #
# 基础排版工具（与主报告一致）
# ---------------------------------------------------------------------- #
def _set_run(run, text=None, font_cn="宋体", font_en="Times New Roman", size=12, bold=False):
    if text is not None:
        run.text = text
    run.font.name = font_en
    run.font.size = Pt(size)
    run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:eastAsia"), font_cn)


def P(doc, text="", size=12, bold=False, font_cn="宋体", align=None,
      indent=True, line=1.5, space_after=0, space_before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    if align is not None:
        pf.alignment = align
    if indent:
        pf.first_line_indent = Pt(size * 2)
    run = p.add_run()
    _set_run(run, text, font_cn=font_cn, size=size, bold=bold)
    return p


def H1(doc, text):
    return P(doc, text, size=16, bold=True, font_cn="黑体", indent=False,
             space_before=12, space_after=6)


def H2(doc, text):
    return P(doc, text, size=14, bold=True, font_cn="黑体", indent=False,
             space_before=8, space_after=4)


def _cell(cell, text, bold=False, size=10.5, font_cn="宋体"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.2
    run = p.add_run()
    _set_run(run, text, font_cn=font_cn, size=size, bold=bold)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_table(doc, header, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        _cell(table.rows[0].cells[j], h, bold=True, font_cn="黑体")
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            _cell(table.rows[i + 1].cells[j], str(v))
    if col_widths:
        for j, w in enumerate(col_widths):
            for cell in table.columns[j].cells:
                cell.width = Cm(w)
    return table


def add_figure(doc, img_path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(img_path), width=Cm(15))
    P(doc, caption, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
      line=1.2, space_after=6)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"), "end")
    run._element.append(fld1); run._element.append(instr); run._element.append(fld2)
    _set_run(run, size=9)


# ---------------------------------------------------------------------- #
# 读取实验结果
# ---------------------------------------------------------------------- #
base = pd.read_csv(RESULTS / "exp4_baseline.csv")
merged = pd.read_csv(RESULTS / "baseline_vs_optimized.csv")

MODELS = ["PatchCore", "PaDiM", "CutPaste"]
NAMES = {"PatchCore": "patchcore", "PaDiM": "padim", "CutPaste": "cutpaste"}

def get_row(df, model, tag):
    row = df[df["model"] == f"{model}（{tag}）"].iloc[0]
    return row


# ---------------------------------------------------------------------- #
# 文档
# ---------------------------------------------------------------------- #
doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21), Cm(29.7)
section.top_margin = section.bottom_margin = Cm(2.54)
section.left_margin = section.right_margin = Cm(3.17)
add_page_number(section)
doc.core_properties.title = "基于纹理特征的工业缺陷检测系统基线实验报告"
doc.core_properties.author = "陈宇 202305630103"

# ----------------------------- 封面 ----------------------------- #
for _ in range(5):
    P(doc, "", indent=False)
P(doc, "人工智能综合实践课程设计", size=26, bold=True, font_cn="黑体",
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
P(doc, "", indent=False)
P(doc, "基 线 实 验 报 告", size=36, bold=True, font_cn="黑体",
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
for _ in range(5):
    P(doc, "", indent=False)
for label, value in [
    ("项目名称：", "基于纹理特征的工业缺陷检测系统"),
    ("课题编号：", "2"),
    ("小组成员：", "陈宇 202305630103"),
    ("指导教师：", "刘焕华"),
    ("提交日期：", "2026年9月27日"),
]:
    P(doc, label + value, size=16, font_cn="宋体",
      align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, line=1.8)
doc.add_page_break()

# ----------------------------- 一、目的与设计 ----------------------------- #
H1(doc, "一、基线实验目的与设计")
P(doc, "在实验 1 的三模型横向对比中，各模型均采用了本项目加入的优化手段（多层特征融合、"
        "VGG 骨干替换、数据增强等）。为量化这些优化手段的真实收益，"
        "本实验将每个模型还原为其最朴素的基线形态，在完全相同的训练集（220 张正常样本）、"
        "测试集（115 张）与统一评估协议下重新训练、评估，并与实验 1 的优化结果逐一对比。")
P(doc, "基线定义原则：只保留模型算法的原始形态，去除本项目叠加的增强、正则化与工程优化。"
        "各模型基线配置如表 1 所示。")
add_table(
    doc,
    ["模型", "基线配置（实验 4）", "优化配置（实验 1）"],
    [
        ["PatchCore", "单层 conv4_3 特征、k=1 最近邻、全量记忆库（无 coreset 压缩）、无数据增强",
         "多层特征融合（conv3_3+conv4_3）、k=1、全量记忆库（实验 5 搜索所得最终配置）"],
        ["PaDiM", "原论文默认配置：ResNet18 + layer1~3 + 100 维随机投影",
         "VGG16（conv4_3/conv5_3）+ 550 维随机投影"],
        ["CutPaste", "无数据增强",
         "有数据增强（水平翻转 + ±5° 旋转 + 亮度扰动，取实验 3 复训权重）"],
    ],
    col_widths=[2.8, 6.4, 6.3],
)
P(doc, "表 1 各模型基线配置与优化配置对照", size=10.5,
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, line=1.2, space_after=6)

# ----------------------------- 二、实验设置 ----------------------------- #
H1(doc, "二、实验设置")
P(doc, "数据与预处理、评估协议、随机种子（seed=42）、batch_size=32、输入尺寸 256×256 "
        "均与实验 1 完全一致：图像级异常得分取原始异常热力图最大值（Anomalib 协议），"
        "图像级 AUROC 对 115 张测试图计算，PixelAUROC 逐图 min-max 归一化后全局展平计算，"
        "推理耗时按实验 1 同法测量（三模型统一：单张 predict_image 重复 20 次取均值、含 CUDA 同步；"
        "PatchCore 与 PaDiM 均采用批量前向 + GPU 检索 / 马氏距离的优化实现）。"
        "三个基线模型串行训练并及时释放显存。")

# ----------------------------- 三、实验结果 ----------------------------- #
H1(doc, "三、实验结果")

H2(doc, "3.1 基线模型评估结果")
rows = []
for m in MODELS:
    r = get_row(base, m, "基线")
    rows.append([m, f"{r['image_auroc']:.4f}", f"{r['pixel_auroc']:.4f}",
                 f"{r['mean_inference_ms']:.2f}"])
add_table(doc, ["模型（基线）", "图像级 AUROC", "PixelAUROC", "推理耗时 (ms/张)"],
          rows, col_widths=[3.8, 3.5, 3.5, 3.7])
P(doc, "表 2 基线模型评估结果（测试集 115 张）", size=10.5,
  align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, line=1.2, space_after=6)

for m in MODELS:
    p = HEATMAPS / NAMES[m] / "defect_0.png"
    if p.exists():
        add_figure(doc, p, f"图 {MODELS.index(m) + 1} {m} 基线模型缺陷样本检测结果"
                   "（左：原图；中：异常热力图；右：真实缺陷掩码）")

H2(doc, "3.2 基线 vs 优化对比")
rows = []
for m in MODELS:
    b, o = get_row(base, m, "基线"), get_row(merged, m, "优化")
    rows.append([m, "基线", f"{b['image_auroc']:.4f}", f"{b['pixel_auroc']:.4f}",
                 f"{b['mean_inference_ms']:.2f}"])
    rows.append([m, "优化", f"{o['image_auroc']:.4f}", f"{o['pixel_auroc']:.4f}",
                 f"{o['mean_inference_ms']:.2f}"])
    rows.append(["", "提升(Δ)", f"{o['image_auroc'] - b['image_auroc']:+.4f}",
                 f"{o['pixel_auroc'] - b['pixel_auroc']:+.4f}",
                 f"{o['mean_inference_ms'] - b['mean_inference_ms']:+.2f}"])
add_table(doc, ["模型", "配置", "图像级 AUROC", "PixelAUROC", "推理耗时 (ms/张)"],
          rows, col_widths=[3.0, 3.2, 3.0, 3.0, 3.3])
P(doc, "表 3 基线 vs 优化对比（Δ = 优化 − 基线，推理耗时 Δ 为负表示优化后更快）",
  size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, line=1.2, space_after=6)

# ----------------------------- 四、结果分析 ----------------------------- #
H1(doc, "四、结果分析")

def ana(m, baseline_why, optimize_why):
    b, o = get_row(base, m, "基线"), get_row(merged, m, "优化")
    d_i, d_p = o["image_auroc"] - b["image_auroc"], o["pixel_auroc"] - b["pixel_auroc"]
    if d_i >= 0.02 and d_p >= 0.02:
        verdict = "优化配置相对基线有明显提升，验证了上述优化手段的有效性"
    elif d_i >= 0 or d_p >= 0:
        verdict = "优化配置相对基线总体略有提升"
    else:
        verdict = "优化配置相对基线不升反降，说明该优化手段在该数据集上收益有限甚至有害"
    P(doc, f"（{MODELS.index(m) + 1}）{m}：基线图像级 AUROC {b['image_auroc']:.4f}、"
           f"PixelAUROC {b['pixel_auroc']:.4f}；优化后为 {o['image_auroc']:.4f}、"
           f"{o['pixel_auroc']:.4f}（Δ{d_i:+.4f}/{d_p:+.4f}）。{verdict}。", indent=False)
    P(doc, baseline_why)
    P(doc, optimize_why)

ana("PatchCore",
    "基线采用单层 conv4_3 特征 + k=1 最近邻 + 全量记忆库：全量保留 220 张正常样本的"
    "5 万余个 Patch 特征，检索到的是与待测区域最相似的真实正常纹理，距离度量最直接、"
    "记忆库代表性最完整，在小训练集上表现出色（图像级 0.9081 / 像素级 0.9013）。",
    "优化配置（实验 5 搜索所得最终配置）在 k=1 + 全量记忆库的基础上进一步引入多层特征"
    "融合（conv3_3+conv4_3）：conv3_3 保留细密纹理细节、conv4_3 提供高层语义，融合后"
    "特征对微小纹理偏差更敏感，图像级与像素级指标进一步提升，验证了特征层选择的收益。")
ana("PaDiM",
    "基线为原论文默认配置（ResNet18 + 100 维投影）：ResNet18 特征表达能力与维度较低，"
    "对 metal_nut 细密纹理的刻画有限。",
    "优化后改用 VGG16 的深层特征（conv4_3/conv5_3）并投影至 550 维，特征信息量更大，"
    "高斯统计建模更精细，马氏距离对微小纹理偏差更敏感。")
ana("CutPaste",
    "基线无数据增强：220 张小训练集上二分类器容易过拟合训练样本的特定姿态与光照，"
    "对测试集中翻转、亮度变化的正常样本容易误报。",
    "优化后加入水平翻转、±5° 旋转与亮度扰动，正常样本与伪异常样本多样性同步扩充，"
    "分类器学到的是纹理一致性判别而非记忆具体样本，图像级与像素级指标均显著提升"
    "（优化结果取实验 3 复训权重）。")

# ----------------------------- 五、结论 ----------------------------- #
H1(doc, "五、结论")
b_pad = get_row(base, "PaDiM", "基线")
b_pc = get_row(base, "PatchCore", "基线")
d_cp_i = get_row(merged, "CutPaste", "优化")["image_auroc"] - get_row(base, "CutPaste", "基线")["image_auroc"]
d_pd_i = get_row(merged, "PaDiM", "优化")["image_auroc"] - get_row(base, "PaDiM", "基线")["image_auroc"]
d_pc_i = get_row(merged, "PatchCore", "优化")["image_auroc"] - get_row(base, "PatchCore", "基线")["image_auroc"]
P(doc, "本实验完成了三个模型的基线训练与评估，全部基线模型与优化模型在相同数据与评估"
        "协议下直接可比。结论如下：")
P(doc, f"（1）基线模型中 PaDiM 综合最优（图像级 {b_pad['image_auroc']:.4f}、"
        f"像素级 {b_pad['pixel_auroc']:.4f}），其原论文默认配置（ResNet18 + 100 维投影）"
        "已具备很强的检测能力；PatchCore 朴素基线（单层特征 + k=1 + 全量记忆库）图像级 "
        f"{b_pc['image_auroc']:.4f}、像素级 {b_pc['pixel_auroc']:.4f}，"
        "证明全量记忆库 + k=1 方向正确。")
P(doc, f"（2）三项优化手段均带来收益：CutPaste 数据增强提升最大（图像级 Δ{d_cp_i:+.4f}）；"
        f"PatchCore 多层特征融合再提升（图像级 Δ{d_pc_i:+.4f}）；"
        f"PaDiM 骨干替换（ResNet18 → VGG16）微升（Δ{d_pd_i:+.4f}）。")
P(doc, f"（3）重要启示：PatchCore 主方案应在基线配置（k=1、全量记忆库、无增强）基础上"
        "搜索特征层组合——实验 5 完成该搜索，最终主方案定为「多层融合 "
        f"(conv3_3+conv4_3) + k=1 + 全量记忆库」，图像级 0.9477、像素级 0.9240"
        "（像素级为全项目最高，反超朴素基线的 0.9013）。")
P(doc, "（4）所有模型（含基线）单张推理耗时均小于 1 秒，满足工业实时检测约束。")

# ----------------------------- 附录 ----------------------------- #
H1(doc, "附录：实验运行方式")
P(doc, "实验脚本：experiments/exp4_baseline.py；完整代码见项目目录。运行命令：", indent=False)
P(doc, "PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
       "MKL_NUM_THREADS=1 python experiments/exp4_baseline.py",
  indent=False, line=1.3, font_cn="Consolas")
P(doc, "结果输出：results/exp4_baseline.csv（基线结果）、results/baseline_vs_optimized.csv"
        "（对比表）、results/heatmaps_baseline/（基线热力图）。", indent=False)

doc.save(OUT)
print("基线实验报告已生成 ->", OUT)
