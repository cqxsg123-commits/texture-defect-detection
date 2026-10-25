# -*- coding: utf-8 -*-
"""生成《基于纹理特征的工业缺陷检测系统》实验报告 PPT（实验一至实验五）。

运行：D:\\anaconda\\envs\\defect_det\\python.exe generate_ppt.py
产物：基于纹理特征的工业缺陷检测系统_实验报告.pptx
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = Path(r"D:\基于纹理特征的工业缺陷检测系统")
RES = ROOT / "defect_detection" / "results"
FIG_DIR = RES / "ppt_figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
HEAT = RES / "heatmaps"
HEAT_B = RES / "heatmaps_baseline"

# ---------- 图表设计变量（dataviz 已验证调色板） ----------
BLUE, ORANGE = "#2a78d6", "#eb6834"          # 基线 / 优化 系列色
SURFACE, INK, SEC, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, AXIS_C = "#e1e0d9", "#c3c2b7"

# ---------- PPT 设计变量 ----------
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)      # 深蓝（标题/表头）
ACCENT2 = RGBColor(0x2A, 0x78, 0xD6)     # 蓝
ACCENT_D = RGBColor(0x10, 0x42, 0x81)    # 蓝 700
TINT_B = RGBColor(0xCD, 0xE2, 0xFB)      # 蓝 100（浅底）
TINT_B2 = RGBColor(0x9E, 0xC5, 0xF4)     # 蓝 200（封面浅字）
TINT_O = RGBColor(0xFC, 0xE8, 0xE1)      # 橙浅底
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK_C = RGBColor(0x0B, 0x0B, 0x0B)
SEC_C = RGBColor(0x52, 0x51, 0x4E)
MUTED_C = RGBColor(0x89, 0x87, 0x81)
ALT_C = RGBColor(0xF5, 0xF6, 0xF8)
HAIR = RGBColor(0xD9, 0xD9, 0xD9)
BLUE_C = RGBColor(0x2A, 0x78, 0xD6)
ORG_C = RGBColor(0xEB, 0x68, 0x34)
GOOD_C = RGBColor(0x00, 0x63, 0x00)
CRIT_C = RGBColor(0xD0, 0x3B, 0x3B)
FONT = "Microsoft YaHei"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 图表：基线 vs 优化 AUROC（两幅，各一个 y 轴） =================
MODELS = ["PaDiM", "CutPaste", "PatchCore"]
BASE_IMG = [0.9658, 0.7620, 0.9081]
OPT_IMG = [0.9668, 0.8925, 0.9477]
BASE_PIX = [0.8704, 0.6023, 0.9013]
OPT_PIX = [0.8759, 0.8548, 0.9240]


def bar_chart(path: Path, title: str, base: list, opt: list) -> None:
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    w = 0.34
    b1 = ax.bar(x - 0.19, base, w, color=BLUE, label="基线配置", zorder=3)
    b2 = ax.bar(x + 0.19, opt, w, color=ORANGE, label="优化配置", zorder=3)
    for rect in list(b1) + list(b2):          # 直接标注数值
        v = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=SEC)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS_C)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2,
              handlelength=1.2, handleheight=1.2)
    ax.text(0.995, 0.96, "metal_nut · n=115", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=12.5, color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


FIG_IMG = FIG_DIR / "auroc_image.png"
FIG_PIX = FIG_DIR / "auroc_pixel.png"
bar_chart(FIG_IMG, "图像级 AUROC：基线配置 vs 优化配置", BASE_IMG, OPT_IMG)
bar_chart(FIG_PIX, "PixelAUROC：基线配置 vs 优化配置", BASE_PIX, OPT_PIX)
print("[fig] AUROC 对比图已生成")

# ================= PPT 基础工具 =================


def set_run(r, size, bold=False, color=INK_C, font=FONT):
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = font
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", font)


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, spec in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        if "space_before" in spec:
            p.space_before = Pt(spec["space_before"])
        if "space_after" in spec:
            p.space_after = Pt(spec["space_after"])
        if "line" in spec:
            p.line_spacing = spec["line"]
        for txt, size, bold, color in spec["runs"]:
            r = p.add_run()
            r.text = txt
            set_run(r, size, bold, color)
    return tb


def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    return sp


def header(slide, num, title, sub=None):
    rect(slide, 0.55, 0.44, 0.16, 0.34, ACCENT2)
    runs = []
    if num:
        runs.append((num + "  ", 20, True, ACCENT2))
    runs.append((title, 20, True, INK_C))
    textbox(slide, 0.88, 0.36, 11.5, 0.55, [{"runs": runs}], anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, 0.55, 1.06, 12.23, 0.022, BLUE_C)
    if sub:
        textbox(slide, 0.55, 1.14, 12.2, 0.3, [{"runs": [(sub, 10.5, False, MUTED_C)]}])


def footer(slide, page):
    textbox(slide, 0.55, 7.14, 6, 0.3,
            [{"runs": [("基于纹理特征的工业缺陷检测系统 · 实验报告", 8, False, MUTED_C)]}])
    textbox(slide, 11.9, 7.14, 0.88, 0.3,
            [{"runs": [(f"{page:02d}", 9, True, ACCENT2)], "align": PP_ALIGN.RIGHT}])


def make_table(slide, x, y, w, rows, col_w=None, col_aligns=None, header_size=11.5,
               font_size=10.5, row_h=0.38, header_fill=ACCENT,
               bold_cells=None, color_cells=None, fill_cells=None):
    n_r, n_c = len(rows), len(rows[0])
    if col_w is None:
        col_w = [1.0 / n_c] * n_c
    if col_aligns is None:
        col_aligns = [PP_ALIGN.CENTER] * n_c
    bold_cells = bold_cells or {}
    color_cells = color_cells or {}
    fill_cells = fill_cells or {}
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                Inches(0.5 + row_h * (n_r - 1)))
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    total = sum(col_w)
    for j in range(n_c):
        tbl.columns[j].width = Emu(int(Inches(w) * col_w[j] / total))
    for i in range(n_r):
        tbl.rows[i].height = Inches(0.5 if i == 0 else row_h)
        for j in range(n_c):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
            elif i % 2 == 1:
                cell.fill.fore_color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = ALT_C
            if (i, j) in fill_cells:
                cell.fill.fore_color.rgb = fill_cells[(i, j)]
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = col_aligns[j]
            r = p.add_run()
            r.text = str(rows[i][j])
            if i == 0:
                set_run(r, header_size, True, WHITE)
            else:
                set_run(r, font_size, bold_cells.get((i, j), False),
                        color_cells.get((i, j), SEC_C))
    return tbl


def img_w_for_h(path: Path, h: float) -> float:
    w0, h0 = PILImage.open(path).size
    return w0 * h / h0


def pic(slide, path, x, y, h):
    w = img_w_for_h(path, h)
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    return w


def bullets(specs, size=12, color=INK_C, line=1.25, gap=8, align=PP_ALIGN.LEFT):
    paras = []
    for txt, *rest in specs:
        bold = rest[0] if rest else False
        paras.append({
            "runs": [("•  ", size, True, ACCENT2), (txt, size, bold, color)],
            "line": line,
            "space_after": gap,
            "align": align,
        })
    return paras


# ================= 构建 PPT =================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
prs.core_properties.title = "基于纹理特征的工业缺陷检测系统 实验报告"
prs.core_properties.author = "陈宇 202305630103"

PAGE = 0


def new_slide():
    global PAGE
    PAGE += 1
    return prs.slides.add_slide(BLANK)


def content_slide(num, title, sub=None):
    s = new_slide()
    header(s, num, title, sub)
    return s


# ---- 1 封面 ----
s = new_slide()
rect(s, 0, 0, 13.333, 7.5, ACCENT)
rect(s, 0, 7.28, 13.333, 0.22, ACCENT_D)
textbox(s, 1.17, 0.95, 11, 0.4,
        [{"runs": [("人工智能综合实践课程设计 · 课题编号 2", 13, False, TINT_B2)],
          "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 2.55, 11, 1.1,
        [{"runs": [("基于纹理特征的工业缺陷检测系统", 38, True, WHITE)],
          "align": PP_ALIGN.CENTER}])
rect(s, 4.67, 3.85, 4, 0.02, TINT_B)
textbox(s, 1.17, 4.05, 11, 0.45,
        [{"runs": [("实验报告 · 实验一至实验五", 17, False, TINT_B)],
          "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 5.05, 11, 0.4,
        [{"runs": [("汇报人：陈宇（202305630103）", 14, False, WHITE)],
          "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 5.5, 11, 0.4,
        [{"runs": [("指导教师：刘焕华", 14, False, WHITE)],
          "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 6.0, 11, 0.4,
        [{"runs": [("2026 年 9 月 28 日", 12, False, TINT_B2)],
          "align": PP_ALIGN.CENTER}])
for k, c in enumerate([BLUE_C, ORG_C, RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0xE8, 0x7B, 0xA4)]):
    rect(s, 0.6 + k * 0.3, 6.85, 0.22, 0.22, c)

# ---- 2 目录 ----
s = new_slide()
header(s, "", "目录")
toc = [
    ("01", "项目背景与研究目标"),
    ("02", "数据集与预处理"),
    ("03", "三模型技术路线（初版 5 路线筛选）· 统一评估协议"),
    ("04", "实验一：三模型横向对比"),
    ("05", "实验二：PatchCore 特征消融 · 实验三：数据增强"),
    ("06", "实验四：基线实验设计 · 基线 vs 优化结果"),
    ("07", "关键发现：PatchCore 基线反超 · AUROC 对比"),
    ("08", "实验五：PatchCore 优化搜索 · 最终配置确定"),
    ("09", "系统部署 · 问题与解决 · 总结展望"),
]
y = 1.55
for num, t in toc:
    textbox(s, 1.2, y, 1.1, 0.5, [{"runs": [(num, 17, True, ACCENT2)]}])
    textbox(s, 2.5, y + 0.06, 9.6, 0.5, [{"runs": [(t, 16, False, INK_C)]}])
    rect(s, 1.2, y + 0.52, 10.9, 0.012, HAIR)
    y += 0.62
footer(s, PAGE)

# ---- 3 项目背景与研究目标 ----
s = content_slide("01", "项目背景与研究目标")
rect(s, 0.55, 1.5, 0.08, 0.32, ACCENT2)
textbox(s, 0.75, 1.45, 5, 0.4, [{"runs": [("项目背景", 15, True, INK_C)]}])
textbox(s, 0.55, 2.0, 5.9, 3.4, bullets([
    ("工业质检中人工目检效率低、易漏检，自动化缺陷检测需求迫切",),
    ("金属工件纹理丰富，缺陷形态多样（划痕、弯折、色差、翻转）",),
    ("真实产线缺陷样本稀缺、标注成本高",),
    ("采用无监督异常检测：仅用无缺陷正常样本训练", True),
], size=12.5, gap=10))
rect(s, 6.9, 1.5, 0.08, 0.32, ORG_C)
textbox(s, 7.1, 1.45, 5, 0.4, [{"runs": [("研究目标", 15, True, INK_C)]}])
textbox(s, 6.9, 2.0, 5.9, 3.4, bullets([
    ("以 MVTec AD metal_nut 为对象，实现 5 种无监督检测技术路线并按准确率筛选保留前 3",),
    ("输出图像级异常得分 + 像素级缺陷掩码",),
    ("评估指标：图像级 AUROC、PixelAUROC、单张推理耗时（<1s）",),
    ("设计 5 组对照实验，为最终模型配置提供依据", True),
], size=12.5, gap=10))
rect(s, 0.55, 5.7, 12.23, 0.85, TINT_B)
textbox(s, 0.85, 5.88, 11.7, 0.55, [
    {"runs": [("任务定义：", 12, True, ACCENT_D),
              ("输入一张金属工件图像，输出图像级异常得分（是否为缺陷件）与像素级异常热力图（缺陷在哪）。",
               12, False, INK_C)]}])
footer(s, PAGE)

# ---- 4 数据集与预处理 ----
s = content_slide("02", "数据集与预处理")
textbox(s, 0.55, 1.3, 4.5, 0.4, [{"runs": [("MVTec AD · metal_nut 子集", 13, True, INK_C)]}])
rows = [
    ["项目", "数量"],
    ["训练集（仅正常样本）", "220 张"],
    ["测试集合计", "115 张"],
    ["　├ good（正常）", "22 张"],
    ["　├ bent（弯折）", "25 张"],
    ["　├ color（色差）", "22 张"],
    ["　├ flip（翻转）", "23 张"],
    ["　└ scratch（划痕）", "23 张"],
    ["像素级真值掩码", "缺陷图均带 GT"],
]
make_table(s, 0.55, 1.8, 4.6, rows, col_w=[2.4, 1.2], row_h=0.36, font_size=11,
           col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER], bold_cells={(1, 0): True, (2, 0): True})
textbox(s, 5.7, 1.3, 6.5, 0.4, [{"runs": [("预处理流水线", 13, True, INK_C)]}])
steps = [
    ("1", "统一缩放", "所有图像缩放至 256×256"),
    ("2", "归一化", "ImageNet 均值 / 方差标准化"),
    ("3", "无监督设定", "训练集仅保留正常样本，测试集带掩码"),
    ("4", "数据增强", "水平翻转、±5° 旋转、亮度扰动（实验三的变量）"),
]
y = 1.85
for num, t, d in steps:
    rect(s, 5.7, y, 0.42, 0.42, ACCENT2, shape=MSO_SHAPE.OVAL)
    textbox(s, 5.7, y + 0.03, 0.42, 0.36,
            [{"runs": [(num, 13, True, WHITE)], "align": PP_ALIGN.CENTER}])
    textbox(s, 6.3, y - 0.03, 6.4, 0.55, [
        {"runs": [(t + "  ", 12, True, INK_C), (d, 11, False, SEC_C)]}])
    y += 0.78
textbox(s, 5.7, 5.15, 6.9, 0.9, bullets([
    ("图像输入统一为 3×256×256，3 模型共用同一数据集",),
    ("中文路径兼容：np.fromfile + cv2.imdecode 读取",),
], size=11.5, gap=6))
textbox(s, 0.55, 5.75, 4.7, 1.0, [
    {"runs": [("缺陷类型示意", 11.5, True, INK_C)], "space_after": 4},
    {"runs": [("弯折 / 色差 / 翻转 / 划痕均表现为纹理局部异常，适合纹理特征检测方法",
               10.5, False, SEC_C)], "line": 1.25},
])
pic(s, HEAT / "padim" / "defect_0.png", 0.55, 6.15, 0.85)
footer(s, PAGE)

# ---- 5 三模型技术路线 ----
s = content_slide("03", "三模型技术路线（初版 5 路线筛选）",
                  sub="无监督设定：仅使用正常样本训练；初版实现 5 条技术路线，按测试集准确率保留前 3")
rows = [
    ["模型", "技术路线", "实现方式", "关键设计"],
    ["PatchCore（主方案）", "特征记忆库 + kNN", "自定义 PyTorch", "冻结 VGG16 多层特征融合 → 全量记忆库 → FAISS kNN"],
    ["PaDiM", "特征统计建模", "Anomalib 2.6.2", "多尺度特征高斯分布 + 马氏距离"],
    ["CutPaste", "自监督伪异常", "自定义 PyTorch", "剪切粘贴伪异常二分类 + GradCAM 定位"],
]
tbl = make_table(s, 0.55, 1.55, 12.23, rows, col_w=[1.8, 1.6, 1.5, 4.0], row_h=0.56,
                 font_size=11.5, col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER,
                                             PP_ALIGN.CENTER, PP_ALIGN.LEFT],
                 bold_cells={(1, 0): True}, fill_cells={(1, 0): TINT_B})
textbox(s, 0.55, 4.55, 12.2, 1.9, bullets([
    ("3 模型统一接口（train / predict_image），共用评估协议，保证对比公平",),
    ("初版另实现 AutoEncoder（重建 0.5293）与 DeepSVDD（单分类 0.4736），按准确率筛选后移除，源码归档 legacy/",),
    ("Anomalib 官方模型库不含 CutPaste，为自定义实现；PaDiM 依托 Anomalib 库（计划书要求）",),
], size=11.5, gap=8))
footer(s, PAGE)

# ---- 6 统一评估协议 ----
s = content_slide("04", "统一评估协议", sub="3 个模型共用同一评估模块，实验条件严格一致（控制变量）")
cards = [
    ("图像级得分", "取原始异常热力图的最大值作为图像级异常得分（Anomalib 协议）；聚合消融验证 95/99 分位替代不优于 max"),
    ("图像级 AUROC", "115 张测试图：得分 vs 正常/缺陷标签"),
    ("PixelAUROC", "逐图 min-max 归一化后全局展平，与像素级真值掩码比对"),
    ("推理耗时", "20 次重复取均值（含 CUDA 同步），约束：单张 <1s"),
]
pos = [(0.55, 1.55), (6.7, 1.55), (0.55, 3.6), (6.7, 3.6)]
for (t, d), (x, y) in zip(cards, pos):
    rect(s, x, y, 5.9, 1.75, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    rect(s, x, y, 0.09, 1.75, ACCENT2)
    textbox(s, x + 0.3, y + 0.2, 5.4, 0.4, [{"runs": [(t, 14, True, INK_C)]}])
    textbox(s, x + 0.3, y + 0.7, 5.3, 0.9, [{"runs": [(d, 11, False, SEC_C)], "line": 1.25}])
rect(s, 0.55, 5.65, 12.23, 0.85, TINT_B)
textbox(s, 0.85, 5.83, 11.7, 0.55, [
    {"runs": [("控制变量：", 12, True, ACCENT_D),
              ("同一数据集、同一图像尺寸、同一评估协议 —— 仅实验因子（特征/增强/配置）变化，差异可归因。",
               12, False, INK_C)]}])
footer(s, PAGE)

# ---- 7 实验一：三模型横向对比 ----
s = content_slide("05", "实验一：三模型横向对比", sub="metal_nut 测试集 115 张 · 批量前向 + GPU 检索提速 · 全部满足单张推理 <1s 约束")
rows = [
    ["模型", "图像级 AUROC", "PixelAUROC", "推理耗时", "说明"],
    ["PaDiM", "0.9668", "0.8759", "44.44 ms", "Anomalib 库实现，图像级最优"],
    ["PatchCore（主方案）", "0.9477", "0.9240", "25.96 ms", "实验四/五优化后像素级最优"],
    ["CutPaste", "0.9003", "0.6013", "12.45 ms", "自实现图像级 0.9003（实验三复训像素级 0.8548）"],
]
make_table(s, 0.55, 1.55, 12.23, rows, col_w=[1.7, 1.35, 1.35, 1.15, 4.0], row_h=0.45,
           font_size=11, col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER,
                                     PP_ALIGN.CENTER, PP_ALIGN.LEFT],
           bold_cells={(1, 1): True, (2, 2): True, (2, 0): True},
           color_cells={(1, 1): BLUE_C, (2, 1): BLUE_C, (2, 2): BLUE_C},
           fill_cells={(2, 0): TINT_B, (2, 1): TINT_B, (2, 2): TINT_B, (2, 3): TINT_B, (2, 4): TINT_B})
textbox(s, 0.55, 4.55, 12.2, 1.9, bullets([
    ("PaDiM 图像级最优 0.9668；PatchCore 经优化后图像级次优 0.9477、像素级最优 0.9240（见实验四/五）", True),
    ("三者即按准确率筛选后的最终模型集合（初版另含 AutoEncoder 0.5293、DeepSVDD 0.4736，已移除归档）",),
    ("推理提速（批量前向 + GPU 检索 / 马氏距离）：PatchCore 216.30 → 25.96 ms（8.3×）、PaDiM 271.94 → 44.44 ms（6.1×）、CutPaste 12.45 ms",),
], size=12, gap=8))
footer(s, PAGE)

# ---- 8 实验一：热力图样例 ----
s = content_slide("06", "实验一：热力图样例", sub="三联图从左到右：原图 / 异常热力图 / 真实缺陷掩码")
textbox(s, 0.55, 1.35, 6, 0.4, [{"runs": [("PaDiM（综合最优）", 12.5, True, INK_C)]}])
pic(s, HEAT / "padim" / "defect_0.png", 0.55, 1.8, 1.75)
textbox(s, 6.85, 1.35, 6, 0.4, [{"runs": [("PatchCore（主方案 · 最终配置）", 12.5, True, INK_C)]}])
pic(s, HEAT / "patchcore" / "defect_0.png", 6.85, 1.8, 1.75)
rect(s, 0.55, 4.15, 12.23, 1.15, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 4.32, 11.7, 0.9, [
    {"runs": [("对比：", 12, True, ACCENT_D),
              ("PaDiM 与 PatchCore 最终配置的热力图均准确定位缺陷、边界清晰，与其高 AUROC 一致（图像级 0.9668 / 0.9477，像素级 0.8759 / 0.9240）。",
               12, False, INK_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 9 实验二：PatchCore 消融 ----
s = content_slide("07", "实验二：PatchCore 特征与 k 值消融", sub="变量：单层 conv4_3 vs 多层特征融合；k ∈ {1, 3, 5, 9}")
rows = [
    ["配置", "图像级 AUROC", "PixelAUROC", "推理耗时"],
    ["单层 conv4_3 + k=5（默认）", "0.6994", "0.5395", "27.53 ms"],
    ["多层融合 + k=1", "0.4809", "0.4096", "38.89 ms"],
    ["多层融合 + k=3", "0.4946", "0.4357", "39.45 ms"],
    ["多层融合 + k=5", "0.4071", "0.4469", "40.84 ms"],
    ["多层融合 + k=9", "0.5088", "0.4622", "44.40 ms"],
]
make_table(s, 0.55, 1.6, 12.23, rows, col_w=[3.2, 1.5, 1.5, 1.3], row_h=0.48,
           font_size=11.5, col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER,
                                       PP_ALIGN.CENTER],
           fill_cells={(1, 0): TINT_B, (1, 1): TINT_B, (1, 2): TINT_B, (1, 3): TINT_B},
           bold_cells={(1, 0): True})
rect(s, 0.55, 4.75, 12.23, 1.3, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 4.95, 11.7, 1.0, bullets([
    ("coreset 0.1 压缩下：单层 conv4_3 + k=5 全面优于多层融合，图像级最多差 0.29",),
    ("该结论随记忆库压缩率反转：实验五全量记忆库下多层融合反超单层（见实验五）", True),
], size=12, gap=8))
footer(s, PAGE)

# ---- 10 实验三：数据增强消融 ----
s = content_slide("08", "实验三：数据增强消融", sub="变量：有无数据增强（水平翻转、±5° 旋转、亮度扰动）· 对象：CutPaste")
rows = [
    ["模型", "配置", "图像级 AUROC", "PixelAUROC"],
    ["CutPaste", "无增强", "0.7620", "0.6023"],
    ["CutPaste", "有增强", "0.8925", "0.8548"],
]
make_table(s, 0.55, 1.6, 12.23, rows, col_w=[2.0, 1.2, 1.7, 1.7], row_h=0.55,
           font_size=12, fill_cells={(2, 0): TINT_B, (2, 1): TINT_B, (2, 2): TINT_B, (2, 3): TINT_B},
           bold_cells={(2, 2): True, (2, 3): True}, color_cells={(2, 2): BLUE_C, (2, 3): BLUE_C})
rect(s, 0.55, 4.0, 12.23, 1.9, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 4.2, 11.7, 1.6, bullets([
    ("CutPaste 增强收益显著：图像级 +0.1305、像素级 +0.2525（0.6023 → 0.8548）", True),
    ("增强对自监督类方法更关键：伪异常 + 真实增强共同提升判别性",),
    ("初版五路线中 AutoEncoder 对增强不敏感（差异 <0.01）；模型筛选后实验三仅针对保留模型 CutPaste",),
], size=12, gap=8))
footer(s, PAGE)

# ---- 11 实验四：基线实验设计 ----
s = content_slide("09", "实验四：基线实验设计", sub="目的：逐一检验「优化设计」是否真实带来收益，为最终配置提供依据")
rows = [
    ["模型", "基线配置（朴素形态）", "优化配置"],
    ["PaDiM", "原论文默认：ResNet18 + layer1~3 + 100 维投影", "VGG16 双层特征 + 550 维投影"],
    ["CutPaste", "无数据增强", "有数据增强"],
    ["PatchCore", "单层 conv4_3 + k=1 + 全量记忆库（无 coreset 压缩）", "多层融合 (conv3_3+conv4_3) + k=1 + 全量记忆库"],
]
make_table(s, 0.55, 1.6, 12.23, rows, col_w=[1.3, 4.1, 3.5], row_h=0.6,
           font_size=11.5, col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
           bold_cells={(3, 0): True})
textbox(s, 0.55, 5.6, 12.2, 0.9, [
    {"runs": [("控制变量：", 12, True, ACCENT_D),
              ("基线 / 优化仅在目标设计点上不同，数据集、评估协议、训练流程完全一致（CutPaste 优化值取实验三复训权重）。",
               12, False, INK_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 12 实验四：基线 vs 优化结果 ----
s = content_slide("10", "实验四：基线 vs 优化结果", sub="Δ = 优化 − 基线 · 三项优化设计均带来正收益")
rows = [
    ["模型", "配置", "图像级 AUROC", "PixelAUROC", "Δ 图像级", "Δ 像素级"],
    ["PaDiM", "基线", "0.9658", "0.8704", "—", "—"],
    ["", "优化", "0.9668", "0.8759", "+0.0010", "+0.0055"],
    ["CutPaste", "基线", "0.7620", "0.6023", "—", "—"],
    ["", "优化", "0.8925", "0.8548", "+0.1305", "+0.2525"],
    ["PatchCore", "基线", "0.9081", "0.9013", "—", "—"],
    ["", "优化", "0.9477", "0.9240", "+0.0396", "+0.0227"],
]
tbl = make_table(s, 0.55, 1.55, 9.0, rows, col_w=[1.4, 1.0, 1.3, 1.3, 1.0, 1.0], row_h=0.36,
                 font_size=10.5,
                 color_cells={(2, 4): GOOD_C, (2, 5): GOOD_C, (4, 4): GOOD_C, (4, 5): GOOD_C,
                              (6, 4): GOOD_C, (6, 5): GOOD_C},
                 bold_cells={(2, 4): True, (2, 5): True, (4, 4): True, (4, 5): True,
                             (6, 4): True, (6, 5): True,
                             (5, 0): True, (5, 1): True, (5, 2): True, (5, 3): True,
                             (6, 0): True, (6, 1): True, (6, 2): True, (6, 3): True},
                 fill_cells={(5, 0): TINT_B, (5, 1): TINT_B, (5, 2): TINT_B, (5, 3): TINT_B,
                             (6, 0): TINT_B, (6, 1): TINT_B, (6, 2): TINT_B, (6, 3): TINT_B})
for r in (1, 3, 5):
    tbl.cell(r, 0).merge(tbl.cell(r + 1, 0))
    tbl.cell(r, 0).vertical_anchor = MSO_ANCHOR.MIDDLE
textbox(s, 9.9, 1.7, 2.9, 3.2, bullets([
    ("CutPaste：增强收益最大", True),
    ("PatchCore：多层融合再提升", True),
    ("PaDiM：对配置不敏感（骨干鲁棒）",),
], size=11, gap=9))
textbox(s, 0.55, 5.3, 12.2, 0.9, [
    {"runs": [("所有基线 / 优化配置推理均 <1s：批量前向 + GPU 检索 / 马氏距离提速后，最慢 PaDiM 44.44 ms、PatchCore 25.96 ms，均远低于约束。",
               11, False, SEC_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 13 AUROC 对比图 ----
s = content_slide("11", "基线 vs 优化：AUROC 对比图")
pic(s, FIG_IMG, 0.45, 1.5, 3.75)
pic(s, FIG_PIX, 6.85, 1.5, 3.75)
rect(s, 0.55, 5.65, 12.23, 1.1, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 5.82, 11.7, 0.85, [
    {"runs": [("要点：", 12, True, ACCENT_D),
              ("三项优化设计均带来正收益：CutPaste 增强提升最大（+0.13 / +0.25）；PatchCore 多层融合 +0.04 / +0.02；PaDiM 对配置不敏感（微升，骨干替换可行）。",
               12, False, INK_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 14 关键发现：PatchCore 三阶段优化 ----
s = content_slide("12", "关键发现：PatchCore 三阶段优化路径",
                  sub="0.6994（初版优化）→ 0.9081（朴素基线）→ 0.9477（最终配置）· 图像级 AUROC")
rect(s, 0.55, 1.75, 3.85, 1.85, TINT_O, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
textbox(s, 0.85, 1.9, 3.4, 0.35, [{"runs": [("① 初版优化配置 · 被否定", 10.5, True, RGBColor(0xB3, 0x48, 0x1F))]}])
textbox(s, 0.85, 2.28, 3.4, 0.55, [{"runs": [("coreset 0.1 + k=5 + 增强", 9.5, False, SEC_C)], "line": 1.2}])
textbox(s, 0.85, 2.85, 3.4, 0.55, [{"runs": [("0.6994 / 0.5395", 20, True, RGBColor(0xB3, 0x48, 0x1F))]}])
rect(s, 4.75, 1.75, 3.85, 1.85, TINT_B, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
textbox(s, 5.05, 1.9, 3.4, 0.35, [{"runs": [("② 朴素基线 · 方向验证", 10.5, True, ACCENT_D)]}])
textbox(s, 5.05, 2.28, 3.4, 0.55, [{"runs": [("单层 conv4_3 + k=1 + 全量记忆库", 9.5, False, SEC_C)], "line": 1.2}])
textbox(s, 5.05, 2.85, 3.4, 0.55, [{"runs": [("0.9081 / 0.9013", 20, True, ACCENT_D)]}])
rect(s, 8.95, 1.75, 3.83, 1.85, ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
textbox(s, 9.25, 1.9, 3.4, 0.35, [{"runs": [("③ 最终配置 · 实验五确定", 10.5, True, TINT_B2)]}])
textbox(s, 9.25, 2.28, 3.4, 0.55, [{"runs": [("多层融合 (conv3_3+conv4_3) + k=1 + 全量记忆库", 9.5, False, TINT_B2)], "line": 1.2}])
textbox(s, 9.25, 2.85, 3.4, 0.55, [{"runs": [("0.9477 / 0.9240", 20, True, WHITE)]}])
rect(s, 0.55, 3.95, 12.23, 1.15, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 4.1, 11.7, 0.95, [
    {"runs": [("原因分析：", 12, True, ACCENT_D),
              ("① coreset 0.1 压缩在 220 张小训练集上损失记忆库代表性；② k=5 平均稀释「最相似 patch」的判别信号 → 改全量记忆库 + k=1；③ 多层特征融合补充细节特征，最终 0.9477 / 0.9240。",
               12, False, INK_C)], "line": 1.25}])
rect(s, 0.55, 5.3, 12.23, 0.85, TINT_B, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
textbox(s, 0.85, 5.42, 11.7, 0.65, [
    {"runs": [("行动：", 12, True, ACCENT_D),
              ("否定初版优化 → 朴素基线验证方向（实验四）→ 特征层搜索（实验五）确定最终配置，写入 experiment.yaml 并部署 Gradio。",
               12, True, INK_C)], "line": 1.2}])
textbox(s, 0.55, 6.3, 6, 0.3,
        [{"runs": [("PatchCore 基线热力图样例：缺陷清晰定位", 10.5, False, MUTED_C)]}])
pic(s, HEAT_B / "patchcore" / "defect_0.png", 0.55, 6.6, 0.42)
footer(s, PAGE)

# ---- 15 实验五：PatchCore 优化搜索 ----
s = content_slide("13", "实验五：PatchCore 主方案优化搜索",
                  sub="在实验四已验证方向（全量记忆库、无增强）上搜索 k × 特征层，确定最终配置")
rows = [
    ["配置", "图像级 AUROC", "PixelAUROC"],
    ["多层融合 (conv3_3+conv4_3) + k=1 ★最终", "0.9477", "0.9240"],
    ["多层融合 + k=3", "0.9345", "0.9193"],
    ["多层融合 + k=5", "0.9228", "0.9171"],
    ["单层 conv4_3 + k=1", "0.9081", "0.9013"],
    ["单层 conv4_3 + k=3", "0.8974", "0.8961"],
    ["单层 conv4_3 + k=5", "0.8856", "0.8935"],
]
make_table(s, 0.55, 1.6, 6.9, rows, col_w=[3.6, 1.5, 1.5], row_h=0.44, font_size=11.5,
           col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
           fill_cells={(1, 0): TINT_B, (1, 1): TINT_B, (1, 2): TINT_B},
           bold_cells={(1, 0): True, (1, 1): True, (1, 2): True},
           color_cells={(1, 1): BLUE_C, (1, 2): BLUE_C})
textbox(s, 7.85, 1.65, 4.95, 4.3, bullets([
    ("k 越小越好：k=1 > k=3 > k=5（与实验四一致）", True),
    ("全量记忆库下多层融合反超单层（修正实验二结论）", True),
    ("最终配置：多层融合 + k=1 + 全量记忆库", True),
    ("图像级 0.9477 / 像素级 0.9240", True),
], size=11.5, gap=10))
rect(s, 0.55, 5.6, 12.23, 1.25, TINT_B, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 5.78, 11.7, 1.0, [
    {"runs": [("优化成果：", 12, True, ACCENT_D),
              ("较初版优化配置（0.6994 / 0.5395）图像级 +0.2483、像素级 +0.3845，像素级 0.9240 成为三模型第一。推理提速：批量前向 + GPU 检索 / 马氏距离，单张 216.30 → 25.96 ms（PaDiM 271.94 → 44.44 ms）。聚合消融：95 分位替代 max 在强模型上反而下降（0.9477 → 0.8754），保留 max 协议。配置已写入 experiment.yaml 并部署至 Gradio 演示系统。",
               12, False, INK_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 16 系统部署 ----
s = content_slide("14", "系统部署：Gradio 交互系统")
steps = [
    ("STEP 1", "启动系统", "python app.py"),
    ("STEP 2", "打开页面", "http://127.0.0.1:7860"),
    ("STEP 3", "上传图片", "切换 PatchCore / PaDiM / CutPaste"),
    ("STEP 4", "输出结果", "原图 · 热力图 · 掩码 · 合格判定"),
]
x = 0.55
for t, d1, d2 in steps:
    rect(s, x, 2.0, 2.8, 1.75, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.07)
    textbox(s, x + 0.25, 2.18, 2.3, 0.35, [{"runs": [(t, 10.5, True, ACCENT2)]}])
    textbox(s, x + 0.25, 2.6, 2.3, 0.4, [{"runs": [(d1, 13, True, INK_C)]}])
    textbox(s, x + 0.25, 3.05, 2.3, 0.6, [{"runs": [(d2, 10.5, False, SEC_C)], "line": 1.2}])
    if x < 9.5:
        rect(s, x + 2.88, 2.72, 0.22, 0.32, ACCENT2, shape=MSO_SHAPE.RIGHT_ARROW)
    x += 3.08
rect(s, 0.55, 4.35, 12.23, 1.05, TINT_B, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 4.5, 11.7, 0.85, [
    {"runs": [("技术栈：", 12, True, ACCENT_D),
              ("PyTorch 2.5.1（CUDA 12.4）· Gradio 界面 · 推理端三模型（PatchCore / PaDiM / CutPaste）· PaDiM 依托 Anomalib 2.6.2；热力图半透明叠加 + 得分实时输出，满足演示要求。",
               12, False, INK_C)], "line": 1.3}])
rect(s, 0.55, 5.55, 12.23, 1.15, ALT_C, line=HAIR, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
textbox(s, 0.85, 5.68, 11.7, 0.95, [
    {"runs": [("质检判定（参考 B2 组桌面系统）：", 12, True, ACCENT_D),
              ("自动阈值由训练集正常样本统计（PatchCore 留一法剔除自匹配），灵敏度倍数滑杆微调（默认 1.0 = 自动阈值），异常得分 ≥ 阈值判不合格。测试集判定表现：PaDiM 检出率 95.7% / 误报率 18.2%，PatchCore 73.1% / 4.5%。",
               12, False, INK_C)], "line": 1.3}])
footer(s, PAGE)

# ---- 17 问题与解决 ----
s = content_slide("15", "问题与解决")
rows = [
    ["问题", "解决方案"],
    ["CUDA 报错 driver version is insufficient", "驱动 551.86 仅支持 CUDA 12.4 → 安装 cu124 版 PyTorch（勿用 cu126/cu130）"],
    ["Anomalib PaDiM 特征层名错误", "改用 timm 阶段名 features.22 / features.30（conv4_3 / conv5_3；8GB 显存约束）"],
    ["cv2 中文路径读图失败", "np.fromfile + cv2.imdecode 绕过 imread 的中文路径限制"],
    ["OpenBLAS 内存分配失败", "限制 OPENBLAS_NUM_THREADS / OMP / MKL 线程数为 1"],
    ["GBK 编码错误（rich 进度条特殊字符）", "重定向日志时设置 PYTHONIOENCODING=utf-8"],
    ["显存不足", "3 模型串行训练、训练完即释放显存"],
]
make_table(s, 0.55, 1.55, 12.23, rows, col_w=[3.4, 6.2], row_h=0.62, font_size=11.5,
           header_size=12, col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT])
textbox(s, 0.55, 6.15, 12.2, 0.5, [
    {"runs": [("以上问题均已定位并解决，解决方案已沉淀至 README「常见问题」。", 11, False, SEC_C)]}])
footer(s, PAGE)

# ---- 18 总结与展望 ----
s = content_slide("16", "总结与展望")
rect(s, 0.55, 1.5, 0.08, 0.32, ACCENT2)
textbox(s, 0.75, 1.45, 5, 0.4, [{"runs": [("总结", 15, True, INK_C)]}])
textbox(s, 0.55, 2.0, 5.9, 4.2, bullets([
    ("初版实现 5 种技术路线，按准确率筛选保留 3 模型；完成 5 组对照实验（对比 / 消融 / 增强 / 基线 / 优化搜索）",),
    ("PatchCore 最终主方案：多层融合 + k=1 + 全量记忆库 → 0.9477 / 0.9240（像素级第一）", True),
    ("PaDiM 图像级最优（0.9668 / 0.8759）；PatchCore 为自实现最优路线",),
    ("推理加速：批量前向 + GPU 检索 / 马氏距离，单张 12.45~44.44 ms（PatchCore 提速 8.3×、PaDiM 6.1×），全部满足 <1s",),
], size=12, gap=10))
rect(s, 6.9, 1.5, 0.08, 0.32, ORG_C)
textbox(s, 7.1, 1.45, 5, 0.4, [{"runs": [("展望", 15, True, INK_C)]}])
textbox(s, 6.9, 2.0, 5.9, 4.2, bullets([
    ("扩展到更多类别（screw、cable 等）与更多材质表面",),
    ("桌面化部署：PyInstaller 打包独立 EXE（参考 B2 组方案），脱离 Python 环境一键运行",),
    ("模型轻量化与推理加速（ONNX / TensorRT），适配边缘部署",),
    ("引入半监督与少样本学习，进一步降低标注依赖",),
], size=12, gap=10))
footer(s, PAGE)

# ---- 19 致谢 ----
s = new_slide()
rect(s, 0, 0, 13.333, 0.18, ACCENT2)
rect(s, 0, 7.32, 13.333, 0.18, ACCENT2)
textbox(s, 1.17, 2.6, 11, 1.1,
        [{"runs": [("谢谢聆听", 44, True, INK_C)], "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 3.85, 11, 0.5,
        [{"runs": [("恳请各位老师批评指正", 16, False, SEC_C)], "align": PP_ALIGN.CENTER}])
textbox(s, 1.17, 4.9, 11, 0.4,
        [{"runs": [("汇报人：陈宇（202305630103）· 指导教师：刘焕华", 12, False, MUTED_C)],
          "align": PP_ALIGN.CENTER}])
for k, c in enumerate([BLUE_C, ORG_C, RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0xE8, 0x7B, 0xA4)]):
    rect(s, 6.07 + k * 0.3, 5.7, 0.22, 0.22, c)

OUT = ROOT / "基于纹理特征的工业缺陷检测系统_实验报告.pptx"
prs.save(OUT)
print(f"[ppt] 已保存：{OUT}")
print(f"[ppt] 共 {len(prs.slides.__iter__.__self__._sldIdLst)} 页")
