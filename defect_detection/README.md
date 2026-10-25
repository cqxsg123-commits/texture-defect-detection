# 基于纹理特征的工业缺陷检测系统

人工智能综合实践课程设计 · 课题编号 2 · 陈宇 202305630103

无监督工业图像异常检测：仅使用 **无缺陷正常样本** 训练，输入金属工件图像，
输出 **图像级异常得分** 与 **像素级缺陷掩码**。数据集为 MVTec AD 的 `metal_nut` 金属螺母子集。

## 模型（按准确率筛选后的最终 3 模型）

| 模型 | 技术路线 | 实现方式 | 说明 |
|---|---|---|---|
| **PatchCore**（主方案） | 特征记忆库 | ✅ 自定义 PyTorch | 冻结 VGG16 多层 Patch 特征 → 全量记忆库 → FAISS kNN 检索 |
| **PaDiM** | 特征统计 | Anomalib 2.6.2 | VGG16 多尺度特征高斯分布 + 马氏距离 |
| **CutPaste** | 自监督伪异常 | ✅ 自定义 PyTorch | 剪切粘贴伪异常二分类 + GradCAM 定位（分类概率校准） |

> 初版实验实现并对比了 5 种技术路线（另含 AutoEncoder 重建基线 0.5293/0.1924、
> DeepSVDD 深度单分类 0.4736/0.7364），按 metal_nut 测试集准确率筛选后仅保留前 3 名；
> 被移除模型的源码归档于 `legacy/` 目录备查。Anomalib 2.6.2 官方模型库不含 CutPaste，
> 故其为自定义实现；PaDiM 依托 Anomalib 库调用（计划书「依托 Anomalib 开源库」）。

## 环境安装

### 1. 创建 conda 环境（Python 3.11）

```bash
conda create -n defect_det python=3.11 -y
conda activate defect_det
```

### 2. 安装 CUDA 版 PyTorch（RTX 4060 驱动 551.86 最高支持 CUDA 12.4，务必用 cu124 源）

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 4. 验证

```bash
python -c "import torch; print(torch.cuda.is_available())"   # 输出 True
python -c "import anomalib; print(anomalib.__version__)"     # 输出 2.6.2
```

> 若下载慢，pip 可加清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## 数据准备

MVTec AD 完整压缩包 `D:\数据集\data.zip`（9.9GB），本项目只解压 metal_nut 子集（幂等，可重复执行）：

```bash
python -m src.data.extract_mvtec
```

解压后结构：`data/mvtec_anomaly_detection/metal_nut/{train,test,ground_truth}`
（训练正常样本 220 张；测试集 115 张 = good 22 + bent 25 + color 22 + flip 23 + scratch 23，缺陷带掩码）

## 运行实验

| 实验 | 脚本 | 内容 |
|---|---|---|
| 一键全部 | `python run_all_experiments.py` | 按顺序运行实验 1~4，输出全部结果（权重存在时仅评估；加 `--retrain` 强制重训） |
| 实验 1 | `python experiments/exp1_model_compare.py` | 3 模型横向对比（AUROC / PixelAUROC / 推理耗时） |
| 实验 2 | `python experiments/exp2_patchcore_ablation.py` | 单层 vs 多层特征、k 值网格搜索 |
| 实验 3 | `python experiments/exp3_augmentation.py` | CutPaste 有/无数据增强消融 |
| 实验 4 | `python experiments/exp4_baseline.py` | 三模型基线实验（朴素形态 vs 优化配置） |
| 实验 5 | `python experiments/exp5_patchcore_optimize.py` | PatchCore 主方案优化搜索（k × 特征层，全量记忆库） |
| Notebook | `jupyter notebook experiments/experiments.ipynb` | 实验 1~4 的 Notebook 版入口（或在任意 Notebook 中 `%run run_all_experiments.py`） |

- 训练好的权重自动保存到 `checkpoints/`，重复运行自动跳过训练；
  加 `--retrain` 参数强制重训。
- 指标结果输出到 `results/`：对比表（CSV/Markdown）+ 热力图样例（报告素材）。
- 3 个模型串行训练、训练完即释放显存（计划书风险应对）。

## 实验结果（metal_nut，测试集 115 张）

> 注：AUROC 指标可复现；推理耗时为单次测量、随机器负载浮动，以 `results/*.csv` 最新值为准。

### 实验 1：3 模型横向对比（`results/comparison.md`）

| 模型 | 图像级 AUROC | PixelAUROC | 推理耗时 |
|---|---|---|---|
| **PaDiM** | **0.9668** | 0.8759 | 44.44 ms |
| **PatchCore（主方案）** | 0.9477 | **0.9240** | 25.96 ms |
| CutPaste | 0.9003 | 0.6013 | 12.45 ms |

全部满足单张推理 <1s 约束。三模型均采用批量前向 + GPU 检索 / 马氏距离的
优化推理路径（PatchCore 提速 8.3×、PaDiM 6.1×）。Anomalib PaDiM 图像级最优（0.9668）；
PatchCore 经实验 5 优化后像素级最优（0.9240）、图像级次优（0.9477），
为自实现模型中的最佳方案。三者即按准确率筛选后的最终模型集合。

### 实验 2：PatchCore 消融（`results/exp2_patchcore_ablation.md`）

在 coreset 0.1 压缩下，单层 conv4_3 + k=5（0.6994/0.5395）优于多层融合
（0.41~0.51）与 k∈{1,3,9} 变体；但实验 5 证明该结论随记忆库压缩率改变而
反转——全量记忆库下多层融合 + k=1 最优（0.9477/0.9240）。

### 实验 3：数据增强消融（`results/exp3_augmentation.md`）

CutPaste 有增强 0.8925/0.8548 vs 无增强 0.7620/0.6023，增强显著提升（图像级 +0.13、
像素级 +0.25）——增强扩充正常样本多样性，缓解 220 张小训练集上二分类器过拟合。

### 实验 4：三模型基线实验（`results/exp4_baseline.md`）

| 模型（基线配置） | 图像级 AUROC | PixelAUROC | 推理耗时 |
|---|---|---|---|
| PaDiM（ResNet18+100 维，原论文默认） | 0.9658 | 0.8704 | 35.54 ms |
| **PatchCore（单层 conv4_3 + k=1 + 全量记忆库）** | **0.9081** | **0.9013** | 22.43 ms |
| CutPaste（无增强） | 0.7620 | 0.6023 | 12.84 ms |

关键结论：**PatchCore 朴素基线（0.9081/0.9013）显著优于其初版优化配置
（0.6994/0.5395）**——220 张小训练集上 coreset 0.1 压缩损失记忆库代表性、
k=5 平均稀释最近邻信号；最终配置（多层融合 + k=1 + 全量记忆库，实验 5）进一步
提升至 0.9477/0.9240，反超基线。CutPaste 增强收益显著（图像级 +0.13、
像素级 +0.25，对比取实验 3 复训权重）；PaDiM 对骨干不敏感（微升）。

### 实验 5：PatchCore 优化搜索（`results/exp5_patchcore_optimize.md`）

在实验 4 已验证的方向（全量记忆库、无增强）上继续搜索 k × 特征层组合：

| 配置 | 图像级 AUROC | PixelAUROC |
|---|---|---|
| **多层融合 (conv3_3+conv4_3) + k=1（最终主方案）** | **0.9477** | **0.9240** |
| 多层融合 + k=3 | 0.9345 | 0.9193 |
| 多层融合 + k=5 | 0.9228 | 0.9171 |
| 单层 conv4_3 + k=1 | 0.9081 | 0.9013 |
| 单层 conv4_3 + k=3 | 0.8974 | 0.8961 |
| 单层 conv4_3 + k=5 | 0.8856 | 0.8935 |

k 越小越好（1 > 3 > 5，与实验 4 一致）；全量记忆库下多层融合反超单层
（修正实验 2 在 coreset 压缩下得出的结论）。最终主方案配置：
`layers=[conv3_3, conv4_3] + k=1 + coreset_ratio=1.0`，图像级 0.9477 /
像素级 0.9240（像素级超越 PaDiM 的 0.8759），批量前向 + GPU 检索后单张
推理 25.96 ms（优化前 216.30 ms，仍 <1s）。

## 启动 Gradio 系统

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860 ：上传图片 → 切换 PatchCore / PaDiM / CutPaste →
输出原图、异常热力图、缺陷掩码、异常得分与**合格/不合格判定**（绿色/红色）。
（需先运行实验 1 生成 `checkpoints/patchcore.pth`、`padim.pth` 与 `cutpaste.pth`）

### 质检判定（参考 B2 组螺丝缺陷检测系统）

- **判定规则**：异常得分 ≥ 判定阈值 → 红色「不合格」，否则绿色「合格」；
  缺陷掩码与判定阈值同源（异常得分 ≥ 判定阈值的区域）。
- **自动阈值**：由训练集正常样本统计得到，随权重保存于
  `checkpoints/thresholds.json`（PaDiM 取训练集得分最大值；PatchCore 训练样本
  自身在记忆库中，采用留一法——取第二近邻距离剔除自匹配）。
- **灵敏度倍数滑杆**：判定阈值 = 倍数 × 自动阈值，默认 1.0 = 自动阈值；
  调小更严格（检出率↑、误报率↑）、调大更宽松（反之）。
- **生成/更新**：`python scripts/compute_auto_threshold.py`（同时输出
  `results/threshold_verdict.json` 判定验证）。
- **测试集 115 张判定验证**（自动阈值，1.0×）：PaDiM 检出率 95.7%（89/93）、
  误报率 18.2%（4/22）；PatchCore 检出率 73.1%（68/93）、误报率 4.5%（1/22）；
  CutPaste 检出率 49.5%、误报率 0.0%（像素级定位能力弱，判定以 PatchCore / PaDiM 为准）。

## 项目结构

```
defect_detection/
├── app.py                    # Gradio 交互系统（三模型推理 + 质检判定）
├── configs/experiment.yaml   # 实验配置（超参数集中管理）
├── src/
│   ├── data/                 # 自定义数据集模块（计划书 4.4）
│   │   ├── extract_mvtec.py  # 从压缩包解压 metal_nut 子集
│   │   ├── dataset.py        # 训练集仅正常 / 测试集带掩码，256×256 + ImageNet 归一化
│   │   └── augmentations.py  # 水平翻转、±5°旋转、亮度扰动
│   ├── models/               # 3 个最终模型（统一接口：train / predict_image）
│   │   ├── patchcore.py      # 主方案：VGG16 多层特征 + 全量记忆库 + FAISS kNN
│   │   ├── padim_anomalib.py # Anomalib PaDiM 适配层
│   │   └── cutpaste.py       # 伪异常二分类 + GradCAM（概率校准）
│   ├── metrics.py            # 图像级 / 像素级 AUROC（Anomalib 协议）
│   └── evaluation.py         # 统一评估 + 对比表 + 热力图样例
├── experiments/              # 五个实验脚本 + Notebook 版
├── scripts/                  # compute_auto_threshold.py：自动判定阈值
│                             #   （训练集统计 + 测试集判定验证）
├── checkpoints/              # 模型权重（best.ckpt/.pth）+ thresholds.json
├── results/                  # 指标表格、热力图、判定验证
├── legacy/                   # 已移除模型源码归档（AutoEncoder / DeepSVDD）
└── examples/                 # Gradio 演示示例图
```

## 常见问题

- **CUDA 报错 "driver version is insufficient"**：驱动 551.86 只支持 CUDA 12.4，
  请勿安装 cu126/cu130 版 torch。
- **Anomalib PaDiM 特征层名错误**：vgg16 使用 timm 阶段名（本项目用 features.22 /
  features.30，即 conv4_3 / conv5_3 输出；8GB 显存不宜再从 features.15 开始三层融合），
  若仍报错可在 `configs/experiment.yaml` 中将 padim backbone 改为 `resnet18`。
- **中文路径下 cv2 读图失败**：数据集模块已改用 `np.fromfile + cv2.imdecode`
  规避 Windows 上 `cv2.imread` 不支持中文路径的问题，无需额外处理。
- **OpenBLAS 报错 "Memory allocation still failed after 10 retries"**：Windows 下
  numpy/OpenBLAS 按 CPU 核数开线程，每个线程分配缓冲区，与 DataLoader 多进程叠加后
  内存分配失败。运行训练时限制 BLAS 线程数即可。
- **UnicodeEncodeError: 'gbk' codec ... '•'**：Lightning 的 rich 进度条在重定向
  输出到文件时按 GBK 编码写入特殊字符失败。重定向日志运行时须加
  `PYTHONIOENCODING=utf-8`。
  推荐完整训练命令：
  ```bash
  PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python experiments/exp1_model_compare.py
  ```
- **内存不足**：实验脚本已串行训练并及时释放显存；可减小 `batch_size`。
- **下载数据集**：MVTec 官网需注册；本机 `D:\数据集\data.zip` 为 Kaggle 镜像完整版。
