# legacy/ — 已移除模型源码归档

本项目初版实现并对比了 5 种无监督异常检测技术路线，最终按 metal_nut 测试集
准确率筛选出前 3 名作为最终系统：

| 模型 | 图像级 AUROC | PixelAUROC | 去向 |
|---|---|---|---|
| PaDiM | 0.9668 | 0.8759 | ✅ 保留 |
| PatchCore（主方案） | 0.9477 | 0.9240 | ✅ 保留 |
| CutPaste | 0.9003 | 0.6013 | ✅ 保留 |
| AutoEncoder | 0.5293 | 0.1924 | ❌ 移除（本目录） |
| DeepSVDD | 0.4736 | 0.7364 | ❌ 移除（本目录） |

本目录存放被移除模型的源码，仅供备查：

- `autoencoder.py` — 重建基线（Denoising AE）
- `deep_svdd.py` — 深度单分类（AE 预训练 + 固定球心）

如需恢复某个模型：将对应文件移回 `src/models/`，在 `src/models/__init__.py`
的 MODEL_REGISTRY 中重新注册，并在 `configs/experiment.yaml` 的 models 段
补回对应超参数即可。
