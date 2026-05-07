# A visual–omics foundation model 学习笔记

> 这是一个**视觉-组学融合基础模型**，将组织病理学图像（H&E）与空间转录组数据统一到一个模型中。它直接解决了"如何将组织形态学与基因表达对齐"的问题。

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理](#4-数据预处理)
5. [Tokenization 与输入编码](#5-tokenization-与输入编码)
6. [预训练](#6-预训练)
7. [下游任务](#7-下游任务)
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics](https://www.nature.com/articles/s41592-025-02707-1), Nature Methods 2025 |
| **架构** | 双编码器 (视觉 ViT + 组学 MLP) + 跨模态注意力 |
| **预训练任务** | 跨模态对比学习 + 掩码预测 |
| **输入** | H&E 图像 (patch) + ST 基因表达 (spot) |
| **输出** | 融合嵌入 / 跨模态预测 |
| **核心能力** | 从 H&E → 预测 ST 表达 / 从 ST → 虚拟 H&E |
| **代码** | 待公开 |

### 核心思想

> "利用自监督学习对齐组织学图像和转录组数据，使得模型可以：(1) 仅从 H&E 图像预测基因表达模式，(2) 从表达模式推断组织形态，(3) 在两种模态间建立一一对应的空间映射。"

---

## 2. 模型架构

### 2.1 整体架构图

```
视觉模态:                          组学模态:
┌──────────┐                     ┌──────────────┐
│ H&E Patch│                     │ ST Expression│
│  768×768 │                     │  per spot    │
└────┬─────┘                     └──────┬───────┘
     │                                  │
┌────┴─────┐                     ┌──────┴───────┐
│ Vision   │                     │  Gene        │
│ Encoder  │                     │  Encoder     │
│ (ViT)    │                     │  (MLP)       │
└────┬─────┘                     └──────┬───────┘
     │                                  │
     └──────────┬───────────────────────┘
                │
         ┌──────┴──────┐
         │  Cross-modal │
         │   Attention  │  ← 跨模态注意力融合
         └──────┬──────┘
                │
         ┌──────┴──────┐
         │    Fusion   │
         │   Embedding │   ← 256/512 维联合嵌入
         └─────────────┘
```

### 2.2 核心组件

#### ViT 视觉编码器

```python
from transformers import ViTModel
vision_encoder = ViTModel.from_pretrained("google/vit-base-patch16-384")
# 输入: H&E 图像 patch → 输出: 视觉嵌入
```

#### 基因表达编码器

```python
class GeneExpressionEncoder(nn.Module):
    def __init__(self, n_genes, d_model):
        self.mlp = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, d_model)
        )
```

---

## 3. 核心创新

### 3.1 与同类模型的对比

| 模型 | 视觉-组学桥接方式 | 特点 |
|------|------------------|------|
| **本模型** | 双编码器 + 跨模态注意力 | 端到端融合，双向预测 |
| STPath | 生成式 ST+WSI | 生成融合，更重 WSI |
| spEMO | 多模态 FM | 支持空间多组学+病理 |

---

## 4. 数据预处理

### 4.1 Pipeline

```python
# 1. H&E 图像处理
import torchvision.transforms as T
patch_transform = T.Compose([
    T.Resize((768, 768)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. ST 表达处理
sc.pp.normalize_total(adata_st, target_sum=1e4)
sc.pp.log1p(adata_st)

# 3. 空间配准: 每个 H&E patch 对应一个 ST spot
patch_spot_pairs = spatial_registration(he_image, st_adata)
```

---

## 5. Tokenization 与输入编码

### 5.1 视觉模态

H&E patch 通过 ViT 的 patch embedding 层编码为视觉 token 序列。

### 5.2 组学模态

ST spot 的基因表达通过 MLP 编码为组学 token。

---

## 6. 预训练

### 6.1 预训练目标

| 任务 | 输入 | 预测目标 |
|------|------|---------|
| 跨模态对比 | H&E ↔ ST 配对的 spot | 拉近配对、推远非配对 (InfoNCE) |
| 掩码基因预测 | H&E + 部分掩码的 ST | 预测掩码基因表达 |
| 图像重建 | ST + 部分掩码的 H&E | 重建掩码图像区域 |

---

## 7. 下游任务

| 任务 | 方法 |
|------|------|
| H&E → 基因表达预测 | 视觉编码器 → 跨模态解码 |
| ST → 虚拟 H&E 生成 | 组学编码器 → 图像解码器 |
| 空间域识别 | 融合嵌入聚类 |
| 病理区域基因特征 | 注意力权重分析 |

---

## 8. 代码结构速览

```
visual-omics-fm/
├── model/
│   ├── vision_encoder.py   ← ViT 视觉编码器
│   ├── omics_encoder.py    ← 基因表达编码器
│   ├── cross_attention.py  ← 跨模态注意力
│   └── decoder.py          ← 跨模态解码器
├── data/
│   ├── st_preprocess.py    ← ST 数据预处理
│   └── image_patch.py      ← H&E patch 提取
├── train.py                ← 多任务预训练
└── evaluate.py             ← 跨模态评估
```

> ⚠️ 代码待公开，以上为推测结构。

---

## 9. 关键概念 Q&A

### Q1: 这个模型能用来做亚细胞空间分析吗？

**A**: 直接使用可能不够，因为它的设计分辨率是 spot 级别（10X Visium ~50μm）。但它的跨模态融合思路可以直接迁移到亚细胞分辨率。

### Q2: 训练时需要配对的 H&E+ST 数据吗？

**A**: 需要**同一组织切片**的 H&E 图像和 ST 数据精确配准。这对数据收集要求较高。

---

## 10. 延伸阅读

- **[STPath](https://www.nature.com/articles/s41746-025-02020-3)**：ST 与全切片图像的生成式融合
- **[spEMO](https://www.biorxiv.org/content/10.1101/2025.01.13.632818v3)**：多模态 FM 用于空间多组学+病理 (2026)

