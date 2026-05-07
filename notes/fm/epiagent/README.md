# EpiAgent 学习笔记

> EpiAgent 是一个**单细胞表观基因组学基础模型**，专注于分析 scATAC-seq 等染色质可及性数据。它是继 scBERT/scGPT 等转录组模型之后，首个专门为**表观组学**设计的大规模基础模型。

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
| **论文** | [EpiAgent: foundation model for single-cell epigenomics](https://www.nature.com/articles/s41592-025-02822-z), Nature Methods 2025 |
| **架构** | Transformer (针对染色质数据定制的注意力机制) |
| **预训练任务** | 峰可及性掩码预测 (Peak Accessibility Masking) |
| **输入** | 染色质峰 (peak) 序列 + 可及性值 |
| **输出** | 峰可及性预测 / 细胞嵌入 |
| **词表** | 基因组峰区域 (~数万至数十万) |
| **数据** | scATAC-seq (单细胞 ATAC-seq) |
| **代码** | 待公开 |

### 核心思想

> "将染色质可及性数据建模为**峰 (peak) 的序列语言**，让模型学习哪些染色质区域在特定细胞类型中开放、哪些关闭，以及开放/关闭模式如何决定细胞身份。"

---

## 2. 模型架构

### 2.1 整体架构图

```
Input: [peak_1(acc), peak_2(acc), ..., peak_N(acc)]
                  │
    ┌─────────────┴─────────────┐
    │   Peak Encoding           │  ← 峰的位置编码 (染色体 + 坐标)
    │   + Accessibility Value   │     + 可及性值 (二值或连续)
    │   + Genomic Context       │     + 邻近基因上下文
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   Transformer Encoder × L │  ← 标准/线性注意力
    │   (定制的峰-峰注意力)      │     捕捉远程染色质交互
    └─────────────┬─────────────┘
                  │
         ┌───────┴───────┐
         │               │
    ┌────┴────┐    ┌────┴────┐
    │  峰可及 │    │  细胞   │
    │  性预测  │    │  嵌入   │
    └─────────┘    └─────────┘
```

---

## 3. 核心创新

### 3.1 表观组 vs 转录组基础模型对比

| 维度 | 转录组 FM (scGPT/scFoundation) | 表观组 FM (EpiAgent) |
|------|-------------------------------|---------------------|
| **输入数据** | 基因表达 (连续值) | 峰可及性 (二值/连续) |
| **词汇** | 基因符号 (~2万) | 基因组峰区域 (~10万+) |
| **序列长度** | 较短 (2000-4096) | 更长 (可达数万) |
| **先验知识** | 基因功能 (GO/KEGG) | 基因组注释 (ENCODE) |
| **预训练任务** | 表达重建 / 掩码预测 | 峰可及性预测 |

---

## 4. 数据预处理

### 4.1 Pipeline

```python
# 1. scATAC-seq 标准处理
import episcanpy as ep
adata = ep.read("fragments.tsv")
ep.pp.filter_cells(adata, min_features=1000)
ep.pp.filter_features(adata, min_cells=10)

# 2. 峰-基因对齐
peak_to_gene = map_peaks_to_genes(adata.var_names)  
# 每个峰关联最近的基因

# 3. 二值化 (scATAC-seq 通常是二值计数)
adata.X = (adata.X > 0).astype(float)
```

---

## 5. Tokenization 与输入编码

### 5.1 峰编码

每个染色质峰通过其**基因组位置**和**可及性值**编码：

```python
# 峰的表示: (染色体, 起始位置, 结束位置, 可及性值)
peak_token = peak_embedding(chrom, start, end) + \
             accessibility_embedding(binary_value)
```

### 5.2 与转录组模型的区别

转录组模型的"词汇"是基因名 (~2万)，表观组模型的"词汇"是基因组峰 (~10万+)，词汇量更大。

---

## 6. 预训练

### 6.1 预训练目标

**峰可及性掩码预测**：随机掩码部分峰的可及性值，模型预测其开放/关闭状态。

```python
loss = BinaryCrossEntropy(predicted_accessibility, true_accessibility)
# 仅计算被掩码的峰位置
```

---

## 7. 下游任务

| 任务 | 方法 |
|------|------|
| 细胞类型注释 | 峰可及性模式 → 细胞嵌入 → 分类 |
| 调控元件发现 | 注意力权重 → 关键调控峰 |
| 染色质交互预测 | 峰-峰注意力 → 远程交互 |
| 基因活性推断 | 峰-基因映射 → 基因表达预测 |

---

## 8. 代码结构速览

```
EpiAgent/
├── model/
│   ├── peak_encoder.py      ← 峰编码器
│   ├── epi_transformer.py   ← 表观组 Transformer
│   └── heads.py              ← 输出头
├── data/
│   ├── atac_preprocess.py   ← scATAC-seq 预处理
│   └── peak_to_gene.py      ← 峰-基因映射
└── train.py                 ← 预训练
```

> ⚠️ 代码待公开，以上为推测结构。

---

## 9. 关键概念 Q&A

### Q1: EpiAgent 可以用在空间数据上吗？

**A**: 理论上可以，如果空间转录组平台也捕获染色质可及性 (如 Spatial-ATAC-seq)。

### Q2: 同类模型还有哪些？

**A**: [EpiFoundation](https://www.biorxiv.org/content/10.1101/2025.02.05.636688v2)、[ChromFound](https://arxiv.org/abs/2505.12638) (NeurIPS 2025)、[Atacformer](https://www.biorxiv.org/content/10.1101/2025.11.03.685753v1)。

---

## 10. 延伸阅读

- **[EpiFoundation](https://www.biorxiv.org/content/10.1101/2025.02.05.636688v2)**：另一 scATAC-seq FM，通过峰-基因对齐预训练
- **[ChromFound](https://arxiv.org/abs/2505.12638)**：通用染色质可及性 FM (NeurIPS 2025)
- **[Atacformer](https://www.biorxiv.org/content/10.1101/2025.11.03.685753v1)**：基于 Transformer 的 ATAC-seq 分析
- **[scATAC-seq 教程](https://github.com/theislab/episcanpy)**：标准分析流程

