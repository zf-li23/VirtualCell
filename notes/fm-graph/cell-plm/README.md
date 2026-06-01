---
status: done
filled: 2026-05-27
id: cell-plm
title: CellPLM
category: fm-graph
code_url: https://github.com/OmicsML/CellPLM
---
# CellPLM 学习笔记

> CellPLM 是首个对细胞间关系进行编码的单细胞预训练语言模型。不同于仅编码单个细胞信息的 scGPT/Geneformer，CellPLM 通过基于图的预训练策略捕捉细胞-细胞相互作用关系，在细胞类型注释任务上持续超越现有方法，同时推理速度比现有预训练模型快 100 倍。论文发表于 ICLR 2024。

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
| **论文** | [CellPLM](https://openreview.net/forum?id=BKXvPDekud) |
| **发布日期** | 2024 |
| **出版** | ICLR 2024 |
| **架构** | Transformer + 图编码器（双编码器架构） |
| **预训练任务** | 掩码基因预测 + 细胞间关系预测 |
| **输入** | 基因表达谱 + 细胞-细胞关系图 |
| **输出** | 基因表达预测 / 细胞嵌入 |
| **词表** | ~30K 基因 |
| **参数规模** | 85M |
| **预训练数据** | 1M+ 细胞（多组织） |
| **代码** | [GitHub](https://github.com/OmicsML/CellPLM) |
| **许可** | BSD-2 |

### 核心思想

> CellPLM 是首个对细胞间关系进行编码的单细胞预训练语言模型。不同于仅编码单个细胞信息的 scGPT/Geneformer，CellPLM 通过基于图的预训练策略捕捉细胞-细胞相互作用关系，在细胞类型注释任务上持续超越现有方法，同时推理速度比现有预训练模型快 100 倍。论文发表于 ICLR 2024。

---

## 2. 模型架构

### 2.1 双编码器架构

CellPLM 采用双编码器架构，由两个核心组件构成：

**基因编码器（Gene Encoder）**: 基于 Transformer，处理每个细胞的基因表达谱。输入为基因 token + 表达值的 pair，类似于 scGPT 的编码方式。

**细胞-细胞图编码器（Cell-Cell Graph Encoder）**: 创新的图编码器，建模细胞之间的空间/功能关系。通过 kNN 构建细胞图，利用消息传递机制聚合邻居信息。

### 2.2 预训练策略

联合两个预训练目标：
1. **掩码基因预测**: 随机掩码部分基因的表达值，让模型从上下文恢复
2. **细胞间关系预测**: 预测两个细胞之间的关系类型（邻居/非邻居）

### 2.3 推理加速

CellPLM 在推理时使用 rapids 库加速，推理速度比 scGPT 和 Geneformer 快约 100 倍（作者宣称），主要通过优化的批处理和图计算实现。

## 3. 核心创新

### 3.1 超越单细胞的预训练

主流方法（scGPT, Geneformer）仅对单个细胞内的基因关系建模，CellPLM 首次将细胞间关系纳入预训练。

### 3.2 统一的预训练-微调范式

提出一个统一的框架，预训练和下游任务使用相同的双编码器结构，无需任务特定的架构修改。

### 3.3 高效的推理

利用 rapids 加速库实现 100 倍推理加速，使实际应用中的计算负担大幅降低。

## 4. 数据预处理

### 4.1 标准流程

1. 对数归一化（Log-normalization）
2. 高变基因筛选（top 2000-5000 HVGs）
3. 构建细胞-细胞 kNN 图（k=15-30）
4. 基因表达分箱（类似 scGPT 的 bin 策略）

### 4.2 数据格式

输入为 AnnData 对象，基因表达矩阵预处理后转换为基因 pair token 序列。

## 5. Tokenization 与输入编码

### 5.1 基因 Pair Token

输入格式类似于 scGPT：每个基因表示为 (gene_id, expression_bin) pair。
- 基因 ID 从预定义词表中查询
- 表达值被离散化为 10-50 个分箱

### 5.2 细胞关系 Token

细胞-细胞关系通过图的边列表表示，在预训练中作为辅助目标。

## 6. 预训练

### 6.1 预训练数据

使用 1M+ 来自多个组织的人类细胞数据进行预训练。

### 6.2 优化

- 优化器: AdamW
- 学习率: 预热 + 余弦退火
- Drop node rate: 0.3
- Mask node rate: 0.75

### 6.3 检查点

公开的预训练权重约 85M 参数，可通过 Dropbox 下载。

## 7. 下游任务

| 任务 | 方法 | CellPLM 表现 |
|------|------|-------------|
| 细胞类型注释 | 微调分类器 | SOTA，超越 scGPT/Geneformer |
| 跨数据集泛化 | 零样本迁移 | 良好的跨批次泛化 |
| 批次整合 | 嵌入对齐 | 竞争性表现 |

CellPLM 在 6 个基准数据集（PBMC12K, Pancreas, HLCA, Immune, Brain, Liver）上的细胞类型注释全面超越 scGPT 和 Geneformer。

## 8. 代码结构速览

```
CellPLM/
├── CellPLM/               # 核心源码
│   ├── encoder/           # 基因编码器
│   ├── decoder/           # 解码器
│   ├── model/             # Transformer + 图编码器
│   └── pipeline/          # 下游任务流水线
├── tutorials/             # Jupyter Notebook 教程
├── ckpt/                  # 预训练权重
└── data/                  # 数据目录
```

## 9. 关键概念 Q&A

**Q: CellPLM 和 scGPT 的核心区别是什么？**
A: scGPT 仅对单个细胞的基因表达建模（每个细胞独立处理），而 CellPLM 同时建模细胞-细胞关系和基因-基因关系。

**Q: 100 倍加速是如何实现的？**
A: 主要来自 rapids 库（cuDF, cuML）的 GPU 加速。

**Q: 支持多组织吗？**
A: 是的，预训练数据涵盖多个组织和器官。

## 10. 延伸阅读

- [论文](https://openreview.net/forum?id=BKXvPDekud)
- [代码](https://github.com/OmicsML/CellPLM)
- [博客解读](https://portal.valencelabs.com/blogs/post/cellplm-pre-training-of-cell-language-model-beyond-single-cells-wKScCQHIyicpXbx)
