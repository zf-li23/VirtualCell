---
status: done
filled: 2026-05-27
---

# scELMo 学习笔记

> scELMo（Single-Cell Embeddings from Language Models）将 NLP 中经典的 ELMo 思路引入单细胞领域——利用预训练语言模型的各层表示，按任务自适应地融合多层嵌入，而非仅使用最后一层。这种方法简单有效，在细胞类型注释、聚类等任务上显著优于仅使用顶层嵌入的方法。

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
| **论文** | [scELMo](https://www.biorxiv.org/content/10.1101/2023.12.07.569910v1) |
| **发布日期** | 2023 |
| **出版** | bioRxiv |
| **架构** | 多层 Transformer + 自适应层融合 |
| **预训练任务** | 掩码基因预测（MLM） |
| **输入** | 基因 token（排序）+ 表达值编码 |
| **输出** | 多层细胞嵌入 + 融合嵌入 |
| **词表** | 约 20,000 个基因 |
| **参数规模** | 约 30M |
| **预训练数据** | 约 1000 万细胞 |
| **代码** | [GitHub]() |
| **许可** |  |

### 核心思想

> scELMo（Single-Cell Embeddings from Language Models）将 NLP 中经典的 ELMo 思路引入单细胞领域——利用预训练语言模型的各层表示，按任务自适应地融合多层嵌入，而非仅使用最后一层。这种方法简单有效，在细胞类型注释、聚类等任务上显著优于仅使用顶层嵌入的方法。

---

## 2. 模型架构


### 2.1 核心思想：多层融合

标准 BERT/Transformer 模型在推理时通常只使用最后一层输出作为细胞嵌入。scELMo 的核心洞见是：**不同层编码了不同级别的生物学信息**（底层→基因共表达模式，中层→通路信息，高层→细胞类型特征），因此按任务自适应融合多层表示可以获得更好的效果。

### 2.2 架构设计

- 使用标准 BERT 架构作为骨干网络
- 预训练时同时保存所有 Transformer 层的输出
- 下游任务时学习各层的加权融合权重

```python
# 各层嵌入的加权融合
elmo_embedding = Σ(weight_i * layer_i_output)
# weight_i 根据下游任务学习
```



## 3. 核心创新


### 3.1 ELMo 范式的单细胞适配

将 NLP 中 ELMo（Embeddings from Language Models）的多层融合策略成功移植到单细胞领域。

### 3.2 任务自适应的层权重

不同任务可以学习不同的层权重组合——细胞类型注释可能更依赖高层特征，而基因关系推断可能更需要中层特征。

### 3.3 与同类模型对比

| 维度 | scELMo | Geneformer | scBERT |
|------|--------|-----------|--------|
| 架构 | BERT + 多层融合 | BERT | BERT |
| 嵌入策略 | **多层加权融合** | 仅顶层 | 仅顶层 |
| 参数量 | ~30M | 6.5M | ~110M |



## 4. 数据预处理


与 Geneformer 兼容的预处理流程：表达量排序 → tokenization。



## 5. Tokenization 与输入编码


使用 Geneformer 的基因 tokenization 方案，基因按表达量从高到低排序作为 token 序列。



## 6. 预训练


- **数据**: 约 1000 万细胞
- **目标**: 标准掩码语言建模（MLM）
- **特点**: 保存所有 Transformer 层的输出用于下游融合



## 7. 下游任务


| 任务 | 方法 | 性能 |
|------|------|------|
| 细胞类型注释 | 多层融合嵌入 + 分类器 | 优于单层嵌入 |
| 细胞聚类 | 融合嵌入 + KMeans | ARI 提升 5-10% |
| 基因关系推断 | 融合嵌入 | 良好 |



## 8. 代码结构速览


```
scELMo/
├── seq2emb/              # 序列→嵌入 转换
├── Clustering/           # 聚类分析
├── 'Cell-type Annotation/'  # 注释
├── 'Batch Effect Correction/' # 批次校正
├── 'In silico treatment/'    # 扰动分析
├── 'Perturbation Analysis/'  # 扰动预测
└── 'Get outputs from LLMs/'  # LLM 输出提取
```



## 9. 关键概念 Q&A


**Q: scELMo 和 ELMo 的异同？**
A: 相同点都是多层加权融合；不同点在于 ELMo 使用 LSTM，scELMo 使用 Transformer BERT。

**Q: 多层融合为什么有效？**
A: Transformer 的不同层关注不同粒度的特征——底层关注基因共现模式，中层关注通路，顶层关注细胞状态。融合它们可以获得更全面的表示。



## 10. 延伸阅读


- [ELMo (Peters et al., 2018)](https://arxiv.org/abs/1802.05365) — 原始 ELMo 论文
- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) — scELMo 的骨干模型
- [BERT](https://arxiv.org/abs/1810.04805) — Transformer Encoder 基础


