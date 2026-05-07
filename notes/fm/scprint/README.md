# scPRINT 学习笔记

> scPRINT (Single-cell Perturbation Response INference with Transformers) 是一个在 **5000 万细胞**上预训练的单细胞基础模型，专门设计用于**基因网络推断**。它属于当前参数量级最大的单细胞基础模型之一，聚焦于"理解基因如何协同工作"而非简单的细胞分类。

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
| **论文** | [scPRINT: pre-training on 50 million cells allows robust gene network predictions](https://www.nature.com/articles/s41467-025-58699-1), Nature Communications 2025 |
| **架构** | 基于 Performer 的 Transformer (线性复杂度注意力) |
| **预训练任务** | 三任务联合：基因掩码预测 + 去噪 + 细胞嵌入对齐 |
| **输入** | 基因表达 (pair tokenization) |
| **输出** | 基因网络 (Gene Regulatory Network) + 细胞嵌入 |
| **词表** | 约 20,000 个人类基因 |
| **参数规模** | ~300M |
| **预训练数据** | 5000 万细胞 (来自 CELLxGENE) |

### 核心思想

> "在巨大规模的细胞数据上进行预训练，使得模型不仅能理解单个基因的表达模式，更能推断**基因之间的调控关系网络**——即哪些基因调控哪些基因、在什么条件下调控。"

---

## 2. 模型架构

### 2.1 整体架构

```
Input: 基因表达对序列 (g_i, v_i), (g_j, v_j), ...
                  │
    ┌─────────────┴─────────────┐
    │   Gene + Value Embedding   │  ← 基因ID嵌入 + 表达值嵌入
    │   + Gene Program Encoding  │     基因程序先验 (GO/KEGG 通路)
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   Performer Encoder × L   │  ← 线性复杂度 Transformer
    │   (FAVOR+ 注意力)         │     O(N) 而非 O(N²)
    └─────────────┬─────────────┘
                  │
         ┌───────┴───────┐
         │               │
    ┌────┴────┐    ┌────┴────┐
    │  Cell   │    │  Gene   │
    │  Embed  │    │  Network│
    │ (CLS)   │    │  Head   │
    └─────────┘    └─────────┘
                      │
                Gene-Gene 注意力权重
                → GRN 推断
```

### 2.2 Performer 注意力

scPRINT 延续了 scFoundation 的 Performer 技术路线，使用 FAVOR+ 机制：

```python
# 线性复杂度注意力的关键优势
传统 Transformer:   O(L² × d)     L = 序列长度, d = 隐藏维度
Performer:         O(L × d × r)   r = 投影维度 (通常 r << L)

# 对 scRNA-seq 的意义：
# 支持更长输入序列 (~4096+ 基因)
```

### 2.3 基因程序先验

scPRINT 独特地整合了**基因程序 (Gene Program)** 信息：

```python
# 基于 GO/KEGG 通路的基因程序编码
gene_programs = load_go_kegg_pathways()  # 已知基因功能集合
gp_encoding = encode_gene_programs(gene_ids, gene_programs)
# 将通路先验注入注意力机制
```

---

## 3. 核心创新

### 3.1 三任务联合预训练

| 任务 | 描述 | 损失函数 |
|------|------|---------|
| **掩码基因预测** | 随机掩码 15% 基因，预测其身份和表达 | CrossEntropy + MSE |
| **表达去噪** | 对输入加入噪声，重构干净表达 | MSE |
| **细胞嵌入对齐** | 同一细胞在不同增强下的嵌入一致性 | 对比学习 (InfoNCE) |

### 3.2 注意力权重 → 基因调控网络

scPRINT 最具创新性的能力：**直接从注意力权重推断 GRN**

```python
# 提取注意力权重作为"基因调控强度"
attention_weights = model.get_attention(gene_ids)
# attention[i][j] → 基因 i 对基因 j 的调控强度
grn = build_grn(attention_weights.mean(axis=0))  # 跨头、跨层平均
```

### 3.3 大规模预训练 (5000 万细胞)

| 数据来源 | 细胞数量 | 覆盖组织 |
|---------|---------|---------|
| CELLxGENE | 4500 万 | >100 种人类组织 |
| GEO 补充 | 500 万 | 专门收集的扰动数据 |
| **总计** | **5000 万** | 泛组织、多平台 |

---

## 4. 数据预处理

### 4.1 Pipeline

```python
# 1. 从 CELLxGENE 加载数据
import cellxgene_census
census = cellxgene_census.open_soma()
adata = cellxgene_census.get_anndata(census, "Mus musculus")

# 2. 标准化
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 3. 基因筛选：使用 scPRINT 的固定基因集
gene_list = load_scprint_gene_list()  # ~20K 人类基因
adata = adata[:, adata.var_names.isin(gene_list)]
```

### 4.2 数据规模

| 数据来源 | 细胞数量 | 组织覆盖 |
|---------|---------|---------|
| CELLxGENE | 4500 万 | >100 种人类组织 |
| GEO 补充 | 500 万 | 专门收集的扰动数据 |

---

## 5. Tokenization 与输入编码

### 5.1 基因-表达值对编码

scPRINT 使用类似 scGPT 的**配对 tokenization**：

```
输入: (g_1, v_1), (g_2, v_2), ..., (g_N, v_N)
        │         │
    ┌───┘         └───┐
    ▼                   ▼
GeneEmbedding      ValueEmbedding
(gene_id → emb)   (value → emb)
    │                   │
    └───────┬───────────┘
            ▼
    Concatenated Embedding
```

### 5.2 基因程序先验

独特地整合了基因程序信息到编码中：

```python
gene_programs = load_go_kegg_pathways()  # 已知基因功能集合
gp_encoding = encode_gene_programs(gene_ids, gene_programs)
# 将通路信息注入注意力机制的偏置项
```

---

## 6. 预训练

### 6.1 三任务联合预训练

scPRINT 使用独特的三任务联合预训练策略：

| 任务 | 描述 | 损失函数 |
|------|------|---------|
| **掩码基因预测** | 随机掩码 15% 基因，预测其身份和表达 | CrossEntropy + MSE |
| **表达去噪** | 对输入加入噪声，重构干净表达 | MSE |
| **细胞嵌入对齐** | 同一细胞在不同增强下的嵌入一致性 | 对比学习 (InfoNCE) |

### 6.2 训练数据

| 属性 | 描述 |
|------|------|
| 数据规模 | 5000 万细胞 |
| 数据来源 | CELLxGENE (4500万) + GEO 补充 (500万) |
| 组织覆盖 | >100 种人类组织 |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| **基因网络推断** | 注意力权重 → 调控边 | SOTA，尤其对转录因子靶点 |
| **扰动预测** | 掩码扰动基因 → 预测下游效应 | 优于线性基线 |
| **细胞嵌入** | CLS token | 批次校正 + 细胞类型区分 |
| **基因重要性** | 注意力归因 | 识别关键调控基因 |

---

## 8. 代码结构速览

```
scPRINT/
├── model/
│   ├── gene_encoder.py      ← 基因编码器
│   ├── performer.py         ← Performer 注意力
│   ├── gene_program.py      ← 基因程序先验模块
│   └── heads.py             ← 三任务输出头
├── data/
│   ├── cellxgene_loader.py  ← CELLxGENE 数据加载
│   └── perturbation.py      ← 扰动数据集处理
├── train.py                 ← 三任务联合训练
└── evaluate.py              ← GRN 评估 + 下游任务
```

---

## 9. 关键概念 Q&A

### Q1: scPRINT vs scPRINT-2?

scPRINT-2 是 scPRINT 的升级版：
- 更大规模 (更多细胞、更多参数)
- 改进的 Benchmark 协议
- 增强的 GRN 推断精度

### Q2: 与 Geneformer 的 GRN 推断有何不同？

| 维度 | Geneformer | scPRINT |
|------|-----------|---------|
| GRN 方法 | 注意力权重 + 微调 | 注意力权重 + 零样本 |
| 预训练数据 | ~3000 万 / ~1 亿 | 5000 万 |
| 架构 | BERT | Performer (线性注意力) |
| 基因编码 | Rank Value Encoding | 标准 pair tokenization |
| 先验知识 | 无 | Gene Program 编码 |

---

## 10. 延伸阅读

- **[scPRINT preprint](https://www.biorxiv.org/content/10.1101/2024.04.04.588190v1)**：原始版本
- **[scPRINT-2 preprint](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v2)**：升级版本
- **[Performer 注意力](https://arxiv.org/abs/2009.14794)**：线性复杂度注意力机制
- **[CELLxGENE](https://cellxgene.cziscience.com/)**：预训练数据来源
