# GeneCompass 学习笔记

> GeneCompass 是一个**知识引导的跨物种基础模型**，通过整合已知的基因调控知识（GO、KEGG、ENCODE）和跨物种单细胞数据，学习**通用的基因调控机制**。

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
| **论文** | [GeneCompass: Deciphering Universal Gene Regulatory Mechanisms with Knowledge-Informed Cross-Species Foundation Model](https://www.nature.com/articles/s41422-024-01034-y), Cell Research 2024 |
| **架构** | Transformer Encoder (BERT-style) + 知识注入模块 |
| **预训练任务** | MLM + 基因调控关系预测 |
| **输入** | 基因 token 序列（跨物种统一词汇表） |
| **输出** | 基因表达预测 / 调控关系评分 |
| **词表** | 跨物种基因集（人类 + 小鼠同源基因） |
| **参数规模** | ~110M |
| **预训练数据** | 跨物种单细胞数据（人类 + 小鼠） |
| **先验知识** | GO / KEGG / ENCODE 基因调控知识 |
| **代码** | 未公开（商业/学术限制） |

### 核心思想

> "将已知的基因调控知识**显式注入**模型训练过程，让基础模型不仅学习统计共表达模式，更理解**进化保守的基因调控机制**，实现跨物种的迁移。"

---

## 2. 模型架构

### 2.1 整体架构图

```
Input: 基因序列 [g_1, g_2, ..., g_N]
                  │
    ┌─────────────┴─────────────┐
    │   Knowledge-Guided        │  ← GO/KEGG 通路编码
    │   Gene Embedding          │     作为基因嵌入的偏置项
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   Transformer Encoder × L │  ← 标准 BERT 编码器
    │   (Multi-head Self-Attn)  │     LayerNorm + FFN
    └─────────────┬─────────────┘
                  │
         ┌───────┴───────┐
         │               │
    ┌────┴────┐    ┌────┴────┐
    │  MLM    │    │  GRN    │
    │  头     │    │  头     │
    └─────────┘    └─────────┘
```

### 2.2 核心组件

#### 知识引导的基因嵌入层

```python
# 概念伪代码
class KnowledgeGuidedEmbedding(nn.Module):
    def __init__(self, n_genes, d_model, n_pathways):
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.pathway_embed = nn.Embedding(n_pathways, d_model)
        
    def forward(self, gene_ids, pathway_ids):
        # 基础基因嵌入
        base = self.gene_embed(gene_ids)
        # 注入通路先验 (每个基因可能属于多个通路)
        pathway_bias = self.pathway_embed(pathway_ids).mean(dim=1)
        return base + pathway_bias  # 融合嵌入
```

---

## 3. 核心创新

### 3.1 知识注入 (Knowledge Injection)

在基因嵌入层显式加入通路先验，与 Geneformer/scGPT 的纯数据驱动方法形成对比：

| 维度 | GeneCompass | Geneformer/scGPT |
|------|-------------|------------------|
| 基因嵌入 | 知识引导 (GO/KEGG/ENCODE) | 纯数据驱动 (随机初始化) |
| 跨物种 | 显式同源映射 | 通常单物种 |
| 先验来源 | 人工整理的知识库 | 无外部先验 |

### 3.2 跨物种基因对齐

通过同源基因映射实现跨物种训练：
- 人类 ↔ 小鼠：基于 HGNC-MGI 同源映射
- 统一的基因词汇表：同源基因共享同一嵌入

### 3.3 双任务预训练

同时优化两个目标：掩码基因预测（通用表示）和基因调控关系预测（结构知识）。

---

## 4. 数据预处理

### 4.1 Pipeline

```python
# 1. 人类→小鼠同源基因映射
gene_mapping = load_homology_map("human_mouse_homology.tsv")
# 2. 跨物种标准化
adata_human = sc.pp.normalize_total(adata_human, target_sum=1e4)
adata_mouse = sc.pp.normalize_total(adata_mouse, target_sum=1e4)
# 3. 合并为统一词汇表空间
combined = union_genes(adata_human, adata_mouse, gene_mapping)
```

### 4.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_genes | ~20000 | 跨物种基因集大小 |
| 标准化 | log(CPM+1) | 标准 log 标准化 |

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

每个基因通过 Knowledge-Guided Embedding 编码，基础嵌入 + 通路先验偏置。

### 5.2 表达值编码

使用连续值编码（类似 scGPT 的 continuous 模式），表达值通过 MLP 映射后与基因嵌入融合。

### 5.3 特殊 Token

标准 BERT 特殊 token：`[CLS]`, `[MASK]`, `[PAD]`。

---

## 6. 预训练

### 6.1 预训练数据

| 数据来源 | 物种 | 细胞数量 |
|---------|------|---------|
| GEO 公共数据 | 人类 | 数百万 |
| GEO 公共数据 | 小鼠 | 数百万 |

### 6.2 预训练目标

```python
# 双任务联合损失
loss = λ₁ * MLM_loss + λ₂ * GRN_loss

# MLM: 标准掩码预测 (CrossEntropy)
# GRN: 基因调控关系预测 (Binary CrossEntropy)
# 基于已知的 GO/KEGG 调控关系作为监督信号
```

| 任务 | 损失 | 权重 λ |
|------|------|--------|
| MLM | CrossEntropy | 1.0 |
| GRN | BCE | 0.1 |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| 基因调控网络推断 | GRN head 预测 | 优于数据驱动方法 |
| 跨物种细胞类型注释 | 同源基因映射 + 分类 | 零样本迁移 |
| 通路活性分析 | 通路嵌入分析 | 可解释性 |

---

## 8. 代码结构速览

```
GeneCompass/
├── model/
│   ├── gene_embedding.py    ← 知识引导嵌入
│   ├── transformer.py       ← BERT 编码器
│   └── heads.py             ← MLM + GRN 输出头
├── data/
│   ├── preprocess.py        ← 跨物种预处理
│   └── homology_map.py      ← 同源基因映射
├── train.py                 ← 双任务训练
└── evaluate.py              ← GRN 评估
```

> ⚠️ 代码未公开，以上为基于论文描述推导的推测结构。

---

## 9. 关键概念 Q&A

### Q1: GeneCompass 与 Geneformer 的核心区别？

**A**: GeneCompass 显式注入 GO/KEGG 知识到基因嵌入中，而 Geneformer 使用纯数据驱动的 Rank Value Encoding。GeneCompass 的优势是跨物种对齐更好，劣势是依赖知识库的完整性。

### Q2: 代码为什么不公开？

**A**: Cell Research 论文，作者为华大基因团队，可能涉及商业限制。

---

## 10. 延伸阅读

- **[GO 知识库](http://geneontology.org/)**：Gene Ontology 基因功能分类
- **[KEGG 通路](https://www.genome.jp/kegg/)**：通路数据库
- **[ENCODE](https://www.encodeproject.org/)**：调控元件百科全书
- **[SATURN](https://www.nature.com/articles/s41592-024-02191-z)**：类似跨物种细胞嵌入方法

