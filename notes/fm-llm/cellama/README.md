---
status: done
filled: 2026-05-27
---

# CELLama 学习笔记

> CELLama（Cell Embedding Leveraging Language Model Ability）提出了一种**利用预训练 Sentence Transformer 直接生成细胞嵌入**的方法。它不训练新的单细胞模型，而是将细胞的基因表达和元数据转化为自然语言描述句子，然后用现成的句子嵌入模型（如 all-MiniLM-L6-v2）编码为细胞嵌入。这种方法天然**零样本**、**模态无关**，同时覆盖 scRNA-seq 和空间转录组学。

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
| **论文** | [CELLama] |
| **发布日期** | 2024 |
| **出版** | Advanced Science |
| **架构** | **Sentence Transformer**（预训练，如 all-MiniLM-L6-v2） |
| **预训练任务** | **无需单细胞训练**（使用 NLP 预训练模型） |
| **输入** | 细胞描述句子（基因名 + 表达 + 元数据） |
| **输出** | 细胞嵌入向量 |
| **词表** | NLP 词表 + 基因名称 |
| **参数规模** | ~22M（Sentence Transformer） |
| **预训练数据** | 各种公开 scRNA-seq + 空间转录组数据 |
| **代码** | [GitHub](https://github.com/portrai-io/CELLama) |
| **许可** | - |

### 核心思想

> **不要为单细胞数据训练新模型，让 NLP 模型来理解细胞。** 将细胞"翻译"成自然语言句子，然后用现成的文本嵌入模型直接编码——跨模态的对齐在句子层面完成，而非在模型训练层面。

---

## 2. 模型架构

### 2.1 整体流程

```text
细胞表达谱 + 元数据
        │
        ▼
  句子生成器 (Sentence Generator)
   "Cell from blood, type T cell, top genes: MALAT1, B2M..."
        │
        ▼
  Sentence Transformer (预训练)
   all-MiniLM-L6-v2 / etc.
        │
        ▼
  细胞嵌入 (Cell Embedding)
```

### 2.2 句子生成策略

CELLama 支持多种句子模板：
- **GeneRank**: "MALAT1 B2M EEF1A1 ACTB GAPDH ..." (仅基因排序)
- **Full Description**: "Cell type: T cell. Tissue: blood. Disease: normal. Top genes: MALAT1, B2M..."
- **Spatial Context**: "... nearest cells: cell_A (type X), cell_B (type Y)"

---

## 3. 核心创新

### 3.1 无需训练的零样本细胞嵌入

最大的创新：**不需要任何单细胞数据预训练**，直接用 NLP 的句子编码器嵌入细胞。

### 3.2 模态无关

同一模型同时适用于 scRNA-seq 和空间转录组学（只需在句子中包含空间上下文）。

### 3.3 与同类模型的对比

| 维度 | CELLama | Geneformer | scGPT | Cell2Sentence |
|------|---------|-----------|-------|--------------|
| **需要训练** | **❌** | ✅ | ✅ | ✅ |
| **零样本** | ✅ | ❌ | ❌ | ✅ |
| **跨模态** | scRNA+ST | scRNA | scRNA+多组学 | scRNA |
| **嵌入模型** | Sentence-BERT | 自建 BERT | 自建 GPT | GPT-2 |

---

## 4. 数据预处理

```python
# 从 h5ad 生成细胞描述句子
import cellama
sentences = cellama.make_sentence(
    adata,
    use_metadata=["cell_type", "tissue"],
    use_gene_rank=True,
    max_genes=512,
    spatial_context=True
)
# 输出: ["Cell from blood. Type T cell. Genes: MALAT1 B2M...", ...]
```

---

## 5. Tokenization

使用 Sentence Transformer 的标准 NLP tokenizer，基因名称和元数据文本作为普通 token。

---

## 6. 嵌入生成

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)  # 零样本细胞嵌入

# 可选：在特定数据集上微调
model.fit(train_sentences, train_labels)
```

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| 细胞类型注释 | 嵌入 + 分类器 | 与专用模型竞争 |
| 跨数据集整合 | 嵌入 + Harmony/scVI | 良好 |
| 空间邻居分析 | 嵌入 + 空间上下文 | 独特能力 |

---

## 8. 代码结构

```
CELLama/
├── cellama.py            # 主模块
├── _nn_model.py          # 神经网络模型
├── _examples_to_json.py  # 示例转换
└── requirements.txt
```

---

## 9. 关键概念 Q&A

**Q: CELLama 与其他模型在概念上最大不同？**
A: 其他模型试图"让单细胞数据适配深度学习架构"，CELLama 是"让 NLP 模型理解细胞数据"——思路完全相反。

**Q: 细胞描述句子如何保证质量？**
A: 句子的质量取决于模板设计和元数据的完整性。CELLama 提供了灵活的句子生成配置。

---

## 10. 延伸阅读

- [Sentence Transformers](https://www.sbert.net/) — CELLama 的基础模型
- [Cell2Sentence](https://arxiv.org/abs/2309.11222) — 另一种"细胞到句子"方法
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [CELLama] |
| **发布日期** | 2024 |
| **出版** | Advanced Science |
| **架构** | 如 BERT / GPT / VAE / GNN / 对比学习 |
| **预训练任务** | 如 MLM / 生成式 / 对比学习 / 自回归 |
| **输入** | 如基因 token + 表达值 |
| **输出** | 如细胞嵌入 / 基因表达预测 / 类型分类 |
| **词表** | 基因数量 |
| **参数规模** | 如 6.5M / 50M / 300M |
| **预训练数据** | 数据来源与规模 |
| **代码** | [GitHub](链接) |
| **许可** | MIT / Apache 2.0 / 自定义 |

### 核心思想

> **用一句话说明这个模型的角度**：它解决了什么问题？用了什么独特的方法？

---
