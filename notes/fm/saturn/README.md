# SATURN 学习笔记

> SATURN (Single-cell ATlas of Universal Representation Network) 是一个用于**跨物种细胞嵌入对齐**的模型，通过对比学习在基因嵌入空间中建立不同物种的"语义对应"，实现零样本的跨物种细胞类型匹配。

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
| **论文** | [Toward universal cell embeddings: integrating single-cell RNA-seq datasets across species with SATURN](https://www.nature.com/articles/s41592-024-02191-z), Nature Methods 2024 |
| **架构** | 双编码器 (基因序列编码器 + 细胞表达编码器) + 对比学习 |
| **预训练任务** | 对比学习 (跨物种细胞类型对齐) |
| **输入** | 基因序列 (DNA) + 基因表达 (scRNA-seq) |
| **输出** | 跨物种细胞嵌入 |
| **词表** | 跨物种基因词汇表 (基于序列相似度) |
| **核心能力** | 跨物种 (人类↔小鼠↔斑马鱼↔...) 细胞类型匹配 |
| **代码** | [GitHub](https://github.com/snap-stanford/saturn) |

### 核心思想

> "利用基因**序列的跨物种保守性**——同源基因在不同物种中序列相似→功能相似→嵌入相似——在基因嵌入空间建立跨物种的'锚点'，使细胞嵌入空间自动对齐，无需配对训练数据。"

---

## 2. 模型架构

### 2.1 整体架构图

```
                     ┌──────────────────────┐
 人类细胞: g_h1, g_h2,... →  Cell Encoder → Cell_emb_h ───┐
                     └──────────────────────┘              │
                                                            ├──→ 对比学习
                     ┌──────────────────────┐              │    (拉近同类型
 小鼠细胞: g_m1, g_m2,... →  Cell Encoder → Cell_emb_m ───┘     跨物种对)
                     └──────────────────────┘

 基因嵌入初始化:
 DNA 序列 → Sequence Encoder (CNN) → 基因嵌入 (初始化 Cell Encoder)
```

### 2.2 核心组件

#### 基因序列编码器 (DNA Sequence Encoder)

```python
class DNAEncoder(nn.Module):
    """将基因的 DNA 序列映射为嵌入向量"""
    def __init__(self, d_model=1280):
        self.conv = nn.Conv1d(4, d_model, kernel_size=15)  # 4种碱基
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, dna_seq):
        # dna_seq: (batch, 4, seq_len) one-hot 编码
        h = self.conv(dna_seq)       # → (batch, d_model, L')
        h = self.pool(h).squeeze(-1) # → (batch, d_model)
        return h
```

---

## 3. 核心创新

### 3.1 DNA 序列引导的基因嵌入初始化

SATURN 的关键创新：利用 DNA 序列编码初始化基因嵌入，而非随机初始化。

```python
# 核心思想：同源基因序列相似 → 嵌入相似
gene_dna_seq = get_gene_sequence(gene_id)   # 基因启动子/编码序列
gene_seq_embed = dna_encoder(gene_dna_seq)  # CNN 编码
gene_embedding.weight[gene_id] = gene_seq_embed  # 初始化嵌入
```

### 3.2 对比 UCE 的差异

| 维度 | SATURN | UCE |
|------|--------|-----|
| 基因嵌入 | DNA 序列编码初始化 | 随机初始化 + 训练 |
| 跨物种对齐 | 序列相似 → 嵌入相似 | 对比学习对齐 |
| 预训练任务 | 对比学习 (InfoNCE) | 掩码预测 + 对比学习 |
| 零样本迁移 | ✅ | ✅ |

---

## 4. 数据预处理

### 4.1 Pipeline

```python
# 1. 获取每个基因的 DNA 序列 (启动子区域 ±2kb)
gene_seqs = get_promoter_sequences(genes, genome_fasta)

# 2. 对每个物种分别标准化
for adata in [adata_human, adata_mouse]:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
# 3. 筛选跨物种共有基因
common_genes = intersect_orthologs(adata_human.var_names, 
                                    adata_mouse.var_names)
```

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

每个基因的嵌入由其 DNA 序列通过 CNN 编码器初始化，在训练过程中微调。

### 5.2 表达值编码

与 scBERT 类似，使用连续 log 标准化表达值。

---

## 6. 预训练

### 6.1 预训练目标

```python
# InfoNCE 对比损失
loss = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )

# z_i, z_j: 同一细胞类型、不同物种的细胞嵌入 (正样本)
# z_k: 其他细胞嵌入 (负样本)
# sim: 余弦相似度
# τ: 温度参数
```

### 6.2 训练数据

| 物种 | 细胞数量 | 组织来源 |
|------|---------|---------|
| 人类 | 数十万 | PBMC, 胰腺, 大脑等 |
| 小鼠 | 数十万 | 同源组织 |
| 斑马鱼 | 少量 | 胚胎 |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| 跨物种细胞类型分类 | 人类训练 → 小鼠/斑马鱼零样本 | SOTA |
| 批次校正 | 嵌入空间统一 | 与 Harmony 相当 |
| 稀有细胞发现 | 嵌入空间密度分析 | 跨物种检测 |

---

## 8. 代码结构速览

```
SATURN/
├── model/
│   ├── dna_encoder.py      ← 基因序列编码 (CNN)
│   ├── cell_encoder.py     ← 细胞表达编码 (Transformer)
│   └── contrastive.py      ← 对比学习损失
├── data/
│   ├── preprocess.py       ← 跨物种预处理
│   └── genome.py           ← DNA 序列提取
├── train.py                ← 对比学习训练
└── evaluate.py             ← 跨物种评估
```

---

## 9. 关键概念 Q&A

### Q1: SATURN 需要每个物种都提供 DNA 序列吗？

**A**: 只需要参考基因组。对每个基因提取启动子序列后一次性编码，推理时不需要 DNA 序列。

### Q2: 与 UCE 选哪个？

**A**: 如果主要关注跨物种迁移，SATURN 的序列先验更直接；如果关注泛化性和多任务，UCE 更通用。

---

## 10. 延伸阅读

- **[UCE](https://github.com/snap-stanford/universal-cell-embeddings)**：类似跨物种细胞嵌入方法
- **[Geneformer](https://www.nature.com/articles/s41586-023-06139-9)**：Rank Encoding vs Sequence Encoding
- **[InfoNCE 损失](https://arxiv.org/abs/1807.03748)**：对比学习理论基础

