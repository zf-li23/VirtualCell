# scBERT 学习笔记

> scBERT 是一个基于 **BERT (Transformer Encoder)** 架构的大规模预训练深度语言模型，专门用于单细胞 RNA-seq 数据的细胞类型注释。作为最早将 BERT 范式引入单细胞领域的工作之一，scBERT 开创了"基因预训练 + 细胞类型微调"的经典路线。

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
| **论文** | [scBERT as a Large-scale Pretrained Deep Language Model for Cell Type Annotation of Single-cell RNA-seq Data](https://www.nature.com/articles/s42256-022-00534-z), Nature Machine Intelligence 2022 |
| **架构** | BERT (Transformer Encoder — 双向注意力) |
| **预训练任务** | Masked Language Modeling (MLM) — 掩码基因表达值预测 |
| **输入** | 基因 token (基于 Gene2Vec 嵌入) + 表达值 |
| **输出** | 被掩码位置的基因表达值 / 细胞类型概率 |
| **词表** | 约 17,000+ 基因 (基于 Ensembl ID) |
| **参数规模** | ~50M |
| **代码** | [https://github.com/TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT) |

### 核心思想

> "将单细胞转录组视为**基因序列 + 表达值的语言**，通过 BERT 的双向注意力学习每个基因在其细胞上下文中的语义角色，使模型理解：**在给定其他所有基因表达的前提下，某个基因应该有多高的表达**。"

---

## 2. 模型架构

### 2.1 整体架构

```
Input: [CLS] g_1(v_1), g_2(v_2), ..., g_N(v_N) [SEP]
                  │
       ┌──────────┴──────────┐
       │   Gene Embedding     │  ← Gene2Vec 预训练基因嵌入 (200维)
       │   (冻结 / 微调)       │    将基因 ID 映射为 d_model 维向量
       └──────────┬──────────┘
                  │
       ┌──────────┴──────────┐
       │   Expression +       │  ← 表达值嵌入 (通过 MLP 编码)
       │   Position Encoding  │     + 可学习位置编码
       └──────────┬──────────┘
                  │
       ┌──────────┴──────────┐
       │   Transformer × L   │  ← BERT Encoder (双向 Self-Attention)
       │   Encoder Layers     │     LayerNorm + Dropout + GELU
       └──────────┬──────────┘
                  │
       ┌──────────┴──────────┐
       │   Pre-training Head  │  ← MLM 预测头 (Linear + GELU + LayerNorm)
       │   / Fine-tune Head   │     微调时替换为分类头
       └──────────────────┘
```

### 2.2 组件详解

#### 基因嵌入层 (Gene Embedding)

```python
# 使用 Gene2Vec 预训练嵌入初始化
self.gene_embedding = nn.Embedding(vocab_size, d_model)  # vocab ≈ 17000
# Gene2Vec: 基于共表达网络的基因向量预训练
```

- 每个基因 ID 映射为一个 d_model 维向量
- Gene2Vec 预训练嵌入提供了基因之间的语义先验（功能相似的基因在嵌入空间中接近）
- 在预训练阶段可选择冻结或微调

#### 表达值编码

scBERT 将表达值作为连续特征输入：

```python
# 表达值经过 MLP 编码后加入基因嵌入
expression_features = mlp(expression_values)  # Linear + GELU + LayerNorm
final_embedding = gene_embedding + expression_features
```

#### 位置编码

使用可学习的位置编码（与原始 BERT 一致）：

```python
self.pos_encoder = nn.Embedding(max_seq_len, d_model)
position_ids = torch.arange(seq_len).unsqueeze(0)
pos_embedding = self.pos_encoder(position_ids)
```

---

## 3. 核心创新

### 3.1 Gene2Vec 预训练基因嵌入

scBERT 使用 Gene2Vec 初始化基因嵌入层 — 这是**将基因功能先验注入模型**的关键设计：

- **Gene2Vec 训练**：基于大量 scRNA-seq 数据的基因共表达矩阵
- **核心思想**：功能相关的基因（共同参与同一通路）在嵌入空间中距离更近
- **与随机初始化的区别**：提供了生物学先验，而非从零学习基因关系

### 3.2 Per-gene Masking 策略

不同于 NLP 中按 token 随机掩码，scBERT 的掩码策略**考虑了基因的结构化特性**：

```
原始序列:  [CLS] g_1(5.2), g_2(1.3), g_3(0.0), g_4(3.8), g_5(2.1), [SEP]
掩码后:   [CLS] g_1(5.2), [MASK],     g_3(0.0), g_4(3.8), [MASK],  [SEP]
```
- 15% 的基因位置被掩码
- 其中 80% 替换为 [MASK] token
- 10% 替换为随机基因
- 10% 保持原样

### 3.3 两阶段微调策略

```
第一阶段：全参数微调 (All-layer Fine-tuning)
  解冻所有层，对整个模型在目标数据集上微调

第二阶段：线性探测 (Linear Probing)
  冻结编码器，仅训练分类头 → 评估表示质量
```

---

## 4. 数据预处理

### 4.1 输入格式

```
每个细胞的输入:
  - 基因序列: 按 Gene2Vec 嵌入排序或随机排序
  - 表达值: log(1 + CPM) 标准化后的连续值
  - 最大长度: 截断至 N 个基因 (如 2000)
```

### 4.2 预处理 Pipeline

```
原始 count 矩阵
    │
    ▼
筛选高变基因 (top 2000)
    │
    ▼
Log(CPM) 标准化
    │
    ▼
按基因 ID 排序 → 固定顺序
    │
    ▼
补零 / 截断至固定长度
    │
    ▼
生成注意力掩码 (padding 位置为 0)
```

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

scBERT 使用 Gene2Vec 预训练嵌入初始化基因嵌入层：

```python
self.gene_embedding = nn.Embedding(vocab_size, d_model)
# 使用 Gene2Vec 权重初始化
gene_embedding.weight.data.copy_(gene2vec_weights)
```

### 5.2 表达值编码

表达值作为连续特征，通过 MLP 编码后加入基因嵌入：

```python
expression_features = mlp(expression_values)  # Linear + GELU + LayerNorm
final_embedding = gene_embedding + expression_features
```

### 5.3 特殊 Token

标准 BERT 特殊 token：`[CLS]`, `[MASK]`, `[PAD]`, `[SEP]`。

---

## 6. 预训练

### 5.1 预训练数据

- **训练数据**：来自 GEO 和 ArrayExpress 的约 50 万个人类 scRNA-seq 细胞
- **覆盖组织**：血液、脑、肝脏、肺、心脏等多种组织
- **数据多样性**：包含不同平台 (10X, Smart-seq2, Drop-seq) 的数据

### 5.2 预训练目标

**Masked Expression Prediction (MLM)**:
```python
# 对掩码位置预测 log(CPM+1) 表达值
loss = MSE(predicted_expression, true_expression)  # 仅计算掩码位置
```

### 5.3 预训练超参数

| 参数 | 值 |
|------|-----|
| 层数 | 6 (或 12) |
| 隐藏维度 | 512 (或 768) |
| Attention Head | 8 |
| 掩码比例 | 15% |
| 学习率 | 1e-4 |
| Batch Size | 64 |
| 训练步数 | 100K+ |
| 优化器 | AdamW |

---

## 7. 下游任务

### 6.1 细胞类型注释 (主要任务)

```
流程:
  1. 加载预训练 scBERT 权重
  2. 替换 [MASK] 预测头为线性分类头
  3. [CLS] token 的最后一层表示 → 细胞类型概率
  4. 在标注数据上微调
```

### 6.2 Zero-shot 迁移

scBERT 可以零样本迁移到未见过的数据集，利用预训练学到的基因表达模式直接进行细胞类型推断。

### 6.3 实验结果

| 数据集 | 任务 | 准确率 | 对比方法 |
|--------|------|--------|---------|
| PBMC 3K | 细胞类型分类 | ~0.92 | SVM: 0.85, MLP: 0.88 |
| 胰腺数据集 | 细胞类型分类 | ~0.90 | SVM: 0.81 |
| 小鼠大脑 | 跨物种迁移 | ~0.70 | 其他方法 < 0.60 |

---

## 9. 关键概念 Q&A

### Q1: scBERT 与 Geneformer 的区别？

| 维度 | scBERT | Geneformer |
|------|--------|------------|
| 架构 | BERT (双向) | BERT (双向) |
| 基因编码 | Gene2Vec 嵌入 | Rank Value Encoding |
| 表达值处理 | 连续值 MLP 编码 | 排序值 (rank) |
| 预训练数据 | ~50万细胞 | ~3000万细胞 |
| 预训练任务 | 表达值回归 (MSE) | 基因分类 (CE) |
| 主要应用 | 细胞类型注释 | 基因网络 / 扰动 |

### Q2: scBERT 有什么局限性？

1. **输入长度限制**：受 BERT 的 O(n²) 复杂度限制，只能输入 ~2000 个基因
2. **忽略基因排序**：基因顺序固定或随机，未利用表达排序信息
3. **单平台偏差**：主要在 10X 数据上训练，对 Smart-seq2 等全长数据泛化有限
4. **计算资源**：相比线性模型，需要 GPU 进行训练和推理

---

## 10. 延伸阅读

- **[BERT 原始论文](https://arxiv.org/abs/1810.04805)**：Devlin et al., NAACL 2019 — scBERT 的架构基础
- **[Gene2Vec 概念](https://pubmed.ncbi.nlm.nih.gov/29446786/)**：基因共表达网络的分布式表示
- **[scBERT vs 后续模型对比]**: scGPT (2024) 使用 GPT 架构 + 基因对 tokenization，在 33M 细胞上训练

---

## 9. 代码结构速览

```
scBERT/
├── model/
│   ├── gene_embedding.py      ← Gene2Vec 基因嵌入层
│   ├── bert_encoder.py        ← BERT Transformer 编码器
│   └── heads.py               ← MLM + 分类头
├── data/
│   ├── preprocess.py          ← 标准化 + 高变基因选择
│   └── dataset.py             ← PyTorch Dataset
├── pretrain/
│   └── pretrain.py            ← MLM 预训练脚本
├── finetune/
│   └── cell_type.py           ← 细胞类型注释微调
└── evaluate.py                ← 评估脚本
```
