---
status: done
filled: 2026-05-27
---

# Cell2Sentence 学习笔记

> Cell2Sentence（C2S）提出了一种将**单细胞转录组数据转化为自然语言句子**的创新方法，使得**预训练的大语言模型（如 GPT-2）可以直接理解和生成细胞数据**。这种方法桥接了 NLP 和单细胞基因组学两个领域，开创了"细胞→句子→LLM"的全新范式。

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
| **论文** | [Cell2Sentence] |
| **发布日期** | 2024 |
| **出版** | ICML |
| **架构** | **Fine-tuned GPT-2**（因果语言模型） |
| **预训练任务** | **自回归下一个 token 预测**（标准 LM 训练 + 细胞句子） |
| **输入** | "细胞句子"——按表达降序排列的基因名称序列 |
| **输出** | 生成的基因名称序列（可反解为表达谱） |
| **词表** | 基因名称（约 20K）+ GPT-2 原始词表 |
| **参数规模** | GPT-2 Small（~124M）等 |
| **预训练数据** | 多种 scRNA-seq 数据集转化的细胞句子 |
| **代码** | [GitHub](https://github.com/vandijklab/cell2sentence-ft) |
| **许可** | MIT |

### 核心思想

> **细胞可以像一句话一样被理解。** 将每个细胞的基因按表达量排序后，转化为一串基因名字符序列。LLM 本质上是序列模型，可以像处理英语句子一样"读懂"这些细胞句子，进而完成细胞注释、生成新细胞等任务。

---

## 2. 模型架构

### 2.1 整体架构图

```text
细胞表达谱              细胞句子
[0, 0, 1.2, 0, 5.6] → "MALAT1 B2M EEF1A1 ACTB GAPDH ..."
                                │
                          ┌─────┴─────┐
                          │   GPT-2   │  ← Fine-tune
                          └─────┬─────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            细胞类型预测             新细胞句子生成
               (zero-shot)          → 反解为表达谱
```

### 2.2 核心组件

#### 细胞→句子转换（Transform）

将每个细胞的表达谱转化为一串基因名称，基因按表达量从高到低排列：

```python
# 输入：表达向量
# 输出：基因名称序列
gene_names = adata.var_names
expressions = adata.X[cell_idx].toarray().flatten()
sorted_genes = gene_names[np.argsort(-expressions)]
cell_sentence = " ".join(sorted_genes[:max_length])
# 结果: "MALAT1 B2M EEF1A1 ACTB GAPDH TMSB4X ..."
```

#### GPT-2 Fine-tune

在标准 GPT-2 上微调，使用细胞句子作为训练数据，任务为标准的自回归语言建模。

#### 句子→表达反解（Inverse Transform）

将 LLM 生成的基因名称序列转回表达谱：

```python
generated = llm.generate("MALAT1 B2M", max_length=2048)
# → "MALAT1 B2M EEF1A1 ACTB GAPDH TMSB4X ..."
expression = inverse_transform(generated)
# → 稀疏表达向量（位置编码+表达值估计）
```

---

## 3. 核心创新

### 3.1 细胞到句子的转化（核心创新）

这是**首个将单细胞转录组数据转化为自然语言句子**的工作。细胞句子保留了基因表达排名信息，完全兼容 NLP pipeline。

### 3.2 利用预训练 LLM

无需从头训练模型，直接在 **GPT-2 等预训练 LLM 上微调**，借助 LLM 已有的语言理解能力学习生物学知识。

### 3.3 细胞生成能力

LLM 不仅可以从细胞句子中"读懂"细胞类型，还可以**生成全新的细胞句子**（相当于生成新的细胞表达谱），实现了细胞级别的生成式 AI。

### 3.4 与同类模型的对比

| 维度 | Cell2Sentence | Geneformer | scGPT | CELLama |
|------|--------------|-----------|-------|---------|
| **范式** | LLM fine-tune | BERT 预训练 | GPT 预训练 | 句子嵌入 |
| **模型** | GPT-2 | 自建 BERT | 自建 GPT | Sentence-BERT |
| **细胞生成** | **✅ 原生支持** | ❌ | 部分支持 | ❌ |
| **NLP 协同** | **天然兼容** | ❌ | ❌ | 部分 |

---

## 4. 数据预处理

```python
# 1. 读取细胞数据
adata = sc.read_h5ad("data.h5ad")
# 2. 转化为细胞句子
python transform.py --data_filepath data/adata.h5ad
# 输出: cell_sentences/*.txt, cell_sentences_hf/*.arrow
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| max_genes | 2048 | 每个细胞保留的基因数 |
| min_expression | 0 | 最低表达阈值 |

---

## 5. Tokenization 与输入编码

### 5.1 基因作为 Token

使用基因名称作为 token（如 "MALAT1"，"B2M"），与自然语言 token 共享词表空间。这需要扩展 GPT-2 的原始词表，加入基因名称。

### 5.2 排序编码

基因按表达量降序排列——**排序本身隐含了表达量信息**，无需直接编码数值。

### 5.3 特殊 Token

使用 GPT-2 的原始特殊 token 加自定义的句子分隔符。

---

## 6. 预训练/微调

### 6.1 训练数据

多种公开 scRNA-seq 数据集转化的细胞句子。

### 6.2 训练目标

```python
loss = CrossEntropyLoss(predicted_logits, target_gene_tokens)
# 标准自回归语言建模
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 模型 | GPT-2 (124M) |
| 学习率 | 2e-5 |
| Batch Size | 16 |
| 训练轮数 | 10 |
| 优化器 | AdamW |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| **零样本细胞类型注释** | LLM 文本嵌入 + 分类 | 有前景 |
| **细胞生成** | 从 LLM 生成细胞句子 → 反解为表达谱 | 新颖能力 |
| **微调注释** | 在标注数据上继续微调 | 与 SOTA 相当 |

---

## 8. 代码结构速览

```
cell2sentence-ft/
├── transform.py          # 细胞→句子 转换
├── train.py              # GPT-2 微调
├── generate.py           # 细胞生成
├── retrieve_example_data.py  # 示例数据下载
├── environment.yml       # 环境配置
└── data/                 # 数据目录
```

### 快速开始

```bash
git clone https://github.com/vandijklab/cell2sentence-ft
cd cell2sentence-ft
conda env create -f environment.yml
conda activate c2s
python retrieve_example_data.py
python transform.py
python train.py
```

---

## 9. 关键概念 Q&A

**Q: 为什么叫 "细胞句子"？**
A: 因为基因按表达量排序后的序列看起来像一个"句子"——MALAT1 B2M EEF1A1... 这些基因名就像句子中的单词，顺序编码了表达信息。

**Q: 细胞句子丢失了表达量的精确数值，这重要吗？**
A: 排序信息在很多任务中已足够（类似于 Geneformer 的做法），但确实丢失了量化信息。这是简化带来的取舍。

**Q: Cell2Sentence 能生成真实的细胞吗？**
A: 它可以生成统计上合理的细胞句子，但生成的细胞是否"真实"（具有生物学意义）是一个开放问题，需要实验验证。

---

## 10. 延伸阅读

- [CellTok](https://www.biorxiv.org/content/10.1101/2025.10.22.684047v1) — Cell2Sentence 的后续工作，改进 tokenization
- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) — 同期排序方法
- [scGPT](https://www.nature.com/articles/s41592-024-02201-0) — 生成式单细胞模型
