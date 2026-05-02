# Geneformer 学习笔记

> Geneformer 是一个基于 Transformer 架构的**单细胞转录组基础模型**（foundational model），通过在大量人类单细胞转录组数据上进行自监督预训练，学习基因调控网络动力学。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [已下载的四个模型对比](#3-已下载的四个模型对比)
4. [核心创新：Rank Value Encoding](#4-核心创新rank-value-encoding)
5. [数据流 Pipeline](#5-数据流-pipeline)
6. [预训练 (Pretraining)](#6-预训练-pretraining)
7. [下游任务 (Downstream Tasks)](#7-下游任务-downstream-tasks)
8. [代码结构速览](#8-代码结构速览)
9. [快速上手示例](#9-快速上手示例)
10. [关键概念 Q&A](#10-关键概念-qa)
11. [延伸阅读](#11-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [Transfer learning enables predictions in network biology](https://rdcu.be/ddrx0), Nature 2023 |
| **V2 论文** | [Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology](https://rdcu.be/famFk), Nature Computational Science 2026 |
| **架构** | BERT (BertForMaskedLM) — Transformer Encoder |
| **预训练任务** | Masked Language Modeling (MLM) — 掩码基因预测 |
| **输入** | 单细胞转录组的 Rank Value Encoding（排序编码） |
| **输出** | 预测被掩码位置应是什么基因 |
| **词表** | ~20K~25K 个基因（蛋白编码基因 + miRNA） |
| **许可** | Apache 2.0 |

### 核心思想

> "将每个细胞的转录组表示为**基因的排序列表**，通过 MLM 任务让模型学到：在特定细胞状态下，基因之间的**上下文关系**和**网络层级结构**。"

---

## 2. 模型架构

Geneformer 使用标准的 **BERT (Transformer Encoder)** 架构，关键组件：

### 2.1 架构参数

```
Input:  [CLS] gene_rank_1, gene_rank_2, ..., gene_rank_N [EOS]
                │
        ┌───────┴───────┐
        │ Token Embedding│  ← 将每个基因 token 映射为向量
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │ Position Embed │  ← 基因的顺序位置（排序位置）
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │ N × Transformer│
        │  Encoder Layer │  ← 多头自注意力 + FFN
        └───────┬───────┘
                │
Output: [hidden states for each position]
```

### 2.2 激活函数

- **使用 ReLU 而非 GELU**：这是与标准 BERT 的一个关键区别。
  - ReLU 在单细胞数据上表现更好，因为基因表达数据通常是稀疏且非负的。
  - GELU 更平滑但对稀疏数据可能不够鲁棒。

### 2.3 Attention 的可解释性

> 模型在自注意力权重中**编码了基因网络的层级结构**（network hierarchy）。

- 通过分析注意力权重，可以识别哪些基因是调控网络中的**上游调控因子**（high in-degree）。
- 这是一个**零样本（zero-shot）**的能力 —— 模型从未被显式训练过预测网络层级。

---

## 3. 已下载的四个模型对比

| 模型 | 参数量 | 训练数据 | 输入长度 | 词表大小 | hidden_size | layers | heads | 特点 |
|------|--------|----------|----------|----------|-------------|--------|-------|------|
| **Geneformer-V1-10M** | 10M | ~30M 细胞 (Genecorpus-30M) | 2048 | 25,426 | 256 | 6 | 4 | 原始模型 (2021) |
| **Geneformer-V2-104M** | 104M | ~104M 细胞 (Genecorpus-104M) | 4096 | 20,275 | 768 | 12 | 12 | 升级版 (2024) |
| **Geneformer-V2-316M** | 316M | ~104M 细胞 | 4096 | 20,275 | 1152 | 18 | 18 | 最大版本 |
| **Geneformer-V2-104M_CLcancer** | 104M | 104M正常 + ~14M 癌细胞 | 4096 | 20,275 | 768 | 12 | 12 | 癌症持续学习版 |

### 关键差异

```
V1 vs V2 核心变化：

1. 数据规模: ~30M → ~104M 细胞
2. 输入长度: 2048 → 4096 (可以容纳更多基因)
3. 词表: 25,426(含非编码RNA) → 20,275(仅蛋白编码基因)
4. Special token: 无CLS/EOS → 有CLS/EOS
5. 模型大小: 可选 104M 或 316M 参数
```

> **💡 实际调用**：V2 模型通过 `AutoModelForMaskedLM.from_pretrained('./Geneformer-V2-104M')` 即可加载。V1 同样兼容。

---

## 4. 核心创新：Rank Value Encoding

这是 Geneformer **最核心的贡献**，理解它就能理解整个模型的设计哲学。

### 4.1 为什么要用排序而不是原始表达量？

**问题**：scRNA-seq 数据有严重的**技术批次效应**（batch effects），不同实验、不同平台测得的绝对表达量不可比。

**解决方案**：用**相对排序**代替**绝对数值**。

### 4.2 两步归一化

```
原始数据: 某细胞中每个基因的 UMI count
                │
    Step 1: 按总 counts 归一化 (CPM: counts per 10K)
                │
    Step 2: 除以基因的中位表达量 (跨 Genecorpus 的 non-zero median)
                │
    Step 3: 按归一化后的值降序排列，得到基因的排序列表
                │
    Step 4: 截断到前 N 个基因 (V2: 4096)，添加 CLS/EOS
                │
    Output: [CLS, g_42, g_7, g_103, ..., g_201, EOS]
              ↑    ↑                        ↑     ↑
            特殊 排名第1的  排名第2的        排名最后的  特殊
```

### 4.3 这种编码的妙处

| 特性 | 说明 |
|------|------|
| **非参数化** | 不需要学习任何参数，直接将表达数据转为排序 |
| **鲁棒性** | 对批次效应、技术噪声不敏感（排序比绝对值稳定） |
| **抑制管家基因** | 管家基因普遍高表达 → 跨细胞中位值高 → 归一化后被压低排名 |
| **突出关键基因** | 转录因子等低表达但在特定细胞中关键的基因 → 排名上升 |
| **稀疏性** | 只保留最重要的前 N 个基因 (V2: 4096)，忽略大量噪声 |

### 4.4 代码对应

```python
# tokenizer.py 中的核心逻辑
def tokenize_cell(gene_vector, gene_tokens):
    """Convert normalized gene expression to rank value encoding."""
    nonzero_mask = np.nonzero(gene_vector)[0]           # 只保留检测到的基因
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])

def rank_genes(gene_vector, gene_tokens):
    """Rank gene expression vector by value descending."""
    sorted_indices = np.argsort(-gene_vector)            # 降序排列
    return gene_tokens[sorted_indices]                   # 返回排序后的 token
```

---

## 5. 数据流 Pipeline

```
原始 scRNA-seq 数据 (.loom / .h5ad / .zarr)
        │
        ▼
┌─────────────────────────────┐
│   TranscriptomeTokenizer     │  tokenizer.py
│   - 加载基因字典 & token字典  │
│   - 合并重复 Ensembl ID       │
│   - Rank Value Encoding       │
│   - 添加 CLS/EOS token        │
│   - 裁剪到模型输入长度         │
└─────────────┬───────────────┘
              │
              ▼
    Tokenized Dataset (.dataset)
    HuggingFace Dataset 格式
    {"input_ids": [token_ids], "length": N, ...}
              │
              ▼
┌─────────────────────────────┐
│   GeneformerPretrainer       │  pretrainer.py
│   - 使用 DataCollatorForMLM  │
│   - 随机掩码 15% 的 token     │
│   - 预测被掩码的基因           │
│   - LengthGroupedSampler      │
└─────────────────────────────┘
```

---

## 6. 预训练 (Pretraining)

### 6.1 预训练目标

标准的 **Masked Language Modeling (MLM)**：

- **15%** 的基因 token 被随机掩码
- 模型根据上下文预测被掩码位置应为什么基因
- 损失函数：Cross-entropy loss

### 6.2 预训练过程

```python
# 伪代码：预训练流程
precollator = GeneformerPreCollator(token_dictionary=token_dict)
# 不直接使用 HuggingFace 的 tokenizer
# 而是用自定义的 GeneformerPreCollator 作为 MLM collator

data_collator = DataCollatorForLanguageModeling(
    tokenizer=precollator,
    mlm=True,
    mlm_probability=0.15
)

trainer = GeneformerPretrainer(
    model=model,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    args=training_args
)

trainer.train()
```

### 6.3 关键设计：LengthGroupedSampler

```python
# 按序列长度分组采样，提高训练效率
# 同一 batch 内的序列长度相近 → 减少 padding 浪费
sampler = LengthGroupedSampler(
    dataset=self.train_dataset,
    batch_size=batch_size,
    lengths=example_lengths  # 预计算的长度向量
)
```

---

## 7. 下游任务 (Downstream Tasks)

### 7.1 Fine-tuning 任务

| 任务类型 | 说明 | 代码模块 |
|---------|------|---------|
| **细胞分类 (Cell Classification)** | 对细胞状态/疾病状态分类 | `classifier.py` |
| **基因分类 (Gene Classification)** | 根据细胞中的基因集分类 | `classifier.py` |
| **多任务分类 (MTL)** | 同时预测多个细胞属性 | `mtl_classifier.py` |

### 7.2 Zero-shot 任务

| 任务类型 | 说明 | 代码模块 |
|---------|------|---------|
| **In Silico Perturbation** | 删除/过表达某个基因，观察嵌入变化 | `in_silico_perturber.py` |
| **嵌入提取** | 提取细胞或基因的嵌入向量 | `emb_extractor.py` |
| **批次整合** | 零样本批次校正 | 论文结果 |

### 7.3 典型 Fine-tuning 流程

```python
from geneformer import Classifier

cc = Classifier(
    classifier="cell",                          # 细胞分类
    cell_state_dict={"state_key": "disease",    # 标签列名
                     "states": "all"},           # 使用所有类别
    filter_data={"cell_type": ["Cardiomyocyte"]},# 过滤特定细胞类型
    training_args=training_args,
    freeze_layers=2,                            # 冻结前2层，只微调上层
    num_crossval_splits=5                       # 5折交叉验证
)

# 准备数据
cc.prepare_data(
    input_data_file="path/to/tokenized_data.dataset",
    output_directory="output/",
    output_prefix="experiment_name"
)

# 训练和验证
metrics = cc.validate(
    model_directory="path/to/pretrained_model",
    prepared_input_data_file="output/experiment_name_labeled.dataset",
    id_class_dict_file="output/experiment_name_id_class_dict.pkl",
    output_directory="output/",
    output_prefix="experiment_name"
)
```

---

## 8. 代码结构速览

```
Geneformer/
├── geneformer/                    # 核心代码包
│   ├── tokenizer.py               # 转录组 → Rank Value Encoding
│   ├── pretrainer.py              # 预训练训练器 + 数据预处理器
│   ├── classifier.py              # 细胞/基因分类器（fine-tuning）
│   ├── classifier_utils.py        # 分类器工具函数
│   ├── collator_for_classification.py  # 分类数据批处理
│   ├── emb_extractor.py           # 嵌入提取器 + 可视化
│   ├── in_silico_perturber.py     # 原位扰动分析
│   ├── in_silico_perturber_stats.py # 扰动分析统计
│   ├── perturber_utils.py         # 扰动工具函数
│   ├── evaluation_utils.py        # 评估工具
│   ├── mtl_classifier.py          # 多任务分类器
│   ├── mtl/                       # 多任务学习子模块
│   │   ├── model.py
│   │   ├── data.py
│   │   ├── train.py
│   │   └── collators.py
│   └── *.pkl                      # 基因字典/token字典等
│
├── Geneformer-V1-10M/             # V1 模型 (10M params)
├── Geneformer-V2-104M/            # V2 模型 (104M params)
├── Geneformer-V2-316M/            # V2 模型 (316M params)
├── Geneformer-V2-104M_CLcancer/   # 癌症持续学习版本
├── examples/                      # Jupyter Notebook 示例
│   ├── cell_classification.ipynb
│   ├── gene_classification.ipynb
│   ├── tokenizing_scRNAseq_data.ipynb
│   ├── in_silico_perturbation.ipynb
│   └── ...
└── fine_tuned_models/             # 预微调模型
```

---

## 9. 快速上手示例

### 9.1 加载模型

```python
from transformers import AutoModelForMaskedLM

# 加载 V2-104M
model = AutoModelForMaskedLM.from_pretrained('./Geneformer-V2-104M')

# 加载 V2-316M (默认模型)
model = AutoModelForMaskedLM.from_pretrained('./Geneformer-V2-316M')

# 加载 V1
model = AutoModelForMaskedLM.from_pretrained('./Geneformer-V1-10M')

# 加载癌症版
model = AutoModelForMaskedLM.from_pretrained('./Geneformer-V2-104M_CLcancer')
```

### 9.2 Tokenize 数据

```python
from geneformer import TranscriptomeTokenizer

# 初始化
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "cell_type": "cell_type",
        "disease": "disease"
    },
    nproc=4,
    model_version="V2"  # 自动设置 input_size=4096, special_token=True
)

# Tokenize
tk.tokenize_data(
    data_directory="path/to/scRNAseq_data/",
    output_directory="path/to/tokenized/",
    output_prefix="my_experiment",
    file_format="h5ad"
)
```

### 9.3 提取嵌入

```python
from geneformer import EmbExtractor

emb_ext = EmbExtractor(
    model_type="Pretrained",
    emb_mode="cell",       # 提取细胞嵌入
    cell_emb_style="mean_pool",  # 平均池化
    max_ncells=1000,
    forward_batch_size=200,
    nproc=4
)

embeddings = emb_ext.extract_embs(
    model=model,
    filtered_input_data=tokenized_dataset,
)
```

---

## 10. 关键概念 Q&A

### Q1: Geneformer 和标准 BERT 有什么区别？

| 方面 | 标准 BERT | Geneformer |
|------|-----------|------------|
| 输入 | 自然语言文本 (subword tokens) | 基因排序编码 (gene tokens) |
| 词表 | ~30K WordPiece tokens | ~20K 基因 IDs |
| 激活函数 | GELU | **ReLU** |
| 位置编码 | 绝对位置 (句子中的词序) | 绝对位置 (基因的表达排名) |
| 预训练数据 | Wikipedia + BooksCorpus | ~104M 单细胞转录组 |
| 特殊 token | [CLS], [SEP], [MASK] | [CLS], [EOS], [MASK], [PAD] |

### Q2: "基础模型"在生物学语境下是什么意思？

- **基础模型 (foundation model)** 指在大规模、多样的数据上预训练的模型，可以通过微调适应多个下游任务。
- 与传统的单细胞分析方法（如基于 PCA + 聚类的流程）不同，Geneformer 学习的是**基因调控的"语法"** —— 在特定细胞状态下哪些基因应该高表达、它们的共表达模式等。

### Q3: 模型的输入"max_position_embeddings=4096"是什么？

- 每个细胞最多保留 **4096 个最关键的基因**（经过 rank encoding 后排名前 4096 的基因）。
- V1 是 2048。这决定了每个输入序列的长度。
- 如果细胞检测到的基因数 > 4096，取前 4096 个；如果 < 4096，则 padding。

### Q4: 为什么 V2 的 vocab_size 从 25,426 减到 20,275？

- V1 包含蛋白编码基因 + 非编码 RNA（如 lncRNA、miRNA 等）
- V2 专注于**蛋白编码基因** (~20K)，因为蛋白编码基因的调控关系更明确，更容易解释
- 这也有助于减少计算开销

### Q5: 什么是 CLcancer 版本？

- **Continual Learning (持续学习)**：先在 104M 正常细胞上预训练，再在 ~14M 癌细胞数据上继续训练
- 目的是让模型适应**癌细胞特有的基因调控网络**（如拷贝数变异、突变导致的网络重连）
- 用于癌症相关的下游任务效果更好

---

## 11. 延伸阅读

### 核心论文

1. **C V Theodoris et al.** *Transfer learning enables predictions in network biology.* Nature, 2023. [阅读](https://rdcu.be/ddrx0)
   - Geneformer 原始论文。介绍了模型架构、预训练策略、零样本扰动和微调应用。

2. **H Chen et al.** *Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology.* Nature Computational Science, 2026. [阅读](https://rdcu.be/famFk)
   - V2 版本。扩展模型规模、引入量化、持续学习策略。

3. **C V Theodoris et al.** *Continual learning foundation model for cell state perturbation.* bioRxiv, 2024. [阅读](https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1)
   - 持续学习和多任务学习策略的详细介绍。

### 相关资源

- [Geneformer 官方文档](https://geneformer.readthedocs.io)
- [HuggingFace 模型页面](https://huggingface.co/ctheodoris/Geneformer)
- [Genecorpus-30M 数据集](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)
- [Genecorpus-104M 数据集](https://huggingface.co/datasets/theodoris-lab/Genecorpus-104M)

### 更广阔的虚拟细胞学习路径

```
1. ✅ Geneformer         ← 你现在在这里
2. scGPT                ← 同样的单细胞基础模型，但架构细节不同
3. scFoundation        ← 基于 xTrimo 架构，更大的模型
4. UCE                 ← 通用细胞嵌入，跨物种
5. NicheFormer / Novae ← 加入空间信息
6. GraphST / SpaceFlow ← 空间转录组分析方法
7. OpenBioMed          ← 多模态生物医学模型
```

---

> 💡 **学习建议**：先动手跑一下 `examples/` 目录里的 Notebook，特别是 `tokenizing_scRNAseq_data.ipynb` 和 `cell_classification.ipynb`，在实践中加深对 Rank Value Encoding 和 Fine-tuning 流程的理解。
