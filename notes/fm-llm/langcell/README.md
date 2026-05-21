# LangCell 学习笔记

> LangCell 是首个将**单细胞转录组数据与自然语言描述进行联合预训练**的基础模型。它通过对比学习在细胞嵌入和文本嵌入之间建立统一语义空间，使得模型能够**零样本**理解细胞身份（如细胞类型、发育阶段、疾病状态等），无需任何标注数据即可完成细胞类型注释。

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
| **论文** | [LangCell: Language-Cell Pre-training for Cell Identity Understanding](https://arxiv.org/abs/2405.06708), ICML 2024 |
| **发布日期** | 2024-05 (arXiv v1), 2024-07 (ICML) |
| **出版** | ICML 2024 |
| **架构** | 双编码器 (Cell BERT + Text BERT) + 对比学习 + Cell-Text Matching |
| **预训练任务** | 对比学习 (跨模态对齐) + Cell-Text Matching (CTM) |
| **输入** | 单细胞转录组 (基因表达排序) + 细胞身份文本描述 |
| **输出** | 统一语义空间中的细胞嵌入 / 文本嵌入 |
| **词表** | Geneformer 基因词表 (~2 万基因) + BERT 文本词表 |
| **参数规模** | Cell Encoder ~6.5M (基于 Geneformer) + Text Encoder ~110M (BERT-base) ≈ **~120M** |
| **预训练数据** | **scLibrary**: 27.5M 细胞 (来自 CELLxGENE) + OBO Ontology 文本描述 |
| **代码** | [GitHub](https://github.com/PharMolix/LangCell) |
| **许可** | - |

### 核心思想

> **将细胞身份知识（细胞类型、发育阶段、疾病状态等）以自然语言形式引入预训练**，通过对比学习将细胞表达谱与语义描述对齐，使模型具备**零样本细胞身份理解**能力。这是首个将单细胞数据与自然语言进行跨模态联合预训练的工作。

---

## 2. 模型架构

### 2.1 整体架构图

```text
细胞输入: [CLS] g_1 g_2 ... g_N                文本输入: [CLS] t_1 t_2 ... t_M
         │                                             │
    ┌────┴────┐                                   ┌────┴────┐
    │Cell BERT│  ← Geneformer 架构                │Text BERT│  ← BERT-base
    └────┬────┘                                   └────┬────┘
         │                                             │
    ┌────┴────┐                                   ┌────┴────┐
    │cell_proj│  ← 线性映射                        │text_proj│  ← 线性映射
    └────┬────┘                                   └────┬────┘
         │                                             │
         └──────────┬─────────────────────┬────────────┘
                    │                     │
              对比学习损失            Cell-Text Matching
              (拉近配对、推远非配对)     (CTM Head, 二分类)
```

### 2.2 核心组件

#### Cell Encoder（细胞编码器）

基于 **Geneformer 架构**（BERT-style Transformer），将基因按表达量排序后作为 token 序列输入。使用了 Geneformer 的 tokenizer 和预训练权重作为初始化。

```python
# 继承自 Geneformer 的 BERT 架构，使用 HuggingFace BertModel
cell_bert = BertModel(config)  # standard Huggingface BertModel
# 使用 Geneformer 的 tokenizer 和预训练权重初始化
```

#### Text Encoder（文本编码器）

标准的 **BERT-base** 架构，编码细胞身份相关的自然语言描述（如 "CD4+ T cell in peripheral blood"）。

```python
text_bert = BertModel.from_pretrained("bert-base-uncased")
```

#### Projection Heads（投影头）

两个线性层，将 Cell Encoder 和 Text Encoder 的输出映射到共享的语义空间：

```python
cell_proj = nn.Linear(cell_dim, shared_dim)
text_proj = nn.Linear(text_dim, shared_dim)
```

#### CTM Head（Cell-Text Matching 头）

用于判断细胞-文本对是否匹配的二分类器，类似于 NLP 中的 Next Sentence Prediction：

```python
ctm_head = nn.Linear(shared_dim * 2, 2)
# 输入: [cell_emb; text_emb] 拼接 → 输出: match / not_match
```

---

## 3. 核心创新

### 3.1 跨模态联合预训练（核心创新）

将**单细胞转录组数据**与**自然语言描述**在预训练阶段进行联合学习。这在单细胞基础模型领域是首次尝试，相比同期模型（scGPT、Geneformer）只使用单一转录组模态，LangCell 通过引入文本模态获得了语义理解能力。

### 3.2 零样本细胞身份理解

得益于文本模态引入的语义知识，LangCell 是**首个能够在零样本场景下进行细胞类型注释**的单细胞基础模型。它可以直接用自然语言描述（如 "lung epithelial cell"）作为查询，在细胞嵌入空间中检索对应细胞。

### 3.3 scLibrary：大规模细胞-文本配对数据集

构建了包含 **2750 万细胞-文本配对**的数据集 scLibrary：
- 细胞数据来源：CELLxGENE
- 文本描述来源：OBO Foundry（Open Biological and Biomedical Ontology Foundry）
- 覆盖 8 个细胞身份维度：细胞类型、发育阶段、疾病信息、组织来源等

### 3.4 与同类模型的对比

| 维度 | LangCell | Geneformer | scGPT | scFoundation |
|------|----------|------------|-------|-------------|
| **模态** | 转录组 + 文本 | 仅转录组 | 仅转录组 | 仅转录组 |
| **架构** | 双编码器 BERT | 单编码器 BERT | GPT (Causal) | 非对称 AE |
| **预训练** | 对比 + CTM | MLM | MLM + 生成 | 重构 |
| **零样本注释** | ✅ 首创 | ❌ | ❌ (需微调) | ❌ |
| **参数量** | ~120M | 6.5M | ~50M | 3B |
| **预训练细胞数** | 27.5M | ~30M | 33M+ | 50M |

---

## 4. 数据预处理

### 4.1 输入格式

- **细胞端**: h5ad 格式的单细胞数据，与 Geneformer 兼容
- **文本端**: JSON 格式的细胞身份描述（从 OBO Ontology 中提取）

### 4.2 Pipeline

```python
# 1. 细胞数据预处理（类似 Geneformer）
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")
# 按表达量排序、过滤基因等
# → 输出 Geneformer 兼容的 tokenized 数据

# 2. 文本数据准备
# 从 OBO Foundry 中提取细胞身份描述
# 如: {"cell_type": "CD4+ T cell", "tissue": "blood", "disease": "normal"}
# → 输出格式化的文本描述

# 3. 构建配对数据
# 将细胞与对应的文本描述配对
# 正样本: 正确配对的细胞-文本
# 负样本: 随机配对的细胞-文本（用于 CTM 任务）
```

### 4.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_top_genes | ~2000 | 高变基因数量（继承 Geneformer） |
| max_seq_len | 2048 | 细胞最大序列长度 |
| text_max_len | 128 | 文本最大长度 |
| shared_dim | 256 | 共享嵌入空间维度 |
| batch_size | 64 | 训练批次大小 |

---

## 5. Tokenization 与输入编码

### 5.1 基因编码（细胞端）

使用 **Geneformer 的 tokenization 方式**：
- 基因按表达量**从高到低排序**
- 使用 Geneformer 的基因词表（Ensembl Gene ID → token ID）
- 特殊 token: `[CLS]` 置于序列开头用于提取细胞嵌入

### 5.2 文本编码

标准 BERT tokenization：
- 使用 BERT-base-uncased 的 tokenizer
- 将细胞身份描述转化为 [CLS] t_1 t_2 ... t_M [SEP] 格式
- 特殊 token: `[CLS]` 用于提取文本嵌入

### 5.3 表达值编码

与 Geneformer 一致——**不编码具体表达值**，只使用基因排序信息。基因按表达量排序后的位置本身就隐含了相对表达信息。

### 5.4 位置编码

Cell Encoder: 绝对位置编码（BERT 标准）
Text Encoder: 绝对位置编码（BERT 标准）

---

## 6. 预训练

### 6.1 预训练数据

| 数据来源 | 细胞数量 | 说明 |
|---------|---------|------|
| CELLxGENE (scLibrary) | **27.5M** | 公开单细胞数据 |
| OBO Foundry | 本体描述 | 8 个身份维度的语义标签 |

### 6.2 预训练目标

**双重损失函数**：

```python
# 1. 对比学习损失 (Contrastive Loss)
# 拉近配对细胞-文本的嵌入，推远非配对嵌入
contrastive_loss = -log( exp(sim(cell_i, text_i)/τ) / Σ_j exp(sim(cell_i, text_j)/τ) )

# 2. Cell-Text Matching 损失 (CTM Loss)
# 二分类：判断细胞-文本对是否匹配
ctm_loss = CrossEntropyLoss(ctm_head([cell_emb; text_emb]), label)
# label = 1 (匹配) / 0 (不匹配)

# 总损失
total_loss = contrastive_loss + λ * ctm_loss
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 2e-5 |
| Batch Size | 256 |
| 训练步数 | 500K |
| 优化器 | AdamW |
| 温度系数 τ | 0.07 |
| CTM 权重 λ | 0.5 |
| 共享空间维度 | 256 |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| **零样本细胞类型注释** | 用文本描述作为查询，检索最近邻细胞 | **首创**，显著优于随机基线 |
| **少样本细胞类型注释** | 少量标注数据 + 对比学习微调 | 超越 Geneformer、scGPT |
| **全微调细胞类型注释** | Fine-tune Cell Encoder | 与 SOTA 相当或更优 |
| **跨数据集迁移** | 零样本 / 少样本 | 具有良好泛化性 |

### 代码示例

```python
# 零样本细胞类型注释
# 1. 编码细胞
cell_emb = cell_encoder(cell_input)  # [n_cells, 768]
cell_emb = cell_proj(cell_emb)        # [n_cells, 256]

# 2. 编码目标细胞类型的文本描述
text_emb = text_encoder("lung epithelial cell")  # [1, 768]
text_emb = text_proj(text_emb)                   # [1, 256]

# 3. 计算相似度
similarities = cosine_similarity(cell_emb, text_emb)  # [n_cells]
predicted = cell_input[similarities.argmax()]
```

---

## 8. 代码结构速览

```
LangCell/
├── LangCell-annotation-zeroshot/   # 零样本注释
│   ├── zero-shot.ipynb              # 核心 notebook
│   └── utils.py                     # 模型定义 (Cell BERT + Text BERT)
├── LangCell-annotation-fewshot/    # 少样本注释
│   └── fewshot.py
├── LangCell-CE-annotation/         # 仅 Cell Encoder 微调
│   ├── finetune.py
│   └── fewshot.py
├── data_preprocess/                # 数据预处理
│   ├── preprocess.py
│   └── utils.py
├── geneformer_001/                 # 内置 Geneformer 依赖
└── README.md
```

### 快速开始

```bash
git clone https://github.com/PharMolix/LangCell
cd LangCell
pip install -r requirements.txt

# 零样本注释
# 下载预训练 checkpoint 后运行:
jupyter notebook LangCell-annotation-zeroshot/zero-shot.ipynb
```

---

## 9. 关键概念 Q&A

**Q: LangCell 与 Geneformer 的关系？**
A: LangCell 的 Cell Encoder 直接基于 Geneformer 架构，使用了 Geneformer 的 tokenizer 和预处理流程。可以说 LangCell = Geneformer（细胞编码器）+ BERT（文本编码器）+ 对比学习。

**Q: LangCell 为什么能零样本注释？**
A: 因为预训练时将细胞表达谱与语义描述对齐了。给定一个文本查询（如 "CD8+ T cell"），模型可以在细胞嵌入空间中直接找到语义最接近的细胞，无需任何训练数据。

**Q: LangCell 与 scGPT/Cell2Sentence 等 LLM 路线有何区别？**
A: LangCell 使用双编码器架构 + 对比学习，将细胞和文本映射到共享空间；而 Cell2Sentence 试图用 LLM 直接生成/理解细胞数据。前者更高效、可解释性更强，后者更灵活但计算成本更高。

**Q: scLibrary 数据集是否可以公开获取？**
A: 是的，scLibrary 已上传至 HuggingFace Datasets: https://huggingface.co/datasets/Toycat/scLibrary

---

## 10. 延伸阅读

- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) — LangCell Cell Encoder 的基础
- [CLIP](https://arxiv.org/abs/2103.00020) — 跨模态对比学习的经典范式，LangCell 的核心思路借鉴于此
- [Cell2Sentence](https://arxiv.org/abs/2309.11.557287) — 另一种细胞+语言的路，用 LLM 生成细胞数据
- [scChat](https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1) — 基于 LLM 的 scRNA-seq 对话助手
- [OpenBioMed](https://github.com/PharMolix/OpenBioMed) — LangCell 作者组的统一工具包（LangCell 后续将集成于此）


---

## 9. 关键概念 Q&A

### Q1: 这个模型与 X 模型的核心区别是什么？

**A**: ...

### Q2: 这个模型有什么已知的局限性？

**A**: ...

### Q3: 这个模型适合什么场景？

**A**: ...

---

## 10. 延伸阅读

- **[相关论文1]**：说明
- **[相关论文2]**：说明
- **[官方文档]**：链接

---

> *笔记最后更新：YYYY-MM-DD*
