# scGPT 学习笔记

> scGPT 是一个基于 **GPT（Causal Transformer）架构**的单细胞基础模型，通过自回归生成式预训练学习单细胞转录组数据，并支持多组学（mRNA + ATAC + 蛋白）分析。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理 Pipeline](#4-数据预处理-pipeline)
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
| **论文** | [scGPT: Toward Building a Foundation Model for Single-Cell Multi-Omics Using Generative AI](https://www.nature.com/articles/s41592-024-02201-0), Nature Methods 2024 |
| **架构** | GPT (Causal Transformer — 单向注意力) |
| **预训练任务** | 基因表达生成式预训练 (生成式 MLM / 自回归) |
| **输入** | 基因 token + 表达值 token 的配对序列 |
| **输出** | 预测下一个基因-表达值对 |
| **词表** | ~30K+ 基因 (基于 HGNC 基因符号) |
| **许可** | MIT |

### 核心思想

> "将每个细胞的转录组建模为 **基因-表达值对的有序序列**，通过 GPT 的因果注意力机制学习基因表达的生成模式，使模型理解：**在给定已观测基因的前提下，下一个基因应该是什么、表达量应该多高**。"

---

## 2. 模型架构

scGPT 使用 **GPT (Causal Transformer)** 架构，关键组件如下：

### 2.1 整体架构图

```
Input: [CLS] (gene_A, val_12) (gene_B, val_5) ... (gene_N, val_1)
                    │
         ┌──────────┴──────────┐
         │    GeneEncoder      │  ← 基因 ID → 基因嵌入 (nn.Embedding)
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   ValueEncoder      │  ← 表达值 → 值嵌入 (支持多种编码策略)
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   BatchEncoder      │  ← 批次 ID → 批次嵌入 (用于批次校正)
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   DSBN (可选)       │  ← Domain-Specific BatchNorm
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │  Transformer × N    │  ← Causal Attention (GPT Decoder)
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   Decoders          │  ← 基因预测头 + 表达值预测头
         └──────────────────┘
```

### 2.2 核心组件详解

#### GeneEncoder (基因编码器)

```python
self.gene_encoder = nn.Embedding(vocab_size, d_model, padding_idx=0)
```

- 将每个基因 ID 映射为 d_model 维嵌入向量
- 支持多组学的 mod_type 编码 (CATEGORY 类型)

#### ValueEncoder (表达值编码器)

支持 **三种编码策略**，由 `input_emb_style` 参数控制：

| 策略 | 说明 | 实现方式 |
|------|------|---------|
| **continuous** (连续值) | 表达值作为连续标量，直接加到基因嵌入上 | `value_encoder(x).squeeze(-1) * gene_emb` |
| **category** (离散化) | 将表达值离散化为若干个 bin，每个 bin 对应一个嵌入 | `value_encoder(x.long())` — bin 类别的 Embedding |
| **scaling** (缩放) | 表达值作为缩放因子，乘以基因嵌入 | 乘法交互 |

#### BatchEncoder (批次编码器)

```python
self.batch_encoder = nn.Embedding(n_batch, d_model)
```

- 将批次 ID 映射为嵌入向量，加到每个 token 上
- 用于保留批次信息，辅助批次校正任务

#### Domain-Specific BatchNorm (DSBN)

```python
# 每个 domain (批次/数据集) 有自己的 BatchNorm 参数
self.dsbn = nn.ModuleList([
    nn.BatchNorm1d(d_model, eps=eps, affine=False)  # 每个 domain 一个
])
# 可学习的 scale 和 shift
self.dsbn_scale = nn.Parameter(torch.ones(n_domains, d_model))
self.dsbn_shift = nn.Parameter(torch.zeros(n_domains, d_model))
```

- DSBN 允许每个数据域有自己的归一化统计量
- 用于**显式建模批次效应**，在归一化层中分离不同数据源的特征分布

### 2.3 输入嵌入组合方式

```python
# continuous 模式：加法组合
if input_emb_style == "continuous":
    x = self.gene_encoder(genes)  # [batch, seq_len, d_model]
    # 表达值连续编码 → 缩放基因嵌入
    value_emb = self.value_encoder(values)  # [batch, seq_len, 1]
    x = x * value_emb
elif input_emb_style == "category":
    x = self.gene_encoder(genes) + self.value_encoder(values.long())
elif input_emb_style == "scaling":
    x = self.gene_encoder(genes)
    value_emb = self.value_encoder(values).squeeze(-1)
    x = x * value_emb

# 添加批次编码
if self.batch_encoder is not None:
    x = x + self.batch_encoder(batch_tags)
```

### 2.4 Cell Embedding (细胞嵌入)

scGPT 支持三种细胞嵌入提取策略：

| 策略 | 说明 |
|------|------|
| **CLS** | 取序列第一个 token ([CLS]) 的输出作为细胞嵌入 |
| **avg-pool** | 对所有 token 的输出做平均池化 |
| **w-pool** | 加权池化 (可学习权重) |

---

## 3. 核心创新

### 3.1 基因-表达值配对 Tokenization

与 Geneformer 的 Rank Value Encoding（仅保留排序，丢弃数值）不同，scGPT **同时保留基因身份和表达值**：

```
scGPT:  [(gene_A, 12.5), (gene_B, 5.3), ..., (gene_N, 1.1)]
                      ↑           ↑                ↑
                   基因token + 值token (配对)

Geneformer:  [gene_A, gene_B, ..., gene_N] (仅排序信息)
```

**意义**：保留表达量数值信息，使模型能够学习"表达量有多高"这个关键维度。

### 3.2 三种表达值编码策略对比

| 策略 | 信息损失 | 连续性 | 适用场景 |
|------|---------|--------|---------|
| continuous | 几乎无 | 连续 | 通用场景，保留精细表达差异 |
| category (binning) | 有（离散化） | 离散 | 降低噪声影响，处理稀疏数据 |
| scaling | 几乎无 | 连续 | 强调基因间的相对表达比例 |

### 3.3 Domain-Specific BatchNorm (DSBN)

这是 scGPT 处理**批次效应**的关键创新：

- **传统方法**：通过 Harmony、Combat 等工具在预处理阶段消除批次效应
- **scGPT 方法**：在模型内部用 DSBN 显式建模批次差异
- **优势**：模型可以区分"批次导致的差异"和"生物学差异"，在下游任务中通过固定 domain 参数实现批次校正

### 3.4 多组学支持 (Multi-Omics)

scGPT 通过 `mod_type` 参数支持多种组学数据类型：

| 组学 | mod_type | 说明 |
|------|----------|------|
| mRNA 表达 | "scRNA" | 标准单细胞转录组 |
| ATAC-seq | "scATAC" | 染色质开放性 |
| 蛋白表达 | "scProtein" | 表面蛋白标记物 |
| 多组学联合 | — | 同时输入多个组学视图 |

### 3.5 Reference Mapping (参考映射)

scGPT 支持将新数据映射到参考数据集上：

```
新数据 ──▶ scGPT 编码 ──▶ 细胞嵌入
                            │
                    FAISS 向量检索
                            │
                    ┌───────┴───────┐
                参考数据集 ────▶ 映射结果
```

- 使用 **FAISS** (Facebook AI Similarity Search) 进行高效最近邻检索
- 可以快速将新细胞映射到已知的参考分类上

---

## 4. 数据预处理 Pipeline

scGPT 的预处理流程通过 `Preprocessor` 类实现：

```
原始 scRNA-seq 数据 (.h5ad)
        │
        ▼
┌─────────────────────────────┐
│ 1. filter_gene_by_counts    │ 过滤低表达基因
│    - min_count = 3          │ (至少在 3 个细胞中检测到)
│    - min_proportion = None  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. filter_cell_by_counts    │ 过滤低质量细胞
│    - min_count = 100        │ (至少检测到 100 个基因)
│    - min_genes = 100        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. normalize_total          │ 文库大小归一化
│    - target_sum = 1e4       │ 每个细胞总计数 → 10K
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. log1p                    │ 对数变换
│    - 处理偏态分布            │ log(x + 1)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 5. subset_hvg               │ 选择高变基因
│    - flavor = "seurat_v3"   │ 基于方差选择前 N 个
│    - n_top_genes = 2000~    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 6. binning (可选)           │ 分箱离散化
│    - n_bins = 50/100/200    │ 表达值 → bin 类别
└─────────────┬───────────────┘
              │
              ▼
    预处理完毕的 AnnData
```

### 关键代码

```python
# scgpt/preprocess.py
class Preprocessor:
    """scGPT 的数据预处理管道"""
    
    def __call__(self, adata: AnnData) -> AnnData:
        if not self.check_logged(adata):
            # 检查数据是否已 log 归一化
            adata = self.filter_gene_by_counts(adata)
            adata = self.filter_cell_by_counts(adata)
            adata = self.normalize_total(adata)
            adata = self.log1p(adata)
        else:
            print("Data already logged, skip normalization")
        
        # 选择高变基因
        adata = self.subset_hvg(adata, n_top_genes=2000)
        return adata
```

---

## 5. Tokenization 与输入编码

### 5.1 GeneVocab (基因词表)

```python
# scgpt/tokenizer/gene_tokenizer.py
class GeneVocab:
    """
    HuggingFace PreTrainedTokenizer 的子类
    
    功能：
    - 构建基因 ID → token ID 的映射
    - 支持添加特殊 token ([PAD], [CLS], [MASK], [SEP])
    - 支持多组学 mod_type token
    """
    vocab: Dict[str, int]  # 基因符号 → token ID
    pad_token: str
    cls_token: str
    mask_token: str
```

### 5.2 Tokenize 流程

```python
# 核心 tokenize 函数
def tokenize_and_pad_batch(
    genes: List[List[str]],      # 每个细胞的基因列表
    values: List[List[float]],   # 对应的表达值
    gene_vocab: GeneVocab,       # 基因词表
    max_len: int,                # 最大序列长度 (e.g., 2000)
    padding: bool = True,
    return_values: bool = True
):
    """
    1. 将基因符号 → 基因 token ID
    2. 在每个序列前添加 [CLS] token
    3. 将所有序列 padding 到相同长度
    4. 创建 attention mask
    """
    ...
```

### 5.3 掩码策略 (随机掩码)

```python
def random_mask_value(
    genes: Tensor,
    values: Tensor,
    mask_ratio: float = 0.15,       # 掩码比例
    mask_value: int = 2,            # [MASK] token ID
    pad_token_id: int = 0,          # [PAD] token ID
):
    """
    对表达值进行随机掩码：
    - 80% 概率 → 掩码为 mask_value
    - 10% 概率 → 随机替换为其他值
    - 10% 概率 → 保持不变
    """
    ...
```

---

## 6. 预训练

### 6.1 预训练目标

scGPT 采用**生成式预训练**，与 GPT 的自回归生成不同，scGPT 使用的是**生成式掩码建模**：

- 随机掩码 15% 的基因-表达值对
- 模型根据未被掩码的上下文，预测被掩码位置的**基因**和**表达值**
- 损失函数：基因预测 CrossEntropy + 表达值预测 MSE/CrossEntropy

### 6.2 预训练数据

| 数据集 | 细胞数量 | 说明 |
|--------|---------|------|
| scGPT 全数据 | ~33M 细胞 | 来自多个公共数据集 |
| 涵盖组织 | 多种 | 血液、脑、心脏、肺、胰腺等 |
| 涵盖物种 | 人类为主 | 少量小鼠数据 |

### 6.3 训练策略

- **优化器**：AdamW
- **学习率调度**：余弦退火 + 预热
- **Batch size**：动态调整
- **混合精度**：FP16 训练
- **梯度裁剪**：防止梯度爆炸

---

## 7. 下游任务

### 7.1 支持的任务类型

| 任务 | 说明 | 方法 |
|------|------|------|
| **细胞类型注释** | 预测细胞类型标签 | Fine-tuning / 零样本 |
| **基因表达预测** | 预测基因表达水平 | 生成式输出 |
| **扰动响应预测** | 预测基因扰动后的表达变化 | Fine-tuning |
| **批次校正** | 整合多个数据集的批次效应 | DSBN + Fine-tuning |
| **药物响应预测** | 预测药物处理后的细胞状态 | Fine-tuning |
| **多组学整合** | 整合 mRNA + ATAC + 蛋白数据 | Zero-shot / Fine-tuning |
| **参考映射** | 将新数据映射到参考数据集 | Zero-shot (FAISS) |

### 7.2 典型 Fine-tuning 流程

```python
import scgpt as scg

# 1. 加载预训练模型
model = scg.model.TransformerModel.from_pretrained("scgpt")

# 2. 准备数据 (预处理 + tokenize)
adata = sc.read_h5ad("data.h5ad")
scg.preprocess(adata)  # 预处理
tokenized = scg.tokenize(adata)  # Tokenize

# 3. 微调
trainer = scg.SingleCellTrainer(
    model=model,
    train_dataset=tokenized,
    task="cell_classification",
    label_key="cell_type"
)
trainer.train()

# 4. 预测
predictions = trainer.predict(adata_new)
```

---

## 8. 代码结构速览

```
scGPT/
├── scgpt/                        # 核心代码包
│   ├── __init__.py
│   ├── preprocess.py             # 数据预处理 (过滤器 + 归一化 + HVG)
│   ├── model/
│   │   ├── model.py              # TransformerModel 主架构
│   │   └── dsmil.py              # 多实例学习模型 (可选)
│   ├── tokenizer/
│   │   └── gene_tokenizer.py     # GeneVocab, tokenize 函数
│   ├── trainer.py                # 训练器 (Fine-tuning)
│   ├── data_collator.py          # 数据批处理
│   └── utils.py                  # 工具函数
│
├── examples/                     # 使用示例
│   ├── scgpt_fine_tune.ipynb
│   ├── scgpt_perturbation.ipynb
│   └── scgpt_reference_mapping.ipynb
│
├── scripts/                      # 训练脚本
├── tests/                        # 单元测试
└── tutorials/                    # 教程
```

---

## 9. 关键概念 Q&A

### Q1: scGPT 和 Geneformer 的核心区别是什么？

| 方面 | Geneformer | scGPT |
|------|-----------|-------|
| **架构** | BERT (Encoder-only) | GPT (Decoder-only) |
| **注意力** | 双向 (Bidirectional) | 单向 (Causal) |
| **输入** | 基因排序列表 (Rank Encoding) | 基因-表达值对 |
| **表达值处理** | 丢弃数值，只保留排序 | 保留数值，编码为嵌入 |
| **预训练任务** | MLM (掩码预测) | 生成式 (掩码 + 预测) |
| **批次效应处理** | 依赖数据预处理 | DSBN 显式建模 |
| **多组学** | 不支持 | 支持 mRNA + ATAC + 蛋白 |
| **零样本能力** | In Silico Perturbation | Reference Mapping |

### Q2: "Continuous" vs "Category" 编码的区别是什么？

- **Continuous**：表达值作为连续浮点数，通过一个线性层映射为标量权重，乘以基因嵌入。保留了精细的表达差异。
- **Category**：将表达值离散化为 N 个 bin (如 50 bins)，每个 bin 对应一个可学习的嵌入向量。有利于处理稀疏数据，但会损失精细信息。

### Q3: DSBN 具体怎么工作的？

假设有 3 个不同批次的数据：
- 每个批次有自己的 BatchNorm 的**均值和方差统计量**
- 模型有 3 组可学习的 scale 和 shift 参数
- 前向时，根据输入的 `batch_tag` 选择对应的归一化参数
- 在微调时，可以固定 DSBN 参数，只微调其他层，实现批次校正

### Q4: 什么是 Reference Mapping？

- Reference Mapping 是将新测序的细胞数据映射到已有的带注释的参考数据集上
- scGPT 提取细胞嵌入后，用 FAISS 进行最近邻搜索
- 属于**零样本**能力：不需要重新训练模型
- 适用于快速注释新数据

### Q5: scGPT 的输入长度和基因选择？

- 默认输入长度：max_len = 2000 (可配置)
- 每个细胞选择前 2000 个高变基因 (HVG)
- 使用 Seurat v3 方法选择 HVG
- 序列按 HVG 排序，而非按表达量排序（与 Geneformer 不同）

---

## 10. 延伸阅读

### 核心论文

1. **H He et al.** *scGPT: Toward Building a Foundation Model for Single-Cell Multi-Omics Using Generative AI.* Nature Methods, 2024. [阅读](https://www.nature.com/articles/s41592-024-02201-0)
   - scGPT 原始论文。详细介绍了模型架构、预训练策略、下游任务评估。

### 相关技术

- **GPT / Causal Attention**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) + 因果掩码
- **DSBN**：[Domain-Specific Batch Normalization](https://arxiv.org/abs/1806.03975)
- **FAISS**：[Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)

### 对比阅读

```
Geneformer (BERT + Rank Encoding)     ← 推荐先学
    │
    ▼
scGPT (GPT + Gene-Value Pairs)       ← 这一篇
    │
    ▼
scFoundation (Performer + Asymmetric AE)  ← 下一篇
```

---

> 💡 **学习建议**：对比 Geneformer 和 scGPT 的输入编码方式（Rank Encoding vs 基因-值配对），这是理解两种不同设计哲学的关键。然后可以看看 scGPT 的 Fine-tuning 示例 Notebook。
