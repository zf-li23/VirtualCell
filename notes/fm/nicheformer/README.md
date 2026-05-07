# NicheFormer 学习笔记

> NicheFormer 是一个基于 **BERT 架构的空间转录组基础模型**，通过整合单细胞与空间转录组数据，学习细胞在其微环境（niche）中的上下文表示，支持多模态数据集成。

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
| **论文** | [NicheFormer: A Foundation Model for Single-Cell and Spatial Omics](https://www.nature.com/articles/s41592-025-02814-z), Nature 2025 |
| **架构** | BERT — Transformer Encoder |
| **预训练任务** | Masked Language Modeling (MLM) — 掩码基因预测 |
| **输入** | 基因 token 序列，融合空间上下文（niche 信息） |
| **输出** | 预测被掩码位置的基因 |
| **特殊 token** | 辅助 token 编码 modality/assay/species 信息 |
| **细胞嵌入** | CLS token 或 Mean Pooling |

### 核心思想

> "将每个细胞的**基因表达信息**与其**空间微环境（niche）**结合，通过 BERT 风格的 MLM 预训练，让模型同时学习：'这个细胞表达了什么基因'和'这个细胞周围是什么细胞'—— 也就是**单细胞 + 空间上下文**的联合表示。"

---

## 2. 模型架构

### 2.1 整体架构

```
Input: [CLS] g_1, g_2, ..., g_N [SEP]
         │
    ┌────┴────┐
    │ Embedding│  ← nn.Embedding(n_tokens + 5, d_model, padding_idx=1)
    └────┬────┘         特殊 token: [PAD]=1, [MASK]=2, [CLS]=3, [SEP]=4, [UNK]=5
         │
    ┌────┴────┐
    │ Position│  ← 可学习位置编码 (nn.Embedding) 或 正弦位置编码
    │ Encoder │
    └────┬────┘
         │
    ┌────┴────────────┐
    │ Transformer × N  │  ← nn.TransformerEncoder
    │ Encoder Layers    │     nn.TransformerEncoderLayer
    └────┬────────────┘
         │
    ┌────┴────┐
    │  Class  │
    │  Head   │  ← Linear(n_tokens) 用于 MLM 预测
    └────┬────┘
         │
    [MASK] 位置 → 预测基因类型
```

### 2.2 组件详解

#### Embedding 层

```python
self.embedding = nn.Embedding(
    num_embeddings=n_tokens + 5,  # 词表 + 5个特殊token
    embedding_dim=d_model,
    padding_idx=1                  # [PAD]=1
)
```

#### Positional Encoding

支持两种位置编码方式：

| 方式 | 说明 |
|------|------|
| **可学习 (learnable)** | `nn.Embedding(max_len, d_model)` — 每个位置一个可学习向量 |
| **正弦 (sinusoidal)** | 标准的正余弦位置编码，固定且不可学习 |

```python
# 可学习位置编码
self.pos_encoder = nn.Embedding(max_len, d_model)

# 或正弦位置编码
self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
```

#### Transformer Encoder

```python
self.encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=n_heads,
    dim_feedforward=d_model * 4,
    dropout=dropout,
    activation='gelu',
    batch_first=True
)
self.transformer = nn.TransformerEncoder(
    encoder_layer, num_layers=n_layers
)
```

#### 预测头

```python
self.classifier_head = nn.Linear(d_model, n_tokens)  # → 预测基因
self.pooler_head = nn.Linear(d_model, d_model)       # → 池化 (可选)
self.cls_head = nn.Linear(d_model, 164)               # → 分类头 (预训练辅助)
```

### 2.3 前向传播

```python
def forward(self, token_ids, attention_mask=None):
    # 1. Token 嵌入
    x = self.embedding(token_ids)
    
    # 2. 位置编码
    positions = torch.arange(token_ids.size(1), device=token_ids.device)
    x = x + self.pos_encoder(positions)
    
    # 3. Transformer
    x = self.transformer(x, src_key_padding_mask=~attention_mask)
    
    # 4. MLM 预测
    logits = self.classifier_head(x)
    
    return logits, x
```

---

## 3. 核心创新

### 3.1 空间微环境 (Niche) 信息融合

NicheFormer 与传统单细胞模型最大的不同是引入了**空间上下文**：

```
传统单细胞模型:
    细胞 A: [g_1, g_2, ..., g_N]  ← 只看自己的基因表达

NicheFormer:
    细胞 A: [g_1, g_2, ..., g_N, 
             细胞B的基因, 细胞C的基因, ...]  ← 还看邻居细胞
    
    ┌─────────┐
    │ 细胞 C  │
    └────┬────┘
         │ 邻居
    ┌────┴────┐  ┌─────────┐
    │ 细胞 B  │──│ 细胞 A  │  ← 中心细胞
    └─────────┘  └─────────┘
```

**如何实现**：通过将空间近邻细胞的基因表达拼接或融合到输入序列中。

### 3.2 辅助 Token 编码

NicheFormer 支持多种辅助 token 来编码元数据：

```
完整的输入序列:
[CLS] [MODALITY] [ASSAY] [SPECIES] g_1, g_2, ..., g_N [SEP]

其中:
- [MODALITY]:  编码数据模态 (scRNA-seq / Spatial / MERFISH 等)
- [ASSAY]:     编码具体实验方法 (10X Visium / Slide-seq 等)
- [SPECIES]:   编码物种信息 (Human / Mouse 等)
```

这些 token 通过 `on_after_batch_transfer` 方法在在数据加载后动态添加。

### 3.3 灵活的位置编码

| 类型 | 优点 | 缺点 | NicheFormer 选择 |
|------|------|------|-----------------|
| **可学习** | 灵活，可适应数据特征 | 需要更多参数 | ✅ 默认选择 |
| **正弦编码** | 无需学习，外推性强 | 可能不够灵活 | 可选 |

### 3.4 兼容单细胞与空间转录组

NicheFormer 的一个关键设计是**同时支持两种数据类型**：

| 数据类型 | 输入 | 是否包含空间信息 |
|----------|------|----------------|
| **单细胞转录组** | 单个细胞的基因列表 | ❌ 无空间上下文 |
| **空间转录组** | 中心细胞 + 邻居细胞的基因 | ✅ 有空间上下文 |

这意味着同一个模型可以同时用于单细胞和空间数据分析，只需在输入中是否包含邻居信息。

---

## 4. 数据预处理

### 4.1 Pipeline

Nicheformer 不需要传统的基因选择/标准化流程，因为其输入直接使用**基因 token ID**（类似 NLP 的词汇表）：

```python
# 1. 将基因符号映射为 token ID
gene_to_token = load_gene_vocabulary()  # ~58K token 词表
token_ids = [gene_to_token[g] for g in adata.var_names]

# 2. 构建输入序列（包含特殊 token）
input_ids = [CLS] + token_ids + [SEP]

# 3. (可选) 添加空间微环境信息
if spatial_context:
    input_ids += niche_tokens  # 邻近细胞的 token
```

### 4.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_tokens | ~58K | 基因 + 特殊 token |
| d_model | 512/768 | 嵌入维度 |
| 特殊 token | [PAD]=1, [MASK]=2, [CLS]=3, [SEP]=4, [UNK]=5 | 5 个特殊 token |

---

## 5. Tokenization 与输入编码

Nicheformer 使用标准的 BERT tokenization：

### 4.1 基因 Tokenization

与 Geneformer 类似，NicheFormer 使用基因 ID 作为 token：

```
基因符号 → 基因 ID → 词表索引 → nn.Embedding 查找 → 嵌入向量
```

### 4.2 特殊 Token 定义

| Token | ID | 用途 |
|-------|-----|------|
| [PAD] | 1 | 填充 |
| [MASK] | 2 | 掩码 |
| [CLS] | 3 | 序列开始，用于聚合整体信息 |
| [SEP] | 4 | 序列结束/分隔 |
| [UNK] | 5 | 未知基因 |

### 4.3 掩码策略

NicheFormer 使用标准的 BERT 掩码策略，但增加了**完全掩码 (complete_masking)** 支持：

```python
def complete_masking(token_ids, mask_ratio=0.15):
    """
    80% → [MASK] token
    10% → 随机 token
    10% → 保持不变
    
    注意：[PAD], [CLS] 等特殊 token 不会被掩码
    """
    mask = torch.rand(token_ids.shape) < mask_ratio
    mask &= (token_ids != 1)  # 不掩码 [PAD]
    mask &= (token_ids != 3)  # 不掩码 [CLS]
    
    # 80% → [MASK] (ID=2)
    # 10% → 随机
    # 10% → 不变
    ...
```

### 4.4 批次构建

```python
# 训练时的数据处理
def training_step(batch):
    # 1. 完全掩码
    masked_ids, labels = complete_masking(batch["input_ids"])
    
    # 2. 添加辅助 token
    batch = on_after_batch_transfer(batch)
    
    # 3. 前向传播
    logits, _ = self.forward(masked_ids, batch["attention_mask"])
    
    # 4. 只计算掩码位置的损失
    loss = self._prepare_target_indices(logits, labels)
    
    return loss
```

---

## 6. 预训练

### 5.1 预训练任务：MLM

与 Geneformer 完全相同，使用标准的 Masked Language Modeling：

```python
# 损失计算
loss_fct = CrossEntropyLoss(ignore_index=-100)  # 忽略非掩码位置

# 只计算被掩码位置的损失
def _prepare_target_indices(logits, labels):
    # labels 中 -100 表示不需要计算损失的位置
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss
```

### 5.2 训练策略

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 权重衰减 | 0.1 |
| 学习率调度 | Cosine Warmup |
| 损失函数 | CrossEntropy (仅掩码位置) |
| 掩码比例 | 15% |

### 5.3 训练数据

NicheFormer 在以下数据上预训练：

| 数据类型 | 数据源 | 说明 |
|----------|--------|------|
| **单细胞数据** | GEO / CellxGene / HCA | 标准 scRNA-seq |
| **空间数据** | 10X Visium / Slide-seq / MERFISH | 包含空间坐标 |
| **涵盖物种** | 人类、小鼠 | 主要物种 |
| **涵盖组织** | 多种组织类型 | 脑、肿瘤、发育组织等 |

---

## 7. 下游任务

### 6.1 Fine-tuning 架构

```python
# _fine_tune_model.py
class FineTuningModel(nn.Module):
    """
    微调包装器：NicheFormer backbone + 任务特定的 head
    """
    def __init__(self, backbone, n_outputs, task_type):
        self.backbone = backbone  # 冻结/微调的 NicheFormer
        self.head = nn.Linear(d_model, n_outputs)  # 简单线性分类头
    
    def forward(self, x):
        # 提取嵌入（支持 layer extraction/combination）
        emb = self.backbone.get_embeddings(x, 
            layer=12,                    # 从第12层提取
            combination="mean",          # 均值组合
            remove_context_tokens=True   # 移除辅助token
        )
        
        # 预测
        return self.head(emb)
```

### 6.2 支持的任务类型

| 任务类型 | 说明 | head 输出 |
|----------|------|-----------|
| **niche_classification** | 微环境分类（多类） | n_classes |
| **niche_binary_classification** | 微环境二分类 | 1 |
| **niche_multiclass_classification** | 多类微环境分类 | n_classes |
| **niche_regression** | 微环境属性回归 | 1 |
| **density_regression** | 细胞密度回归 | 1 |

### 6.3 Cell Embedding 提取

```python
# _nicheformer.py
def get_embeddings(self, token_ids, attention_mask=None, 
                   layer=12, combination="mean",
                   remove_context_tokens=True):
    """
    提取细胞嵌入
    
    参数：
    - layer: 从第几层提取 (-1 = 最后一层)
    - combination: "mean" / "sum" / "cls"
    - remove_context_tokens: 是否移除辅助 token 的嵌入
    """
    x = self.embedding(token_ids)
    x = x + self.pos_encoder(positions)
    
    # 逐层计算并返回指定层的输出
    for i, layer in enumerate(self.transformer.layers):
        x = layer(x)
        if i == target_layer:
            break
    
    if remove_context_tokens:
        x = x[:, 3:]  # 移除 [CLS], [MODALITY], [ASSAY] 等
    
    if combination == "mean":
        return x.mean(dim=1)  # 均值池化
    elif combination == "cls":
        return x[:, 0, :]     # CLS token
```

### 6.4 层提取策略

NicheFormer 支持从**特定层**提取嵌入，这对不同任务可能很重要：

```
早期层 (1-4):   基因级别的特征，保留较多局部信息
中间层 (5-8):   基因组合 / 通路级别的特征
后期层 (9-12):  细胞级别的整体表示
```

---

## 8. 代码结构速览

```
nicheformer/
├── src/
│   └── nicheformer/
│       └── models/
│           ├── _nicheformer.py              # 主模型 (347行)
│           │   ├── Nicheformer(pl.LightningModule)
│           │   ├── embedding, pos_encoder
│           │   ├── transformer, classifier_head
│           │   ├── forward, training_step
│           │   └── get_embeddings
│           │
│           ├── _nicheformer_fine_tune.py     # 微调模型 (405行)
│           │   └── NicheformerFineTune
│           │
│           ├── _fine_tune_model.py           # 微调工具类 (338行)
│           │   └── FineTuningModel
│           │
│           └── _utils.py                    # 工具函数 (63行)
│               └── complete_masking
│
├── notebooks/                               # 使用示例
├── docs/                                    # 文档
├── pyproject.toml                           # 项目配置
└── README.md
```

---

## 9. 关键概念 Q&A

### Q1: NicheFormer 和 Geneformer 的核心区别？

| 方面 | Geneformer | NicheFormer |
|------|-----------|-------------|
| **架构** | BERT | BERT |
| **数据范围** | 仅单细胞 | 单细胞 + 空间 |
| **空间信息** | 不支持 | 支持 (neighbor context) |
| **辅助 token** | 无 | Modality/Assay/Species |
| **位置编码** | 绝对位置 | 可学习 或 正弦 |
| **掩码策略** | 标准 MLM | 标准 MLM + complete_masking |

### Q2: 什么是"细胞微环境" (Niche)？

在生物学中：
- 细胞微环境指一个细胞周围的**物理和化学环境**
- 包括相邻细胞、细胞外基质、信号分子等
- 空间转录组学可以**同时测量**细胞身份和其在组织中的位置

在 NicheFormer 中：
- 空间近邻细胞的基因表达被编码为"上下文"
- 模型学习到："如果细胞 X 周围是 Y 细胞，那么 Z 基因可能高表达"

### Q3: 如何处理不同空间分辨率的数据？

- **低分辨率** (如 10X Visium, ~50μm/spot) → 每个 spot 可能包含多个细胞 → 使用 spot 级别的表达
- **高分辨率** (如 MERFISH, 单细胞级) → 精确的细胞-细胞邻接关系
- NicheFormer 通过**邻居选取半径**参数适应不同分辨率

### Q4: 辅助 token 具体有什么用？

- **Modality token**：告诉模型这是 scRNA-seq 还是空间数据，影响模型如何处理缺失的空间信息
- **Assay token**：不同实验技术有不同的偏差（如 10X vs Slide-seq 的捕获效率不同）
- **Species token**：不同物种的基因命名体系不同，帮助模型区分

### Q5: 单细胞数据和空间数据的训练方式有何不同？

```
单细胞数据:  [CLS] [MOD] [ASSAY] [SPEC] g_1, g_2, ..., g_N
             ↑ 没有邻居信息，模型只学习基因表达模式

空间数据:    [CLS] [MOD] [ASSAY] [SPEC] center_genes, neighbor_genes
             ↑ 包含邻居信息，模型学习空间上下文
```

---

## 10. 延伸阅读

### 核心论文

1. **LR Fleming et al.** *NicheFormer: A Foundation Model for Single-Cell and Spatial Omics.* Nature, 2025. [阅读](https://www.nature.com/articles/s41592-025-02814-z)

### 相关技术

- **BERT MLM**：[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **空间转录组学**：10X Visium, Slide-seq, MERFISH, STARmap
- **空间可变基因检测**：SpatialDE, Trendsceek, SPARK

### 学习路线

```
单细胞基础模型 (Geneformer → scGPT → scFoundation → UCE)
                    │
                    ▼
空间 + 单细胞 (NicheFormer)   ← 这一篇
                    │
                    ▼
Novae (纯空间图模型)
```

---

> 💡 **学习建议**：NicheFormer 是一个从单细胞到空间的**桥梁模型**。重点理解它如何通过辅助 token 和邻居上下文同时兼容两类数据。对比它与 Geneformer 的架构相似性和数据差异。
