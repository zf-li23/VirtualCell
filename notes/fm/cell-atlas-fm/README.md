# UCE 学习笔记

> UCE (Universal Cell Embeddings) 是一个**跨物种通用细胞嵌入模型**，通过在数百万跨物种细胞数据上进行对比学习预训练，学习物种无关的细胞表示，实现零样本的跨物种细胞类型识别。

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
| **论文** | [Universal Cell Embeddings: A Foundation Model for Cell Biology](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1), bioRxiv 2024<br>[A Cell Atlas Foundation Model for Scalable Search of Similar Human Cells](https://www.nature.com/articles/s41586-024-08411-y), Nature 2024 |
| **架构** | Transformer Encoder + Binary Decoder |
| **预训练任务** | 对比学习 (Contrastive Learning) + 掩码预测 |
| **输入** | 基因表达序列 (seq_len_first 格式) |
| **输出** | 细胞嵌入 (CLS token) + 二进制解码 |
| **词表** | 跨物种基因词汇表 |
| **模型维度** | d_model=1280, layers=6, heads=8 |

### 核心思想

> "学习一个**跨物种通用**的细胞嵌入空间，使得在人类数据上训练的细胞类型分类器可以直接应用到小鼠、斑马鱼等物种上 —— 让细胞嵌入成为生物学的'通用语言'。"

---

## 2. 模型架构

### 2.1 整体架构

```
Input: [CLS] g_1, g_2, ..., g_N
                │
    ┌───────────┴───────────┐
    │  Gene Embedding Layer  │  ← 基因 token → 嵌入 (GELU + LayerNorm)
    │  token_dim → d_model   │    d_model=1280
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │  Transformer Encoder   │  ← 6层, 8头注意力
    │  (nn.TransformerEncoder)│    d_model=1280, nhead=8
    └───────────┬───────────┘
                │
         ┌──────┴──────┐
         │             │
    ┌────┴────┐   ┌───┴───┐
    │  CLS    │   │Binary │
    │  Embed  │   │Decoder│
    │ (d=1280)│   │4层MLP │
    └─────────┘   └───────┘
                    │
              Binary Prediction
              (掩码位置的是/否)
```

### 2.2 Gene Embedding Layer

```python
# model.py
self.gene_embedding = nn.Sequential(
    nn.Linear(token_dim, d_model),  # token_dim → 1280
    nn.GELU(),
    nn.LayerNorm(d_model)
)
```

- 将离散基因 token 映射为连续嵌入
- 与标准的 nn.Embedding 不同，这里使用 Linear 层
- 支持更大的输入维度 (token_dim 可能 > d_model)

### 2.3 Transformer Encoder

```python
# model.py
self.encoder_layer = nn.TransformerEncoderLayer(
    d_model=1280,
    nhead=8,
    dim_feedforward=5120,  # 4x d_model
    dropout=0.1,
    activation='gelu',
    batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layer, num_layers=6
)
```

- 标准 PyTorch Transformer Encoder
- 6 层，8 头注意力
- FFN 维度 5120 (4 × d_model)
- GELU 激活函数

### 2.4 Binary Decoder

```python
# model.py
self.binary_decoder = nn.Sequential(
    nn.Linear(d_model + 128, 2048),  # d_model + extra_features → 2048
    nn.GELU(),
    nn.Linear(2048, 512),
    nn.GELU(),
    nn.Linear(512, 128),
    nn.GELU(),
    nn.Linear(128, 1)  # → 二分类 logit
)
```

- 4 层 MLP (1280+128 → 2048 → 512 → 128 → 1)
- 输入拼接了 d_model (基因上下文的嵌入) 和 128 维的额外特征
- 输出是一个标量：预测该位置是否被掩码/该基因的表达状态

### 2.5 Seq-Len-First 格式

UCE 使用 **seq_len_first** 数据格式，这是与大多数模型不同的设计：

```
通常的格式 (batch_first):
    [batch_size, seq_len, d_model]

UCE 的格式 (seq_len_first):
    [seq_len, batch_size, d_model]
    
# 适应 PyTorch TransformerEncoder 的默认格式
# 但实际上 UCE 使用 batch_first=True
```

---

## 3. 核心创新

### 3.1 跨物种对比学习

UCE 的预训练核心是**物种无关的对比学习**：

```
训练数据：人类、小鼠、斑马鱼、果蝇等多个物种的单细胞数据
        │
        ▼
    同一细胞类型（如 T 细胞）的不同物种样本 → 正样本对
    不同细胞类型的样本                      → 负样本对
        │
        ▼
    对比损失：让同类型不同物种的细胞嵌入靠近
             让不同类型细胞的嵌入远离
        │
        ▼
    结果：细胞嵌入空间是物种无关的 (species-agnostic)
          "T cell" 无论来自人类还是小鼠 → 嵌入空间中的相近位置
```

### 3.2 CLS Token 细胞嵌入

- 使用 BERT 风格的 [CLS] token 来聚合整个序列的信息
- 与 Geneformer 类似，但 UCE 更强调 CLS 嵌入在对比学习中的作用
- CLS 嵌入的维度：1280

### 3.3 Binary Decoder 设计

UCE 的解码器设计不同于其他模型：

| 模型 | 解码器任务 | 输出类型 |
|------|-----------|---------|
| Geneformer | 预测被掩码的基因类别 | 分类 (vocab_size 类) |
| scGPT | 预测基因+表达值对 | 分类 + 回归 |
| **UCE** | 预测是否被掩码 (binary) | **二分类** |

**Binary Decoder 的作用**：
- 作为辅助任务，帮助编码器学习更好的表示
- 简单但有效，避免过度复杂的解码器
- 与对比学习损失联合优化

### 3.4 大规模跨物种预训练数据

| 属性 | 描述 |
|------|------|
| **物种数量** | 数十个物种 |
| **涵盖范围** | 人类、小鼠、斑马鱼、果蝇、线虫等 |
| **细胞数量** | 数百万个细胞 |
| **组织类型** | 多样 (免疫系统、神经系统、上皮组织等) |
| **数据来源** | GEO, CellxGene, HCA 等 |

---

## 4. 数据预处理

### 4.1 预处理流程

```
原始 scRNA-seq 数据
        │
        ▼
┌─────────────────────────────┐
│ 1. 质量控制                 │ 过滤低质量细胞/基因
│    - min_genes              │
│    - max_genes              │
│    - max_pct_mito           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. 标准化                   │ 总计数归一化 → log1p
│    - normalize_total         │
│    - log1p                  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. 基因选择                 │ 选择常见基因
│    - 跨物种基因映射          │ 统一不同物种的基因名
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. 格式转换                 │
│    - seq_len_first          │
│    - 添加 CLS token         │
└─────────────┬───────────────┘
              │
              ▼
      模型输入序列
```

### 4.2 跨物种基因映射

UCE 的关键预处理步骤之一是**跨物种基因名称映射**：

```
人类基因:  ENSG000001... (CD3E)
    │
    ▼ 同源基因映射 (Homology Mapping)
    │
小鼠基因:  ENSMUSG000... (Cd3e)
    │
    ▼
统一为:  gene_CD3E (跨物种通用 ID)
```

- 使用同源基因映射表 (如 NCBI HomoloGene / Ensembl Compara)
- 将不同物种的直系同源基因映射到同一 ID
- 使得模型可以"理解"跨物种的基因对应关系

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

UCE 使用 `nn.Linear + GELU + LayerNorm` 作为基因嵌入层，而非标准的 `nn.Embedding`：

```python
self.gene_embedding = nn.Sequential(
    nn.Linear(token_dim, d_model),
    nn.GELU(),
    nn.LayerNorm(d_model)
)
```

### 5.2 表达值编码

UCE 将基因表达值作为连续特征，经过 MLP 映射后与基因嵌入融合。

### 5.3 特殊 Token

- `[CLS]`：用于聚合细胞级表示（输出为 1280 维细胞嵌入）
- 其他特殊 token：标准 BERT 风格

---

## 6. 预训练

### 5.1 预训练损失函数

UCE 的预训练损失由两部分组成：

```python
loss = λ₁ · L_contrastive + λ₂ · L_binary

# L_contrastive: 对比学习损失 (NT-Xent / InfoNCE)
#   让同一细胞类型的不同物种样本靠近
#   让不同细胞类型的样本远离

# L_binary: 二进制解码损失 (BCE)
#   预测基因是否被掩码
```

### 5.2 对比学习策略

```
Batch 中的样本对:

┌─────────┬─────────┬─────────┐
│ H_Tcell │ M_Tcell │ Z_Tcell │  ← 正样本对 (同类型, 不同物种)
├─────────┼─────────┼─────────┤
│ H_Neuron│ M_Neuron│ Z_Neuron│
├─────────┼─────────┼─────────┤
│ H_Tcell │ H_Neuron│ M_Tcell │  ← 负样本对 (不同类型)
└─────────┴─────────┴─────────┘

H = Human, M = Mouse, Z = Zebrafish
```

### 5.3 训练策略

| 超参数 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 学习率 | 1e-4 ~ 3e-4 |
| Batch size | 256 ~ 1024 |
| 预热步数 | 10,000 |
| 梯度裁剪 | 1.0 |

---

## 7. 下游任务

### 6.1 零样本跨物种细胞分类

UCE 最有特色的能力是**零样本跨物种应用**：

```python
# 在人类数据上训练细胞类型分类器
classifier.fit(human_embeddings, human_cell_types)

# 直接在小鼠数据上预测 (!)
mouse_predictions = classifier.predict(mouse_embeddings)
# → 不需要重新训练，不需要微调
```

### 6.2 支持的任务

| 任务 | 说明 | 是否零样本 |
|------|------|-----------|
| **跨物种细胞注释** | 从带注释的物种迁移到未注释物种 | ✅ 零样本 |
| **细胞状态识别** | 识别细粒度的细胞亚型 | 需要微调 |
| **基因扰动效应** | 预测基因敲除后的细胞状态变化 | 需要微调 |
| **发育轨迹分析** | 追踪细胞分化的连续过程 | ✅ 零样本 |
| **疾病状态比较** | 比较健康 vs 疾病细胞 | ✅ 零样本 |

### 6.3 评估方式

```python
# eval_single_anndata.py 和 evaluate.py 中的评估流程
# 1. 编码细胞为嵌入
embeddings = model.encode_cells(adata)

# 2. 在嵌入空间中进行
#    - kNN 分类 (跨物种迁移)
#    - 聚类分析
#    - UMAP 可视化

# 3. 评估指标
#    - Accuracy (如果已知细胞类型标签)
#    - NMI (归一化互信息)
#    - ARI (调整兰德指数)
```

---

## 8. 代码结构速览

```
UCE/
├── model.py                     # Transformer 模型定义 (115行)
│   ├── UCE (nn.Module)
│   │   ├── gene_embedding       # 基因嵌入层
│   │   ├── transformer_encoder  # 6层 Transformer
│   │   └── binary_decoder       # 4层 MLP 解码器
│
├── utils.py                     # 工具函数
│
├── data_proc/                   # 数据预处理脚本
│   └── process_data.py          # 跨物种数据准备
│
├── eval_data.py                 # 训练数据加载
├── eval_single_anndata.py       # 单文件评估
├── evaluate.py                  # 主评估脚本
│
├── examples/                    # 使用示例
└── model_files/                 # 预训练模型权重
```

---

## 9. 关键概念 Q&A

### Q1: UCE 的"通用"体现在哪里？

**跨物种通用性**是 UCE 最核心的卖点：
- 在几十个物种的数据上预训练
- 学习物种无关的细胞嵌入
- 在一个物种上训练的分类器可以直接应用到其他物种
- 解决了生物医学研究中模式动物（小鼠、斑马鱼等）数据注释困难的问题

### Q2: UCE 与其他单细胞基础模型对比？

| 维度 | Geneformer | scGPT | scFoundation | UCE |
|------|-----------|-------|-------------|-----|
| **物种范围** | 人类为主 | 人类为主 | 人类为主 | **多物种** |
| **跨物种能力** | 有限 | 有限 | 有限 | **原生支持** |
| **预训练方式** | MLM | 生成式 | 重建 | **对比学习** |
| **嵌入维度** | 256~1152 | 768 | 1280 | **1280** |
| **解码器** | 预测基因类别 | 预测基因-值对 | 重建表达值 | **二分类** |

### Q3: CLS token 在对比学习中起什么作用？

- CLS token 的最终隐藏状态作为细胞嵌入
- 在对比学习中，CLS 嵌入被直接用于计算相似度
- 与 Geneformer 不同，UCE 的 CLS 嵌入是专门为对比学习优化的

### Q4: Binary Decoder 为什么只需要预测"是否被掩码"？

- 这是一个**辅助任务**，目的是帮助编码器学习更好的表示
- 不需要精确预测被掩码的是什么基因或表达值
- 简单任务可以避免解码器过强，防止信息从解码器"泄漏"

### Q5: 如何处理不同物种的词表差异？

- 通过**同源基因映射**统一基因 ID
- 对于物种特异性基因（无同源物），使用特殊 token 或丢弃
- 模型只使用跨物种共有的基因进行训练
- 这保证了模型可以处理任意物种的输入

---

## 10. 延伸阅读

### 相关技术

- **对比学习**：[SimCLR](https://arxiv.org/abs/2002.05709), [MoCo](https://arxiv.org/abs/1911.05722)
- **跨物种基因映射**：NCBI HomoloGene, Ensembl Compara
- **NT-Xent loss**：[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

### 学习路线

```
单细胞基础模型:
    Geneformer (人类, BERT, MLM)
        ↓
    scGPT (人类, GPT, 生成式)
        ↓
    scFoundation (人类, Performer, 自编码器)
        ↓
    UCE (跨物种, 对比学习)   ← 这一篇
        ↓
    加入空间信息:
    NicheFormer / Novae
```

---

> 💡 **学习建议**：UCE 的跨物种能力是其最大亮点。建议先理解对比学习的基本原理（正负样本对的构建），再关注它如何通过同源基因映射实现物种无关的细胞嵌入。与前面三个模型对比，UCE 的架构最简单但设计最巧妙。
