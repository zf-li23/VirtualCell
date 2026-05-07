# scFoundation 学习笔记

> scFoundation 是一个基于 **xTrimoGene 非对称自编码器架构**的大规模单细胞基础模型（~100M-1B 参数），采用 **Performer 线性复杂度注意力**和**可微分软分箱编码**，在百万级单细胞数据上进行预训练。

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
| **论文** | [xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data](https://arxiv.org/abs/2311.15156), arXiv 2023 |
| **架构** | 非对称自编码器 (Asymmetric Autoencoder) — 大编码器 + 小解码器 |
| **注意力机制** | Performer (线性复杂度，通过正交随机投影矩阵近似 Softmax 注意力) |
| **预训练任务** | 基因表达掩码重建 (Masked Expression Reconstruction) |
| **输入** | 基因 token + 表达值 (通过 AutoDiscretizationEmbedding2 软分箱) |
| **输出** | 重建被掩码位置的基因表达值 |
| **词表** | 19,264 个基因 (基于 10X Genomics 常用基因) |
| **参数规模** | ~100M ~ 1B |

### 核心思想

> "用一个**非对称的自编码器** —— 编码器非常深（~24层 Transformer），解码器非常浅（~2层）—— 强制编码器学习到**紧凑且有意义的细胞表示**，同时用 Performer 注意力将计算复杂度从 O(n²) 降低到 O(n)。"

---

## 2. 模型架构

### 2.1 非对称自编码器设计

```
Input: 基因-表达值序列 [g_1, g_2, ..., g_N]
                │
    ┌───────────┴───────────┐
    │  Expression Embedding  │  ← AutoDiscretizationEmbedding2
    │  + Gene2Vec Position   │  ← 基因位置编码
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │   Deep Encoder (24L)   │  ← Performer Encoder
    │   - ~100M~1B params    │    各层交替使用
    │   - Global + Local Attn│    Global: Performer (FA)
    │   - Rotary Position    │    Local: 标准 Attention
    └───────────┬───────────┘
                │  Latent Representation
    ┌───────────┴───────────┐
    │   Shallow Decoder (2L) │  ← Performer Decoder
    │   - 轻量级重建网络     │
    └───────────┬───────────┘
                │
Output: 被掩码位置的表达值重建
```

### 2.2 Performer 注意力 (核心组件)

scFoundation 使用 Performer 的 **FAVOR+** 机制来实现线性复杂度的注意力：

```
传统 Attention (O(n²)):
    Attn(Q,K,V) = softmax(QK^T / √d) · V

Performer Attention (O(n)):
    Q' = orthogonal_proj(Q)   ← 随机正交投影
    K' = orthogonal_proj(K)
    Attn ≈ Q' · (K'^T · V)   ← 通过矩阵乘法结合律降低复杂度
```

```python
# performer.py 中的核心实现
class FastAttention(nn.Module):
    def __init__(self, dim, nb_features=256, kernel="softmax"):
        # nb_features: 随机投影的维度 (特征数)
        # kernel: softmax_kernel / generalized_kernel
        self.orthogonal_matrix = self._generate_orthogonal(dim, nb_features)
    
    def forward(self, queries, keys, values):
        # 1. 随机正交投影
        Q_prime = queries @ self.orthogonal_matrix
        K_prime = keys @ self.orthogonal_matrix
        
        # 2. 核变换 (确保非负性)
        Q_prime = self.kernel_fn(Q_prime)
        K_prime = self.kernel_fn(K_prime)
        
        # 3. 线性复杂度计算: Q' · (K'^T · V)
        attn = Q_prime @ (K_prime.transpose(-2, -1) @ values)
        return attn
```

**关键优势**：
- 计算复杂度从 O(L²d) 降至 O(Ld·nb_features)
- 支持更长的序列（更多基因）
- 正交投影保证低近似误差

### 2.3 混合注意力机制

```python
# 编码器各层交替使用两种注意力
class SelfAttention(nn.Module):
    def __init__(self, dim, attention_type="global"):
        if attention_type == "global":
            # Performer Fast Attention (线性复杂度)
            self.attn = FastAttention(dim, nb_features=256)
        else:
            # 标准 Attention (二次复杂度)
            self.attn = StandardAttention(dim)
    
    def forward(self, x, mask=None):
        # Local 注意力在局部窗口内使用标准注意力
        # Global 注意力在整个序列上使用 Performer
        return self.attn(x, mask)
```

### 2.4 Rotary Position Embedding (RoPE)

scFoundation 使用旋转位置编码来编码基因的位置信息：

```python
# 旋转位置编码
# 在 attention 的 Q 和 K 中注入位置信息
# 通过旋转矩阵实现，支持相对位置编码
```

### 2.5 Gene2Vec 位置嵌入

```python
# 基因级别的预训练位置嵌入
class Gene2VecPositionalEmbeddingIdx(nn.Module):
    """
    基于 Word2Vec 思路的基因位置嵌入
    
    不是用绝对位置索引（如 1,2,3,...,N），
    而是对每个基因 ID 学习一个位置嵌入，
    表示该基因在细胞中的"功能位置"
    """
    def __init__(self, vocab_size, d_model):
        self.embeddings = nn.Embedding(vocab_size, d_model)
```

---

## 3. 核心创新

### 3.1 AutoDiscretizationEmbedding2 (可微分软分箱)

这是 scFoundation **最独特**的编码方式。与 Geneformer 的排序编码和 scGPT 的 binning/continuous 编码都不同：

```
传统硬分箱 (Hard Binning):
    表达值 5.3 → bin_5 (嵌入向量)   ← 硬边界，不可微

AutoDiscretizationEmbedding2 (软分箱):
              表达值 5.3
                  │
         ┌────────┴────────┐
         │  MLP + LeakyReLU │  ← 学习从表达值到 bin 权重的映射
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │     Softmax     │  ← 软分配概率
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │  加权求和 bin 嵌入│  ← 可微分
         └─────────────────┘
```

```python
# mae_autobin.py
class AutoDiscretizationEmbedding2(nn.Module):
    """
    可微分自动分箱嵌入
    
    输入: 原始表达值 (连续)
    输出: 嵌入向量 (通过软分箱 + 加权求和)
    """
    def __init__(self, num_bins, d_model):
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_bins)
        )
        self.bin_embeddings = nn.Parameter(torch.randn(num_bins, d_model))
        # alpha: 控制软/硬分配的混合系数
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # x: [batch, seq_len] 原始表达值
        logits = self.mlp(x.unsqueeze(-1))  # → [batch, seq_len, num_bins]
        
        # 软分配 (Softmax with temperature)
        weights = F.softmax(logits * self.alpha, dim=-1)
        
        # 加权求和 bin 嵌入
        emb = weights @ self.bin_embeddings  # → [batch, seq_len, d_model]
        return emb
```

**优势**：
- **端到端可微分**：整个分箱过程可以反向传播
- **自适应**：模型自己学习如何分组表达值，而非人工设定 bin 边界
- **软分配**：一个表达值可以分配到多个 bin（加权），避免硬边界的截断效应

### 3.2 非对称自编码器设计哲学

```
编码器 (24层 Transformer):
    - 学习高维、丰富、冗余的细胞表示
    - 捕捉基因之间的复杂调控关系
    - 参数量大，计算资源集中

瓶颈层 (Latent):
    - 紧凑的细胞嵌入向量
    - 编码了细胞状态的完整信息

解码器 (2层 Transformer):
    - 轻量级重建
    - 只需要从 latent 中解码出被掩码的基因
    - 避免解码器"记住"输入（瓶颈设计）
```

**设计动机**：迫使模型学到"有意义的表示"而非简单的"记忆输入"，类似于自编码器的降维思想。

### 3.3 Performer + 混合注意力

scFoundation 的注意力机制结合了两种方案：

| 注意力类型 | 复杂度 | 应用位置 | 特点 |
|-----------|--------|---------|------|
| **Performer (Global)** | O(n) | 奇数层 | 长程依赖，全局上下文 |
| **Standard (Local)** | O(n·w) | 偶数层 | 局部细化，精确匹配 |

这种设计在**全局感知**和**局部精度**之间取得了平衡。

---

## 4. 数据预处理

### 4.1 预处理流程

```
原始 scRNA-seq 数据
        │
        ▼
┌─────────────────────────┐
│ 过滤低质量细胞/基因      │
└───────────┬─────────────┘
            │
┌─────────────────────────┐
│ 标准化 (Total Counts →  │
│ log(CPM+1))             │
└───────────┬─────────────┘
            │
┌─────────────────────────┐
│ 选择 19,264 个基因      │
│ (基于 OS_scRNA_gene_    │
│  index.19264.tsv)       │
└───────────┬─────────────┘
            │
┌─────────────────────────┐
│ 可选: 是否使用掩码       │
│ 一定比例的基因表达值     │
└───────────┬─────────────┘
            │
    GeneExpressionSequence
```

### 4.2 基因索引

scFoundation 使用预定义的 19,264 个基因索引文件：

```
OS_scRNA_gene_index.19264.tsv

格式：每行一个基因，按基因符号排序
用途：确保所有样本使用统一的基因顺序，便于批量处理
```

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

每个基因通过 Gene2Vec 位置嵌入编码，将基因 ID 映射为 d_model 维向量。

### 5.2 表达值编码：AutoDiscretizationEmbedding2（核心创新）

scFoundation 使用**可微分软分箱**将连续表达值映射为离散嵌入：

```python
# 概念伪代码
class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, n_bins=128, d_model=256):
        self.bin_centers = nn.Parameter(torch.randn(n_bins))
        self.bin_embeddings = nn.Parameter(torch.randn(n_bins, d_model))
    
    def forward(self, values):
        # 软分配：将表达值 soft-assign 到最近的 bin
        weights = F.softmax(-(values - self.bin_centers)**2 / τ, dim=-1)
        # 加权求和 bin 嵌入
        value_repr = weights @ self.bin_embeddings
        return value_repr  # 离散化后的连续嵌入
```

### 5.3 四种编码策略对比

| 编码方式 | 描述 | 使用模型 |
|---------|------|---------|
| Rank Encoding | 基因按表达量排序的位置 | Geneformer |
| Pair Tokenization | 基因+表达值配对 token | scGPT |
| **AutoDiscretizationEmbedding2** | 可微分软分箱 | **scFoundation** |
| Continuous | 连续值直接输入 | scBERT |

### 5.4 位置编码

使用 Rotary Position Embedding (RoPE) 编码基因在序列中的顺序。

---

## 6. 预训练

### 6.1 预训练任务

**Masked Expression Reconstruction (MER)**：

- 随机掩码 15% 的基因-表达值对
- 模型需要重建被掩码位置的**原始表达值**
- 损失函数：MSE (均方误差) + 可能结合对比学习损失

### 6.2 与 MLM 的区别

| 方面 | BERT-MLM (Geneformer) | MER (scFoundation) |
|------|----------------------|-------------------|
| 预测目标 | 被掩码的**基因类别** | 被掩码的**表达值** |
| 输出 | 分类 (vocab_size 类) | 回归 (连续值) |
| 信息粒度 | 哪些基因在该位置 | 该基因表达量是多少 |
| 损失函数 | CrossEntropy | MSE |

### 6.3 训练数据

| 属性 | 描述 |
|------|------|
| 数据规模 | 数百万单细胞 (hPancreas, hPBMC, hBrain 等) |
| 数据来源 | GEO, ArrayExpress 等公共数据库 |
| 涵盖组织 | 胰腺、血液、脑、肺、心脏等 |
| 物种 | 人类为主 |

---

## 7. 下游任务

### 7.1 支持的任务

| 任务 | 类型 | 说明 |
|------|------|------|
| **细胞类型注释** | 分类 | 基于嵌入的细胞类型预测 |
| **基因表达预测** | 回归 | 预测基因扰动后的表达 |
| **药物响应预测** | 分类/回归 | 预测药物处理效果 |
| **扰动效应预测** | 回归 | 基因敲除/过表达的效应预测 |
| **批次整合** | 表示学习 | 从嵌入中去除批次效应 |
| **基因模块发现** | 无监督 | 识别共表达基因模块 |

### 6.2 Embedding 提取

```python
# 从编码器中间层提取细胞嵌入
model.encode(cell_data)  # 返回 latent representation

# 可用于：
# 1. 细胞聚类
# 2. 可视化 (UMAP/t-SNE)
# 3. 作为下游分类器的输入特征
```

---

## 8. 代码结构速览

```
scFoundation/
├── model/
│   ├── pretrainmodels/
│   │   ├── mae_autobin.py          # MaeAutobin 主模型 (自编码器)
│   │   ├── performer.py            # Performer 注意力实现 (638行)
│   │   └── ...                     # 其他模型变体
│   └── ...
│
├── preprocessing/                  # 数据预处理工具
│   └── preprocess.py              # 标准化、过滤、基因选择
│
├── annotation/                     # 细胞类型注释模块
│
├── enhancement/                    # 数据增强模块
│
├── mapping/                        # 参考映射模块
│
├── genemodule/                     # 基因模块发现工具
│
├── GEARS/                          # 扰动预测 (GEARS 相关工作)
│
├── SCAD/                           # 单细胞异常检测
│
├── ablation/                       # 消融实验脚本
│
├── DeepCDR/                        # 药物响应预测
│
├── OS_scRNA_gene_index.19264.tsv   # 19264 个基因的索引文件
│
└── apiexample/                     # API 使用示例
```

---

## 9. 关键概念 Q&A

### Q1: scFoundation 和常见 Transformer 模型的主要区别？

| 方面 | 标准 Transformer | scFoundation |
|------|-----------------|-------------|
| **注意力** | Softmax (O(n²)) | Performer (O(n)) |
| **架构** | 编码器=解码器 | 编码器 >> 解码器 |
| **位置编码** | 正余弦 / 可学习 | Rotary + Gene2Vec |
| **输入编码** | 离散 token | 软分箱 (可微分) |
| **预训练** | 预测 token | 重建表达值 |

### Q2: Performer 相比标准 Attention 有什么优缺点？

**优点**：
- 线性复杂度，支持更长的基因序列
- 内存占用显著降低
- 通过正交投影保持近似精度

**缺点**：
- 需要选择合适的核函数和投影维度
- 对某些任务可能不如标准注意力精确
- 实现相对复杂

### Q3: 什么是 Gene2Vec 位置嵌入？

- 类似于 Word2Vec 为每个词学习一个嵌入
- scFoundation 为每个**基因 ID** 学习一个固定嵌入
- 表示该基因在所有细胞中的"平均上下文"
- 与 Transformer 的位置编码**互补**使用

### Q4: 非对称自编码器的好处是什么？

**信息瓶颈效应**：迫使信息通过一个狭窄的瓶颈（latent representation）
- 防止模型"记忆"输入（过拟合的一种形式）
- 强制学习压缩的、有意义的表示
- 类似去噪自编码器的设计哲学

### Q5: 和 scGPT 的 binning 有何不同？

| 方面 | scGPT Category Binning | scFoundation AutoDiscretization |
|------|-----------------------|-------------------------------|
| **边界** | 硬边界 (分位数/等距) | 软边界 (可学习) |
| **可微性** | 不可微 (argmax) | 完全可微 |
| **分配方式** | 一个值 → 一个 bin | 一个值 → 多个 bin 的加权和 |
| **学习方式** | 固定 | 端到端学习 |

---

## 10. 延伸阅读

### 核心论文

1. **J Gong et al.** *xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data.* arXiv, 2023. [阅读](https://arxiv.org/abs/2311.15156)
   - scFoundation 原始论文。介绍非对称自编码器设计和 Performer 应用。

### 相关技术

- **Performer**：[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794), ICLR 2021
- **Rotary Position Embedding**：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **自编码器**：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

### 对比学习路线

```
Geneformer (BERT + Rank Encoding)
    │
    ▼
scGPT (GPT + Gene-Value Pairs)
    │
    ▼
scFoundation (Performer + Asymmetric AE + Soft Binning)  ← 这一篇
    │
    ▼
UCE (Cross-Species + Contrastive Learning)
```

---

> 💡 **学习建议**：重点关注 scFoundation 三个独特创新：(1) Performer 线性注意力，(2) AutoDiscretizationEmbedding2 可微分分箱，(3) 非对称自编码器设计。这些设计思路在效率和表示质量之间取得了独特的平衡。
