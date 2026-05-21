# xTrimoGene 学习笔记

> xTrimoGene 提出了一种**非对称编码器-解码器 Transformer 架构**，灵感来自 MAE（Masked Autoencoder）。它利用单细胞转录组数据的**天然稀疏性**——细胞中 ~90% 的基因为零表达——仅用标准 Transformer FLOPs 的 **3.4%** 即可达到甚至超越全尺寸模型的性能。它是 scFoundation 的前身工作，也是首个系统展示单细胞领域 scaling law 的模型。

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
| **论文** | [xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data](https://openreview.net/forum?id=gdwcoBCMVi), NeurIPS 2023 |
| **发布日期** | 2023-09 (NeurIPS) |
| **出版** | NeurIPS 2023 |
| **架构** | **非对称编码器-解码器 Transformer**（类 MAE） |
| **预训练任务** | **回归掩码预测**（MSE 损失，掩码 ~30% 位置，恢复连续表达值） |
| **输入** | 基因 token ID + 自动离散化表达值嵌入 |
| **输出** | 基因级别的表达值预测 + 细胞嵌入（[CLS]） |
| **词表** | ~19,264 个基因（Ensembl ID） |
| **参数规模** | 3M / 10M / **100M**（xTrimoGene-100M） |
| **预训练数据** | 约 **5000 万细胞**的 scRNA-seq 数据，~500 亿有效基因 token |
| **代码** | - |
| **许可** | - |

### 核心思想

> **利用单细胞数据的天然稀疏性（~90% 零表达），让编码器只处理非零位置（~10% 长度），结合线性注意力解码器恢复全长。** 这使得模型可以用极低的计算成本处理完整的基因表达谱，同时通过掩码预测学习有意义的生物学表示。

---

## 2. 模型架构

### 2.1 整体架构图

```text
输入: [g_1, g_2, ..., g_N] 基因 token + 表达值嵌入
              │
    ┌─────────┴─────────┐
    │   随机掩码 ~30%    │
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  编码器 (Encoder)  │  ← 标准 Transformer
    │  只处理非零+未掩码 │  ← 约 10% 全长
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │  解码器 (Decoder)  │  ← Performer（线性注意力）
    │  处理完整长度+零值 │  ← 填充掩码位置
    └─────────┬─────────┘
              │
    ┌─────────┴─────────┐
    │   预测头 (Head)    │
    │   预测掩码位置的   │
    │   连续表达值       │
    └─────────┬─────────┘
              │
          Output: 恢复的表达值 + 细胞嵌入
```

### 2.2 核心组件

#### 非对称编码器（Asymmetric Encoder）

编码器是标准的 Transformer Encoder，但**只接收非零表达 + 未被掩码的基因位置**。由于 scRNA-seq 的稀疏性，输入长度仅为全长的 ~10%，计算量大幅降低。

```python
# 假设全长 19264 基因，非零约 2000 个
# 掩码 30% → 编码器实际处理约 1400 个 token
# 仅为标准 Transformer 的 7%
encoder_out = transformer_encoder(non_zero_unmasked_tokens)
```

#### Performer 解码器（Linear Attention Decoder）

解码器使用 **Performer（FAVOR+ 线性注意力机制）**，可以线性复杂度处理全长序列（~20K token）。它将编码器的输出插入对应位置，并在掩码/零值位置填入可学习的 [MASK] 标记。

```python
# 将编码器输出放回全长序列的正确位置
full_sequence = scatter(encoder_out, positions, sequence_length=19264)
# 用 Performer 处理全长
decoder_out = performer_decoder(full_sequence)  # O(N) 复杂度
```

#### 自动离散化模块（Auto-discretization）

将连续表达值转化为可学习的向量表示。不同于简单的硬分箱，它使用可微分的软分箱机制：

```python
# 自动离散化：连续值 → 软分箱 → 加权组合
bin_weights = softmax(linear(value))  # [batch, 100 bins]
value_embed = sum(bin_weights * bin_embeddings)  # 加权组合各分箱嵌入
```

---

## 3. 核心创新

### 3.1 非对称编码器-解码器（核心创新）

受 MAE（Masked Autoencoder）启发，但针对单细胞数据做了关键适配：
- **编码器只处理非零 + 未掩码位置**（约 7-10% 全长）
- **解码器用 Performer 线性注意力**处理完整长度
- 计算量仅为标准 Transformer 的 **3.4%**
- 比同等规模的 Performer 还快 **3 倍**

### 3.2 自动离散化（Auto-discretization）

将连续表达值映射到可学习嵌入的创新方法：
- 使用 **100 个可学习分箱嵌入**
- 通过线性层 + LeakyReLU + Softmax 计算软分箱权重
- 最终嵌入为各分箱嵌入的加权和
- **可微分**，随整个模型端到端训练
- 优于 Geneformer 的硬排序和 scBERT 的简单分箱

### 3.3 偏置掩码策略（Biased Masking）

不同于 MAE 的高比例掩码，xTrimoGene 采用：
- 掩码 **30%** 的位置（含零和非零）
- **同等概率掩码零和非零**位置
- 防止模型被零表达主导，强制学习有意义的基因关系

### 3.4 首次展示单细胞领域 Scaling Law

在 5M → 21M → 50M 细胞规模上训练，xTrimoGene 展示了：
- 随着数据量增大，下游任务性能持续提升
- 更大的模型（100M 参数）受益更多
- 为后来 scFoundation 的 3B 模型奠定了基础

### 3.5 与同类模型的对比

| 维度 | xTrimoGene | Geneformer | scGPT | scBERT |
|------|-----------|------------|-------|--------|
| **架构** | 非对称 AE (MAE) | BERT | GPT (Causal) | BERT |
| **Tokenization** | 自动离散化 | 排序 + rank | 连续值编码 | 分箱 |
| **计算效率** | **3.4% FLOPs** | 100% | 100% | 100% |
| **参数** | 100M | 6.5M | ~50M | ~110M |
| **预训练细胞** | ~50M | ~30M | 33M+ | ~1M |
| **独特能力** | 极致高效 | 最轻量 | 多模态 | 最早 |

---

## 4. 数据预处理

### 4.1 输入格式

- 来源：公开 scRNA-seq 数据（多种组织）
- 格式：count 矩阵 → 对数归一化（log1p）
- 基因选择：高变基因 ~19,264 个

### 4.2 Pipeline

```python
# 1. 加载数据
adata = sc.read_h5ad("data.h5ad")
# 2. 对数归一化
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
# 3. 选择高变基因
sc.pp.highly_variable_genes(adata, n_top_genes=19264)
# 4. 构建输入：基因 ID + 表达值
gene_ids = vocab(adata.var_names)  # Ensembl ID → token ID
values = adata.X  # log1p 归一化后的表达值
```

### 4.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_genes | 19,264 | 基因词表大小 |
| mask_ratio | 0.3 | 掩码比例 |
| n_bins | 100 | 自动离散化分箱数 |
| encoder_layers | 6~12 | 编码器层数 |
| decoder_layers | 4~8 | 解码器层数 |

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

基因使用 Ensembl Gene ID 作为 token，通过可学习的嵌入表（nn.Embedding）映射为向量。词表大小约为 19,264 个基因。

### 5.2 表达值编码

使用 **自动离散化（Auto-discretization）**：
- 连续表达值 → 线性层 → LeakyReLU → 线性层 → Softmax（得到 100 个分箱的权重）
- 最终表达嵌入 = Σ(权重 × 对应分箱嵌入向量)
- 整个过程可微分，随模型一起训练

### 5.3 特殊 Token

使用 [CLS] token 位于序列开头，其输出用作细胞嵌入。
使用 [MASK] token 填充被掩码的基因位置。

### 5.4 位置编码

使用标准的可学习绝对位置编码，长度为最大基因数（~19,264）。

---

## 6. 预训练

### 6.1 预训练数据

| 数据来源 | 细胞数量 | 有效 token 数 |
|---------|---------|--------------|
| 多种公开 scRNA-seq 数据 | ~50M | ~500 亿 |

### 6.2 预训练目标

```python
# 回归掩码预测：MSE 损失
loss = MSE(predicted_values[~mask], ground_truth_values[~mask])
# 只计算被掩码位置的损失
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-4 |
| Batch Size | 256 |
| 训练步数 | 500K |
| 优化器 | AdamW |
| 掩码比例 | 30% |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| **细胞类型注释** | 提取细胞嵌入 + 分类器 | Zheng68K / Segerstolpe 数据集 SOTA |
| **扰动响应预测** | xTrimoGene 嵌入 + GEARS | MSE 降低 **14.8%** |
| **协同药物组合预测** | xTrimoGene 嵌入 + DeepDDS | 显著提升 |

---

## 8. 代码结构速览

```
xTrimoGene/
├── model/
│   ├── encoder.py         # 非对称编码器
│   ├── decoder.py         # Performer 解码器
│   └── discretizer.py     # 自动离散化模块
├── pretrain/
│   ├── pretrain.py        # 预训练脚本
│   └── data.py            # 数据加载
├── downstream/
│   ├── classification/    # 细胞类型注释
│   ├── perturbation/      # 扰动预测
│   └── drug_synergy/      # 药物协同预测
├── config.py              # 配置
└── utils.py               # 工具函数
```

### 快速开始

```bash
# xTrimoGene 是 BioMap 内部工作
# 其后续模型 scFoundation 已开源
# 建议直接使用 scFoundation
```

---

## 9. 关键概念 Q&A

**Q: xTrimoGene 与 scFoundation 的关系？**
A: xTrimoGene 是 scFoundation 的前身工作。两者来自同一团队（BioMap Research），xTrimoGene 验证了非对称架构 + 自动离散化的有效性，scFoundation 将其扩展到 3B 参数。

**Q: 为什么编码器只处理非零位置？**
A: scRNA-seq 数据 ~90% 的基因表达为零。这些零值区域几乎没有信息量，跳过它们可以大幅降低计算成本而不损失性能。

**Q: Performer 解码器的作用？**
A: 编码器输出的稀疏表示需要被"展开"回全长序列（填充 MASK 和零值位置），Performer 用线性注意力 O(N) 处理全长，比标准注意力的 O(N²) 高效得多。

---

## 10. 延伸阅读

- [scFoundation](https://www.nature.com/articles/s41592-024-02305-7) — xTrimoGene 的后续工作，3B 参数
- [MAE (Masked Autoencoder)](https://arxiv.org/abs/2111.06377) — 非对称编码器-解码器的原始灵感
- [Performer](https://arxiv.org/abs/2009.14794) — 线性注意力机制，xTrimoGene 解码器的核心
- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) — 同期对比模型
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

## 2. 模型架构

### 2.1 整体架构图

```text
用 ASCII 艺术图展示整体流程，例如：

Input: [CLS] g_1(v_1) g_2(v_2) ... g_N(v_N)
         │
    ┌────┴────┐
    │Encoder  │  ← 组件名 + 说明
    └────┬────┘
         │
    ┌────┴────┐
    │ Head    │  ← 输出头
    └────┬────┘
         │
    Output
```

### 2.2 核心组件

#### [组件1 名称]

```python
# 伪代码或关键代码片段
# 说明每个组件的作用
```

#### [组件2 名称]

```python
# 伪代码
```

---

## 3. 核心创新

总结 2-3 个这个模型最重要的创新点：

### 3.1 [创新点1]

说明 + 代码/公式（可选）

### 3.2 [创新点2]

说明

### 3.3 [与同类模型的对比]

| 维度 | 本模型 | 模型A | 模型B |
|------|--------|-------|-------|
| 架构 | | | |
| 预训练数据 | | | |
| 核心能力 | | | |

---

## 4. 数据预处理

### 4.1 输入格式

### 4.2 Pipeline

```python
# 从原始数据到模型输入的完整流程
```

### 4.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_top_genes | 2000 | 高变基因数量 |
| max_seq_len | 2048 | 最大输入长度 |
| ... | ... | ... |

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

基因是如何表示为 token 的？

### 5.2 表达值编码

表达值是如何编码的？连续值 / 离散化 / 排序？

### 5.3 特殊 Token

[CLS], [SEP], [MASK], [PAD] 等用途。

### 5.4 位置编码

绝对位置 / 相对位置 / 可学习 / 无位置编码？

---

## 6. 预训练

### 6.1 预训练数据

| 数据来源 | 细胞数量 | 组织覆盖 |
|---------|---------|---------|
| ... | ... | ... |

### 6.2 预训练目标

```python
# 损失函数
loss = ...
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-4 |
| Batch Size | 64 |
| 训练步数 | 100K |
| 优化器 | AdamW |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| 细胞类型分类 | Fine-tune / Zero-shot | ... |
| 基因网络推断 | 注意力权重 / 微调 | ... |
| 扰动预测 | ... | ... |
| 批次校正 | ... | ... |

---

## 8. 代码结构速览

```
project/
├── model.py          ← 模型定义
├── dataset.py        ← 数据加载
├── train.py          ← 训练脚本
├── config.py         ← 配置
└── evaluate.py       ← 评估
```

### 快速开始

```bash
# 如何下载和运行
git clone <repo>
cd <repo>
pip install -e .
python run.py --help
```

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
