# scPoli 学习笔记

> scPoli (Single-Cell Population-level Integration) 是一个基于 **条件变分自编码器 (cVAE)** 的框架，用于在种群级别整合单细胞数据集，实现跨样本、跨条件的多尺度分析。它是单细胞领域**最早引入"参考映射 + 迁移学习"范式**的方法之一。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理](#4-数据预处理)
5. [训练策略](#5-训练策略)
6. [下游任务](#6-下游任务)
7. [关键概念 Q&A](#7-关键概念-qa)
8. [延伸阅读](#8-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [Population-level integration of single-cell datasets enables multi-scale analysis across samples](https://www.nature.com/articles/s41592-023-02035-2), Nature Methods 2023 |
| **架构** | 条件变分自编码器 (Conditional VAE) |
| **学习范式** | 半监督 + 自监督 |
| **输入** | 基因表达矩阵 + 样本标签 (可选) |
| **输出** | 细胞嵌入 (低维潜变量) + 重构表达 |
| **核心能力** | 批次校正 + 参考映射 + 标签迁移 |

### 核心思想

> "学习一个**种群级别的嵌入空间**，使得：(1) 同一生物条件下不同样本的细胞对齐，(2) 新样本可以通过**参考映射 (reference mapping)** 快速嵌入到已有空间，(3) 支持从粗粒度（组织）到细粒度（细胞类型）的多尺度分析。"

---

## 2. 模型架构

### 2.1 整体架构

```
Input: x (基因表达), s (样本ID), y (细胞类型标签, 可选)
                  │
    ┌─────────────┴─────────────┐
    │        Encoder q(z|x,s)    │  ← 条件编码器 (输入表达 + 样本ID)
    │   MLP × 2 + 重参数化       │     输出: μ_z, σ_z
    └─────────────┬─────────────┘
                  │ z (潜变量)
    ┌─────────────┴─────────────┐
    │      Classifier q(y|z,s)   │  ← 细胞类型分类器 (半监督)
    │   MLP → Softmax            │     可选的辅助分类任务
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │       Decoder p(x|z,s)     │  ← 条件解码器 (输入 z + 样本ID)
    │   MLP × 2 → 输出分布       │     输出: 零膨胀负二项 (ZINB) 参数
    └─────────────┬─────────────┘
                  │
    Output: 重构表达 + 细胞类型预测
```

### 2.2 条件变分自编码器 (cVAE)

#### 编码器

```python
class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden=128, n_latent=10):
        # n_input: 基因数 (通常 2000-5000 HVGs)
        # n_latent: 潜变量维度 (默认 10)
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.mean = nn.Linear(n_hidden, n_latent)
        self.var = nn.Linear(n_hidden, n_latent)

    def forward(self, x, s):
        # s: 样本 one-hot 编码或嵌入
        x = torch.cat([x, s], dim=-1)  # 条件输入
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.mean(h), self.var(h)
```

#### 解码器 (ZINB 输出)

```python
class Decoder(nn.Module):
    def forward(self, z, s):
        # 输出 ZINB 分布的三个参数
        z = torch.cat([z, s], dim=-1)  # 条件解码
        h = torch.relu(self.fc1(z))
        return {
            'mean': self.mean_head(h),    # 均值
            'disp': self.disp_head(h),    # 离散度
            'dropout': self.dropout(h)    # 丢失概率 (零膨胀)
        }
```

---

## 3. 核心创新

### 3.1 从"对齐"到"映射"的范式转变

scPoli 重新定义了 scRNA-seq 整合的目标：

| 传统方法 (Harmony, scVI) | scPoli |
|-------------------------|--------|
| 同时对所有数据集进行联合嵌入 | 先构建参考嵌入，再映射新数据 |
| 每次整合都需要重新训练 | 参考映射只需一次前向传播 |
| 不支持增量式添加新数据 | 支持流式映射新样本 |
| 计算成本随数据量 O(N³) 增长 | 参考构建 O(N³)，映射 O(1) |

### 3.2 参考映射 (Reference Mapping)

**核心流程**：

```
1. 构建参考
   ┌─────────────┐    ┌──────────────┐
   │ 参考数据集 D1 │ → │ scPoli 训练  │ → 参考潜空间 Z_ref
   │ 参考数据集 D2 │ → │ (cVAE)      │
   └─────────────┘    └──────────────┘

2. 映射新数据
   ┌─────────────┐    ┌──────────────┐
   │ 查询数据集 Q  │ → │ 冻结编码器   │ → 查询嵌入 Z_query
   │ (未见过的)   │    │ (前向传播)   │
   └─────────────┘    └──────────────┘

3. 标签迁移
   Z_query → KNN(Z_ref) → 投票 → 预测细胞类型
```

### 3.3 多尺度分析

通过潜变量的不同分组实现多尺度分析：

```
组织级别: 所有细胞的潜变量 → 聚类 → 组织域识别
样本级别: 同一样本内的细胞 → 条件特异性模式
细胞级别: 单个细胞的潜变量 → 细胞类型 / 状态
```

### 3.4 半监督学习

当部分细胞有标签时，scPoli 使用半监督学习：
```python
# 总损失 = VAE 损失 + 分类损失
loss_total = loss_vae + λ * loss_classification

loss_vae = KL(q(z|x,s) || p(z|s)) + E_q[-log p(x|z,s)]  # ELBO
loss_classification = CrossEntropy(y_pred, y_true)         # 有标签数据
```

---

## 4. 数据预处理

### 4.1 标准预处理 Pipeline

```python
# scanpy 标准流程
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.scale(adata, max_value=10)
```

### 4.2 样本条件编码

```python
# 样本 ID 编码为条件向量
sample_encoder = LabelEncoder()
sample_ids = sample_encoder.fit_transform(adata.obs['sample_id'])
# 转换为 one-hot 或可学习嵌入
sample_embeddings = one_hot(sample_ids, n_samples)
```

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

scPoli 使用标准 scVI-style 的基因编码，不对基因做离散 tokenization，而是将**高变基因的表达值**作为连续向量输入。

### 5.2 样本条件编码

样本 ID 通过 one-hot 编码或可学习嵌入输入编码器和解码器（作为 cVAE 的条件变量）：

```python
sample_encoder = LabelEncoder()
sample_ids = sample_encoder.fit_transform(adata.obs['sample_id'])
sample_embeddings = one_hot(sample_ids, n_samples)
# 或可学习的嵌入
sample_emb = nn.Embedding(n_samples, d_model)(sample_ids)
```

### 5.3 与 FM 类模型的区别

scPoli **不**使用 Transformer/BERT 式的自监督预训练，而是使用 VAE 架构进行表示学习。它的"输入编码"是连续表达向量而非离散 token 序列。

---

## 6. 预训练

### 6.1 两阶段训练策略

```
阶段一：完全自监督（VAE）
  - 使用所有细胞的表达数据
  - 最小化 ELBO 损失
  - 学习基本的生物变异

阶段二：半监督微调
  - 引入部分标注细胞的类型标签
  - 联合优化 VAE + 分类损失
  - 对齐嵌入空间与生物学注释
```

### 6.2 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_latent | 10 | 潜变量维度 |
| n_hidden | 128 | 隐藏层维度 |
| dropout_rate | 0.1 | Dropout 概率 |
| λ (分类权重) | 0.1 | 半监督损失权重 |

---

## 7. 下游任务

### 5.1 两阶段训练

```
阶段一：完全自监督 (仅 VAE)
  - 使用所有细胞的表达数据
  - 最小化 ELBO 损失
  - 学习基本的生物变异

阶段二：半监督微调
  - 引入部分标注细胞的类型标签
  - 联合优化 VAE + 分类损失
  - 对齐嵌入空间与生物学注释
```

### 5.2 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_latent | 10 | 潜变量维度 |
| n_hidden | 128 | 隐藏层维度 |
| n_layers | 2 | 编码器/解码器层数 |
| dropout_rate | 0.1 | Dropout 概率 |
| λ (分类权重) | 0.1 | 半监督损失权重 |

### 5.3 参考映射微调

```python
def map_query(query_adata, reference_model):
    """将查询数据映射到参考空间"""
    # 冻结编码器
    reference_model.eval()
    
    # 前向传播获取潜变量
    query_latent = reference_model.encoder(
        query_adata.X, query_adata.obs['sample_id']
    )
    
    # 可选的投影微调
    # 允许对编码器最后一层进行少量更新以适配新数据
    if args.finetune:
        query_latent = reference_model.encoder_finetune(
            query_adata.X, query_adata.obs['sample_id']
        )
    
    return query_latent
```

---

## 8. 代码结构速览

```
scPoli/
├── model/
│   ├── encoder.py          ← 条件编码器 (cVAE)
│   ├── decoder.py          ← ZINB 解码器
│   └── classifier.py       ← 细胞类型分类头
├── data/
│   ├── preprocess.py       ← scVI 标准预处理
│   └── dataset.py          ← PyTorch Dataset
├── train.py                ← 两阶段训练
├── reference_mapping.py    ← 参考映射新数据
└── evaluate.py             ← 性能评估
```

---

## 9. 关键概念 Q&A

### Q1: scPoli 与 scVI 有何不同？

| 维度 | scVI (2018) | scPoli (2023) |
|------|-------------|---------------|
| 架构 | VAE | cVAE + 半监督 |
| 批次处理 | 批次标签作为编码器输入 | 样本标签作为编码器+解码器条件 |
| 主要用途 | 数据整合 + 批次校正 | 参考映射 + 迁移学习 |
| 新数据 | 需要重新训练 | 前向映射即可 |
| 半监督 | 不支持 | 支持 |

### Q2: scPoli 的局限性？

1. **参考质量依赖**：映射质量强烈依赖于参考数据的质量和代表性
2. **潜变量维度敏感**：维度太小丢失信息，太大引入噪声
3. **不适合发现新类型**：参考映射倾向于将新细胞分类为已知类型
4. **线性映射假设**：假设查询与参考共享相似的基因集和分布

---

## 10. 延伸阅读

- **[scVI](https://www.nature.com/articles/s41592-018-0229-2)**：Lopez et al., Nat Methods 2018 — scPoli 的架构前身
- **[scArches](https://www.nature.com/articles/s41587-021-00867-x)**：Lotfollahi et al., Nat Biotech 2021 — 条件 VAE + 参考映射的早期工作
- **[scANVI](https://www.nature.com/articles/s41587-021-00867-x)**：scVI 的半监督扩展
- **[TotalVI](https://www.nature.com/articles/s41587-021-00867-x)**：多模态 (RNA + 蛋白) VAE
