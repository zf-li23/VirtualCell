---
status: done
filled: 2026-05-28
---

# GEARS 学习笔记

> GEARS (Graph-Enhanced gene Expression And Regulatory System) 是一个结合基因共表达网络和 Gene Ontology 知识的图神经网络方法，用于预测单基因和多基因扰动的转录响应。它的核心思想是通过 GNN 将扰动效应沿基因功能关系图传播，支持零样本预测未见过的多基因组合扰动。

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
| **论文** | [Predicting transcriptional outcomes of novel multigene perturbations with GEARS](https://www.nature.com/articles/s41587-023-01905-6) |
| **发布日期** | 2023-10 |
| **出版** | Nature Biotechnology |
| **架构** | 图神经网络 (GNN) + 基因特异性解码器 |
| **预训练任务** | 扰动后基因表达预测（监督学习） |
| **输入** | scRNA-seq Perturb-seq 数据（基因表达 + 扰动条件） |
| **输出** | 扰动后的全基因组表达预测 |
| **词表** | 基因数量（~10K-20K） |
| **参数规模** | ~10-50M（取决于基因数） |
| **预训练数据** | Norman 2019, Adamson 2016, Dixit 2016, Replogle K562/RPE1 |
| **代码** | [GitHub: snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) |
| **许可** | MIT |

### 核心思想

> **GEARS 的独特角度**：它不把扰动看作独立事件，而是利用先验生物知识（GO 语义相似度 + 基因共表达）来建模扰动如何在基因网络中**传播**。当一个或多个基因被扰动时，其效应沿功能关系图传播到相关基因，从而预测组合扰动效果。

---

## 2. 模型架构

### 2.1 整体架构

GEARS 的核心是一个三组件架构：

1. **基因嵌入层**：为每个基因学习一个嵌入向量（`nn.Embedding`）
2. **双图神经网络**：
   - **Gene Ontology 图（GO 图）**：基于 GO 语义相似度的基因-基因关系图，传递扰动效应的全局上下文
   - **基因共表达图**：基于训练数据中 Pearson 相关系数（阈值 0.4）构建的共表达网络，传递位置编码
3. **基因特异性解码器**：每个基因有独立的权重参数（`indv_w1`, `indv_b1`, `indv_w2`, `indv_b2`），用于输出预测

### 2.2 前向传播流程

```
输入: (x, pert_idx) — 当前表达 x + 扰动索引 pert_idx
                       │
                       ▼
        ┌─── 基因嵌入层 ──────────────────┐
        │   gene_emb → emb_trans → BN     │
        └────────────────────────────────┘
                       │
                       ▼
        ┌─── 位置编码 (共表达 GNN) ───────┐
        │   pos_emb → SGConv × K 层       │
        │   使用 G_coexpress 图传播        │
        └────────────────────────────────┘
                       │
                       ▼
        ┌─── 扰动嵌入 (GO GNN) ──────────┐
        │   pert_emb → SGConv × K 层     │
        │   使用 G_go 图传播扰动信息       │
        └────────────────────────────────┘
                       │
                       ▼
        ┌─── 特征融合 ───────────────────┐
        │   base_emb + 0.2*pos_emb +     │
        │   pert_global_emb (融合)        │
        └────────────────────────────────┘
                       │
                       ▼
              ┌─── 共享 MLP ───┐
              │   recovery_w   │
              └────────────────┘
                       │
                       ▼
              ┌─── 基因特异性解码器 ──┐
              │   indv_w1 × out + b1  │
              │   → cross_gene MLP    │
              │   → indv_w2 × cat + b2 │
              └──────────────────────┘
                       │
                       ▼
              ┌─── 残差连接 ────┐
              │  pred = out + x  │
              └────────────────┘
                       │
                       ▼
              预测的扰动后表达
```

### 2.3 关键技术细节

#### 双图设计

| 图类型 | 构建方式 | 作用 |
|--------|---------|------|
| **GO 图** | GO 语义相似度 → 选取 K 个最相似基因（默认 K=20） | 传播扰动效应的全局生物功能上下文 |
| **共表达图** | Pearson 相关系数 > 阈值（默认 0.4）→ 选取 K 个最相似基因 | 提供基因位置编码，增强同表达模式的基因表示 |

#### 图卷积层

使用 `torch_geometric.nn.SGConv`（简化图卷积），每层聚合 1 跳邻域信息：

```python
for layer in self.sim_layers:
    pert_global_emb = layer(pert_global_emb, G_sim, G_sim_weight)
    if idx < self.num_layers - 1:
        pert_global_emb = pert_global_emb.relu()
```

#### 基因特异性解码

每个基因有一个独立的线性变换参数：

```python
self.indv_w1 = nn.Parameter(torch.rand(num_genes, hidden_size, 1))  # [G, H, 1]
self.indv_b1 = nn.Parameter(torch.rand(num_genes, 1))               # [G, 1]
self.indv_w2 = nn.Parameter(torch.rand(1, num_genes, hidden_size+1))  # [1, G, H+1]
self.indv_b2 = nn.Parameter(torch.rand(1, num_genes))                # [1, G]
```

这种设计使得模型可以为不同基因学习不同的表达变化模式。

### 2.4 不确定性估计

GEARS 支持**不确定性模式**，在解码器旁增加一个不确定性头：

```python
self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1])
```

输出每个基因预测的对数方差 `logvar`，可用于：
- 识别不可靠的预测
- 在损失函数中自适应地加权（不确定性正则化）

---

## 3. 核心创新

### 3.1 图引导的扰动传播（核心创新）

GEARS 的核心创新在于利用**先验生物知识图**来引导扰动效应的传播。与之前将每个基因扰动视为独立输入的方法不同，GEARS 通过双图架构实现了：

- **扰动效应沿功能关联传播**：当一个基因被扰动时，其影响通过 GO 图传播到功能相关的基因上
- **组合扰动理解**：通过组合多个单基因的扰动嵌入来推断组合扰动效果
- **生物学可解释性**：图结构本身反映了真实的基因调控关系和功能关联

### 3.2 零样本组合扰动预测

GEARS 可以预测**训练中从未见过的基因组合**的扰动效果。这是通过将组合扰动分解为其组成的单基因扰动嵌入，融合后输入解码器实现的。

### 3.3 基因特异性解码

使用独立参数的基因特异性解码器（而非共享解码器）让模型可以为每个基因学习个性化的响应模式，这对于捕捉不同基因对扰动的高度异质性响应至关重要。

### 3.4 评估指标

GEARS 使用多维度评估指标：

- **所有基因**：Pearson 相关系数、MSE
- **DE 基因**：差异表达基因上的 Pearson、MSE
- **方向正确率**：预测的变化方向与真实方向一致的基因比例
- **范围内比例**：预测值落在真实值分布范围内的比例

---

## 4. 数据预处理

GEARS 通过 `PertData` 类管理数据加载和预处理：

### 4.1 内置数据集

支持直接加载以下公开数据集：

| 数据集 | 来源 | 描述 |
|--------|------|------|
| `norman` | Norman et al. 2019 | 组合 CRISPR 扰动，~200 个扰动 |
| `adamson` | Adamson et al. 2016 | MAPK 信号通路相关扰动 |
| `dixit` | Dixit et al. 2016 | Perturb-seq 早期数据集 |
| `replogle_k562_essential` | Replogle et al. 2022 | K562 必需基因筛选（经筛选） |
| `replogle_rpe1_essential` | Replogle et al. 2022 | RPE1 必需基因筛选（经筛选） |

### 4.2 自定义数据

用户可以通过提供 Scanpy AnnData 对象来加载自定义数据，需要包含：

- `adata.var['gene_name']`：基因名称
- `adata.obs['condition']`：扰动条件
- `adata.obs['cell_type']`：细胞类型

### 4.3 GO 图构建

从哈佛 Dataverse 下载 `gene2go_all.pkl`，计算基因之间的 GO 语义相似度，构建扰动基因的 GO 图。

### 4.4 数据切分

支持多种切分策略：
- `simulation`：仿真切分（按扰动类型划分）
- `combination`：组合扰动切分
- `gene`：按基因切分

---

## 5. Tokenization 与输入编码

### 5.1 基因 ID 嵌入

每个基因通过 `nn.Embedding(num_genes, hidden_size)` 学习一个嵌入向量，约束为 `max_norm=True`。

### 5.2 图数据格式

GEARS 使用 PyTorch Geometric 的 `Data` 对象表示每个细胞：

```
Data(
    x: [num_genes, 1]        # 当前基因表达
    y: [num_genes, 1]        # 目标基因表达（扰动后）
    pert: [str]              # 扰动类型名称
    pert_idx: [list]         # 扰动索引
    de_idx: [list]           # DE 基因索引
    batch: [num_genes]       # batch 分配
)
```

### 5.3 控制细胞

控制（未扰动）细胞的表达式作为基线，在评估中计算表达变化时使用。

---

## 6. 预训练

GEARS 的训练是**端到端的监督学习**，而非传统的自监督预训练。

### 6.1 训练目标

最小化预测表达与真实表达之间的 MSE：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

### 6.2 不确定性损失

当启用不确定性模式时，损失函数为：

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^N \left(\frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2\right)$$

### 6.3 方向损失

可选的辅助损失项，惩罚预测变化方向与真实方向不一致的预测：

$$\mathcal{L}_{\text{dir}} = \text{mean}\left(\text{ReLU}\left(-\text{sign}(\Delta y_{\text{true}}) \cdot \Delta y_{\text{pred}}\right)\right)$$

### 6.4 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_size` | 64 | 隐藏层维度 |
| `num_go_gnn_layers` | 1 | GO-GNN 层数 |
| `num_gene_gnn_layers` | 1 | 共表达 GNN 层数 |
| `decoder_hidden_size` | 16 | 解码器隐藏维度 |
| `coexpress_threshold` | 0.4 | 共表达图相关系数阈值 |
| `uncertainty` | False | 是否启用不确定性估计 |

---

## 7. 下游任务

GEARS 的主要任务是**扰动后基因表达预测**，支持：

### 7.1 单基因扰动预测

预测单个基因敲除/敲低后的全基因组表达变化。

### 7.2 多基因组合扰动预测

预测多个基因同时扰动后的表达变化。这是 GEARS 区别于传统方法的标志性能力。

### 7.3 基因相互作用（GI）预测

通过 `GI_predict` 方法，GEARS 可以预测基因间的遗传相互作用（合成致死、缓冲等）：

```python
gears_model.GI_predict(['CBL', 'CNN1'])
```

---

## 8. 代码结构速览

```
gears/
├── __init__.py          # 包初始化
├── model.py             # GEARS_Model (PyTorch) — 核心模型定义
│   ├── MLP              # 多层感知机工具
│   └── GEARS_Model      # 主模型（双 GNN + 基因特异性解码器）
├── gears.py             # GEARS 类 — 高层 API
│   ├── model_initialize # 初始化模型参数
│   ├── train            # 训练循环
│   ├── predict          # 预测 API
│   └── GI_predict       # 基因相互作用预测
├── pertdata.py          # PertData 类 — 数据处理
│   ├── load             # 加载内置/自定义数据集
│   ├── prepare_split    # 数据切分
│   └── get_dataloader   # 获取 DataLoader
├── inference.py         # 评估函数
│   ├── evaluate         # 推理模式
│   └── compute_metrics  # 指标计算
├── data_utils.py        # 数据处理工具（DE 基因、切分器）
└── utils.py             # 通用工具函数（图构建、损失函数等）
```

---

## 9. 关键概念 Q&A

### Q1: GEARS 和 CPA 的主要区别是什么？

GEARS 和 CPA 都是 2023 年发布的扰动预测方法，但设计理念不同：

| 方面 | GEARS | CPA |
|------|-------|-----|
| **核心技术** | 图神经网络 + 先验知识图 | 变分自编码器 + 加性解耦 |
| **知识使用** | GO 图 + 共表达网络 | 仅从数据学习（可选 RDKit 嵌入） |
| **组合扰动** | 通过 GNN 传播融合 | 通过加性组合 |
| **剂量建模** | 不支持 | 支持（Doser 网络） |
| **跨细胞类型** | 不支持 | 支持（协变量嵌入） |
| **开源框架** | 独立 PyTorch | scvi-tools 生态 |

### Q2: GEARS 的局限性是什么？

1. **不支持跨细胞类型训练**：目前设计为在单一细胞类型上训练
2. **未在 bulk 数据上测试**：可能不适用于批量测序数据
3. **需要组合扰动数据**：仅用单基因扰动数据训练时，无法可靠预测组合扰动
4. **数据需求**：需要足够多的每个扰动的细胞数和扰动类型数

### Q3: GEARS 在后续工作中有哪些影响？

GEARS 深刻影响了扰动预测领域：
- **Systema**（评估框架）中将其作为对比基线
- **PertAdapt**（2025）尝试解决 GEARS 不支持跨细胞类型的问题
- **scLAMBDA**（2024）在 GEARS 的图传播思想基础上扩展

---

## 10. 延伸阅读

- [CPA](https://www.embopress.org/doi/full/10.15252/msb.202211517) — 同期发表，互补的组合扰动方法
- [Systema](https://www.nature.com/articles/s41587-025-02777-8) — 扰动预测系统性评估框架
- [scLAMBDA](https://www.biorxiv.org/content/10.1101/2024.12.04.626878v1) — 多基因扰动预测的后续工作
- [PertAdapt](https://www.biorxiv.org/content/10.1101/2025.11.21.689655v1) — 跨条件适应方法
