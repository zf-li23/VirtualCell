# SpaceFlow 学习笔记

> SpaceFlow 是一个基于**深度图网络**的空间转录组分析方法，通过结合 **Deep Graph Infomax (DGI)** 和**空间正则化**来识别组织中的空间域，并推断**伪时空图 (pSM)** 用于分析组织发育和疾病进展的连续过程。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [图构建：Alpha Complex](#4-图构建alpha-complex)
5. [训练](#5-训练)
6. [伪时空图 (pSM) 推断](#6-伪时空图-psm-推断)
7. [代码结构速览](#7-代码结构速览)
8. [关键概念 Q&A](#8-关键概念-qa)
9. [延伸阅读](#9-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [SpaceFlow: Identifying Spatial Domains and Spatiotemporal Patterns in Spatial Transcriptomics](https://www.nature.com/articles/s41467-023-40173-4), Nature Communications 2023 |
| **架构** | GNN Encoder (2-layer GCNConv + PReLU) + DGI |
| **预训练任务** | Deep Graph Infomax + 空间正则化 |
| **图构建** | KNN + Alpha Complex (自适应空间邻接) |
| **下游输出** | 空间域 + 伪时空图 (pSM) |
| **分割算法** | Leiden 聚类 |

### 核心思想

> "用图神经网络学习空间转录组的嵌入表示，同时通过**空间正则化**强制嵌入反映组织的空间连续性，再通过**伪时空图 (pSM)** 推断细胞/spot 沿着组织发育或疾病进展的时间顺序。"

---

## 2. 模型架构

### 2.1 整体架构

```
空间坐标 + 基因表达
        │
        ▼
┌─────────────────────┐
│    Alpha Complex    │  ← KNN + Alpha Complex 构建空间图
│    图构建           │
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│   GraphEncoder      │  ← 2层 GCNConv + PReLU
│   图编码器           │     (n_genes → 32 → 32)
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐   ┌───────────┐
│ DGI Loss │   │ 空间正则化  │
│ (互信息)  │   │ (距离惩罚)  │
└─────────┘   └───────────┘
    │             │
    └──────┬──────┘
           │
     ┌─────┴─────┐
     │  嵌入输出   │  d=32
     └─────┬─────┘
           │
     ┌─────┴─────┐
     │  Leiden   │  → 空间域分割
     │  UMAP     │  → 可视化
     │  PAGA     │  → 伪时空图 (pSM)
     └───────────┘
```

### 2.2 GraphEncoder (图编码器)

```python
# SpaceFlow.py 中的编码器
self.encoder = nn.Sequential(
    GCNConv(n_genes, 32),    # 第一层: 基因表达维度 → 32
    PReLU(32),                # 参数化 ReLU (可学习斜率)
    GCNConv(32, 32),          # 第二层: 32 → 32
    PReLU(32),
)
```

**为什么用 PReLU 而非 ReLU？**
- PReLU (Parametric ReLU) 为每个通道学习一个负半轴的斜率参数
- 比 ReLU 更灵活，可以保留潜在的负向信号
- 参数量仅增加 d_model 个额外参数，计算开销极小

### 2.3 DGI (Deep Graph Infomax)

SpaceFlow 使用与 GraphST 相同的 DGI 框架：

```python
# DGI 损失
def dgi_loss(h, s, h_shuffle, discriminator):
    """
    h: 节点嵌入
    s: 整图嵌入 (由所有节点嵌入聚合)
    h_shuffle: 打乱特征后的节点嵌入 (负样本)
    discriminator: Bilinear 判别器
    """
    pos = discriminator(h, s)       # 正样本分数
    neg = discriminator(h_shuffle, s)  # 负样本分数
    
    loss = - (F.logsigmoid(pos).mean() + 
              F.logsigmoid(-neg).mean())
    return loss
```

---

## 3. 核心创新

### 3.1 空间正则化 (Spatial Regularization)

这是 SpaceFlow 区别于 GraphST 的关键创新：

```python
# 空间正则化损失
def spatial_regularization(z, sp_dists, z_dists):
    """
    z: 节点嵌入 [n, d]
    sp_dists: 空间距离矩阵 [n, n]
    z_dists: 嵌入距离矩阵 [n, n]
    
    核心思想：
        如果两个节点在空间中很近 (sp_dists 小)，
        那么它们的嵌入也应该接近 (z_dists 小)
    """
    # 空间距离权重 (高斯核)
    w = torch.exp(-sp_dists² / σ²)  # 近的节点权重大
    
    # 空间正则化: 加权平均的嵌入距离
    loss = (w * z_dists).sum() / w.sum()
    
    return loss
```

**直观理解**：
- 两个空间上邻近的 spot/细胞，通常具有相似的微环境和功能
- 空间正则化强制模型学习到**空间连续的嵌入空间**
- 这有助于平滑分割，减少噪声导致的孤立区域

### 3.2 总损失函数

```python
total_loss = λ₁ · L_dgi + λ₂ · L_spatial

# λ₁: DGI 权重 (通常 1.0)
# λ₂: 空间正则化权重 (可调超参数)
```

### 3.3 从分割到连续轨迹

SpaceFlow 独特的**两步流程**：

```
第一步: 空间域分割 (离散)
    Embedding → Leiden 聚类
    → 识别不同的组织结构区域

第二步: 伪时空推断 (连续)
    Embedding → UMAP → Leiden → PAGA → diffmap → dpt
    → 推断组织内的连续变化轨迹
```

---

## 4. 图构建：Alpha Complex

### 4.1 Alpha Complex 简介

Alpha Complex 是 SpaceFlow 用于构建空间邻接图的方法：

```
Alpha Complex 构建过程：

1. KNN (K-Nearest Neighbors):
   对每个点，找到最近的 k 个邻居
   构建初步的邻接图
   
2. Alpha Complex 过滤:
   计算 Delaunay 三角剖分
   对于每个三角形，如果外接圆半径 > alpha，移除该边
   保留的边构成 Alpha Complex
```

### 4.2 为什么用 Alpha Complex？

| 方法 | 优点 | 缺点 |
|------|------|------|
| **简单 KNN** | 实现简单 | 可能连接不相关的区域 |
| **Delaunay** | 保持拓扑结构 | 可能产生过长连接 |
| **Alpha Complex** | 自适应的局部连接 | 需要选择 alpha 参数 |

Alpha Complex 能在保持局部拓扑的同时，避免跨越组织边界的长连接。

### 4.3 在图中的使用

```python
# SpaceFlow 中的图构建
def build_graph(coords, k=15, alpha=1.0):
    """
    1. 计算 KNN 图 (k=15)
    2. 基于 Alpha Complex 修剪边
    3. 构建 PyG (PyTorch Geometric) 的 Data 对象
    """
    # 计算 KNN
    distances, indices = knn(coords, k=k)
    
    # Alpha Complex 过滤
    edge_index = alpha_complex_filter(
        coords, indices, distances, alpha=alpha
    )
    
    return Data(edge_index=edge_index)
```

---

## 5. 训练

### 5.1 训练流程

```python
class SpaceFlow:
    def train(self, n_epochs=500):
        for epoch in range(n_epochs):
            # 1. 编码
            z = self.encoder(x, edge_index)
            
            # 2. DGI 损失
            s = self.readout(z)
            z_shuffle = self._shuffle_features(x)
            h_shuffle = self.encoder(z_shuffle, edge_index)
            loss_dgi = self.dgi_loss(z, s, h_shuffle)
            
            # 3. 空间正则化损失
            loss_spatial = self.spatial_regularization(
                z, sp_dists, self._pairwise_dist(z)
            )
            
            # 4. 总损失
            loss = loss_dgi + self.lambda_spatial * loss_spatial
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 5.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 嵌入维度 | 32 | 输出嵌入维度 |
| GCN 层数 | 2 | 图卷积层数 |
| KNN k | 15 | 最近邻个数 |
| Alpha | 1.0 | Alpha Complex 参数 |
| λ_spatial | 0.1~1.0 | 空间正则化权重 |
| Epochs | 500 | 训练轮数 |
| 学习率 | 0.001 | Adam 优化器 |

---

## 6. 伪时空图 (pSM) 推断

### 6.1 什么是伪时空推断？

```
问题：空间转录组只提供单时间点的"快照"
但组织内的细胞可能处于不同的发育/疾病阶段

解决方案 (SpaceFlow):
    利用嵌入空间中的连续变化 → 推断"伪时间"
    + 空间位置信息 → 构建"伪时空图"
```

### 6.2 pSM 构建流程

```python
def pseudo_spatiotemporal_map(embeddings, coords):
    """
    1. UMAP 降维到 2D
    2. Leiden 聚类 → 识别亚群
    3. PAGA (Partition-based Graph Abstraction) → 构建轨迹图
    4. diffmap (扩散映射) → 捕获连续变化
    5. dpt (扩散伪时间) → 推断每个细胞的时间顺序
    """
    # 1. 降维
    umap_coords = UMAP(embeddings)
    
    # 2. 聚类
    clusters = leiden(embeddings, resolution=1.0)
    
    # 3. PAGA 轨迹
    paga = PAGA(embeddings, clusters)
    paga.compute()
    
    # 4. 扩散映射 + 伪时间
    dm = DiffusionMap(paga)
    dpt = DPT(dm)
    
    # 返回: 每个细胞的伪时间值
    return dpt.pseudotime
```

### 6.3 伪时空 vs 伪时间

| 概念 | 说明 |
|------|------|
| **伪时间 (Pseudotime)** | 沿发育/疾病轨迹的"时间"顺序 (单变量) |
| **伪时空 (pSM)** | 同时考虑伪时间 + 空间位置的二维映射 |

伪时空图是一个 2D 映射，x 轴为空间位置，y 轴为伪时间，展示**空间-时间的连续变化**。

### 6.4 pSM 的应用

```python
# 可视化伪时空模式
sc.pl.embedding(adata, basis="pSM", color="pseudotime", 
                title="Pseudo-Spatiotemporal Map")

# 识别沿伪时空变化的基因
sc.tl.paga(adata, groups="leiden")
sc.pl.paga(adata, color="pseudotime")
```

---

## 7. 代码结构速览

```
SpaceFlow/
├── SpaceFlow/
│   ├── SpaceFlow.py          # 主类 (486行)
│   │   ├── 预处理和标准化
│   │   ├── 图构建 (graph_alpha)
│   │   ├── 模型定义 (编码器)
│   │   ├── 训练 (DGI + 空间正则化)
│   │   ├── 空间域分割 (Leiden)
│   │   └── 伪时空图 (pSM)
│   │
│   ├── __init__.py
│   └── ...
│
├── tutorials/                 # Jupyter Notebook 教程
├── images/                    # 示例图片
├── pyproject.toml             # 项目配置
├── setup.cfg                  # 安装配置
└── README.md
```

---

## 8. 关键概念 Q&A

### Q1: SpaceFlow 和 GraphST 的核心区别？

| 方面 | GraphST | SpaceFlow |
|------|---------|-----------|
| **预训练** | DGI | DGI + 空间正则化 |
| **图构建** | 普通KNN/半径 | Alpha Complex |
| **激活函数** | ReLU | **PReLU** |
| **嵌入维度** | 16 | **32** |
| **输出** | 空间域 | 空间域 + **pSM** |
| **下游分析** | 空间域分割 | 空间域 + 连续轨迹推断 |

### Q2: 什么是 PReLU 以及为什么用 PReLU？

```
ReLU:  f(x) = max(0, x)
PReLU: f(x) = max(0, x) + α·min(0, x)

其中 α 是可学习的参数 (每个通道一个)
当 α=0 时，PReLU = ReLU
当 α>0 时，保留负值信息

在 SpaceFlow 中，PReLU 被用于 GCNConv 之后
使模型可以保留潜在的负向信号
```

### Q3: 什么是 Alpha Complex？

- Alpha Complex 是计算几何中的概念，是 Delaunay 三角剖分的一个子集
- 通过参数 α 控制连接的"稠密程度"
- α 大 → 更多连接 (接近 Delaunay)
- α 小 → 更少连接 (只保留局部稠密区域)
- 在 ST 中，α 可以根据细胞密度自适应调整

### Q4: 伪时空图的生物学意义？

**真实场景示例 - 脑发育**：
```
空间位置: 脑室区 → 中间区 → 皮质板
伪时间:   早期 → 中期 → 晚期

pSM 揭示：
- 神经干细胞在脑室区 (早期)
- 迁移中的神经元在中间区 (中期)
- 成熟神经元在皮质板 (晚期)

这是一个空间 + 时间的连续过程
常规聚类只能看到"离散的区域"
pSM 能看到"从早期到晚期的连续变化"
```

### Q5: 空间正则化在什么时候特别重要？

- **高噪声数据**：空间正则化提供额外的平滑约束
- **稀疏组织**：当某些区域细胞密度低，正则化帮助维持空间连续性
- **跨切片整合**：多个组织切片的联合分析时，正则化有助于对齐

---

## 9. 延伸阅读

### 核心论文

1. **H Ren et al.** *SpaceFlow: Identifying Spatial Domains and Spatiotemporal Patterns in Spatial Transcriptomics.* Nature Communications, 2023. [阅读](https://www.nature.com/articles/s41467-023-40173-4)

### 相关技术

- **Alpha Complex**：[Computational Topology: An Introduction](https://www.amazon.com/Computational-Topology-Introduction-Herbert-Edelsbrunner/dp/0821849255)
- **PAGA**：[PAGA: graph abstraction reconciles clustering with trajectory inference](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1663-x)
- **扩散伪时间 (DPT)**：[Diffusion pseudotime robustly reconstructs branching cellular lineages](https://www.nature.com/articles/nmeth.3971)

### 学习路线

```
空间转录组图模型：

1. GraphST (DGI, 基础版) 
    ↓ 
2. SpaceFlow (DGI + 空间正则化 + pSM)  ← 这一篇
    ↓
3. SPADE / SpaGCN (加入 H&E 多模态)
```

---

> 💡 **学习建议**：SpaceFlow 的核心创新是**空间正则化**和**伪时空推断**。建议对比 SpaceFlow 和 GraphST 的代码实现，理解 DGI 基础上增加的空间连续性约束如何影响嵌入质量。pSM 的概念值得深入理解——它是空间转录组特有的分析维度，传统单细胞分析没有这个能力。
