# GraphST 学习笔记

> GraphST 是一个基于**图神经网络 (GNN)** 和**自监督对比学习 (Deep Graph Infomax)** 的空间转录组分析方法，通过图上的对比学习学习 spot/细胞的空间感知表示，并支持细胞-空间 spot 的配对映射。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理](#4-数据预处理)
5. [训练 (Deep Graph Infomax)](#5-训练-deep-graph-infomax)
6. [下游任务](#6-下游任务)
7. [代码结构速览](#7-代码结构速览)
8. [关键概念 Q&A](#8-关键概念-qa)
9. [延伸阅读](#9-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [GraphST: Graph-based Self-supervised Learning for Spatial Transcriptomics](https://www.nature.com/articles/s41467-023-40173-4), Nature Communications 2023 |
| **架构** | GNN Encoder + Discriminator (Bilinear) — Deep Graph Infomax (DGI) |
| **预训练任务** | 图-补丁对比 (Deep Graph Infomax) — 最大化整图与局部补丁的互信息 |
| **输入** | 空间转录组的邻接图 + 基因表达 |
| **输出** | spot/细胞的图感知嵌入 |
| **扩展模块** | Encoder_sc (单细胞自编码器), Encoder_map (细胞-spot 映射) |

### 核心思想

> "将空间转录组测到的每个 spot 视为图上的一个节点，通过 **Deep Graph Infomax** 对比学习最大化整图表示与局部节点表示的互信息，让嵌入同时编码**基因表达模式**和**空间组织结构**。"

---

## 2. 模型架构

### 2.1 核心架构：Deep Graph Infomax (DGI)

```
空间转录组图 G = (X, A)
    X: 节点特征 (基因表达)
    A: 邻接矩阵 (空间邻近)
                │
    ┌───────────┴───────────┐
    │      Encoder          │  ← GNN: X' = A·X·W₁ + A²·X·W₂
    │  (GCN / GAT)          │     (2层图卷积)
    └───────────┬───────────┘
                │
         ┌──────┴──────┐
         │             │
         ▼             ▼
    ┌─────────┐   ┌──────────┐
    │ 节点嵌入 │   │ 整图嵌入  │
    │ H ∈ Rⁿᵈ  │   │ s ∈ Rᵈ   │ ← AvgReadout (平均池化)
    └────┬────┘   └────┬─────┘
         │             │
         └──────┬──────┘
                │
         ┌──────┴──────┐
         │ Discriminator│  ← Bilinear: D(hᵢ, s) = hᵢᵀ·W·s
         │ (Bilinear)  │     判断(hᵢ, s)是正样本还是负样本
         └──────┬──────┘
                │
         ┌──────┴──────┐
         │  DGI Loss   │  ← 最大化互信息
         └─────────────┘
```

### 2.2 Encoder (图编码器)

```python
# model.py
class Encoder(nn.Module):
    """
    2层图卷积编码器，支持 DGI 自监督学习
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        # weight1: 第一层参数 (自环连接)
        # weight2: 第二层参数 (邻居聚合)
        self.weight1 = nn.Parameter(torch.randn(in_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.randn(hidden_dim, out_dim))
    
    def forward(self, x, adj):
        # 第一层: X' = A·X·W₁
        h = torch.mm(adj, x)          # 邻居聚合
        h = torch.mm(h, self.weight1) # 线性变换
        h = F.relu(h)
        
        # 第二层: X'' = A·X'·W₂
        h = torch.mm(adj, h)
        h = torch.mm(h, self.weight2)
        
        return h
```

**简化的 GCN 设计**：没有使用 nn.Linear + 激活的标准封装，而是手动实现矩阵乘法。

### 2.3 Discriminator (判别器)

```python
# model.py
class Discriminator(nn.Module):
    """
    Bilinear 判别器: D(hᵢ, s) = hᵢᵀ · W · s
    
    输入: 节点嵌入 hᵢ 和 整图嵌入 s
    输出: 分数 (正样本越高越好)
    """
    def __init__(self, in_dim):
        self.W = nn.Parameter(torch.randn(in_dim, in_dim))
    
    def forward(self, h, s):
        # h: [n_nodes, d], s: [1, d]
        # 输出: [n_nodes, 1] 每个节点的分数
        return torch.mm(torch.mm(h, self.W), s.t())
```

### 2.4 AvgReadout (平均读出)

```python
# model.py
class AvgReadout(nn.Module):
    """
    加权平均池化: 将节点嵌入聚合为整图嵌入
    """
    def __init__(self, in_dim):
        self.attn = nn.Parameter(torch.randn(in_dim, 1))
    
    def forward(self, h, mask=None):
        # 注意力权重
        w = torch.softmax(torch.mm(h, self.attn), dim=0)
        # 加权求和
        s = torch.sum(h * w, dim=0, keepdim=True)
        return s
```

---

## 3. 核心创新

### 3.1 Deep Graph Infomax (DGI) 在 ST 中的应用

DGI 原本是为通用图结构设计的自监督方法，GraphST 将其创新地应用于空间转录组：

```
DGI 核心思想：
    最大化 整图嵌入 s 和 局部节点嵌入 hᵢ 的互信息
        
    正样本: (原始图节点 hᵢ, 整图嵌入 s)
    负样本: (打乱特征的节点 h̃ᵢ, 整图嵌入 s)
    
    损失函数:
    L = -1/N Σᵢ [log(D(hᵢ, s)) + log(1 - D(h̃ᵢ, s))]
    
    其中 D 是 Bilinear 判别器
```

### 3.2 稀疏图支持

```python
# model.py
class Encoder_sparse(nn.Module):
    """
    使用 torch.spmm (稀疏矩阵乘法) 替代普通 mm
    适用于大规模空间转录组数据
    """
    def forward(self, x, adj):
        # adj 是稀疏邻接矩阵
        h = torch.spmm(adj, x)        # 稀疏 × 稠密
        h = torch.mm(h, self.weight1)
        h = F.relu(h)
        
        h = torch.spmm(adj, h)
        h = torch.mm(h, self.weight2)
        return h
```

### 3.3 Encoder_sc (单细胞解耦)

```python
# model.py
class Encoder_sc(nn.Module):
    """
    3层自编码器: 从图嵌入中解耦单细胞信息
    
    架构: 256 → 64 → 32 → 64 → 256
    
    用途: 当需要从空间 spot 数据中推断单细胞级别信息时使用
    """
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )
```

### 3.4 Encoder_map (细胞-spot 映射)

```python
# model.py
class Encoder_map(nn.Module):
    """
    可学习的 M 矩阵: 用于将单细胞数据映射到空间位置
    
    M: [n_cell_types, n_spots] 或 [n_cells, n_spots]
    学习每个细胞类型/细胞与每个 spot 的关联强度
    """
    def __init__(self, n_cells, n_spots):
        self.M = nn.Parameter(torch.randn(n_cells, n_spots))
    
    def forward(self, cell_emb, spot_emb):
        # M 矩阵 × 细胞嵌入 → 推测哪些细胞在哪些 spot
        return torch.mm(self.M.t(), cell_emb)
```

---

## 4. 数据预处理

### 4.1 空间邻接图构建

```
空间坐标 (x, y)
    │
    ▼
┌───────────────────────────────┐
│ 1. 计算空间距离矩阵             │
│    欧氏距离 ||pᵢ - pⱼ||₂       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 2. KNN 构建邻接                │
│    或 半径阈值法 (r < threshold)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 3. 归一化邻接矩阵               │
│    D⁻¹/² · A · D⁻¹/²          │
└───────────────┬───────────────┘
                │
                ▼
        邻接矩阵 A ∈ ℝⁿˣⁿ
```

### 4.2 基因表达预处理

```
原始表达
    │
    ▼
┌───────────────────┐
│ 标准化            │  normalize_total + log1p
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 基因选择           │  选择高变基因
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 降维 (可选)        │  PCA → 50~200 维
└────────┬──────────┘
         │
         ▼
  节点特征 X ∈ ℝⁿˣᵈ
```

---

## 5. 训练 (Deep Graph Infomax)

### 5.1 训练流程

```python
def train_step(model, x, adj):
    # 1. 编码原始图 → 节点嵌入
    h = model.encoder(x, adj)
    
    # 2. 整图嵌入
    s = model.readout(h)
    
    # 3. 创建负样本 (打乱节点特征)
    x_shuffle = shuffle(x)  # 按行打乱
    h_shuffle = model.encoder(x_shuffle, adj)
    
    # 4. 判别器打分
    pos_score = model.discriminator(h, s)     # 正样本: 高分
    neg_score = model.discriminator(h_shuffle, s)  # 负样本: 低分
    
    # 5. DGI 损失
    loss = - (log_sigmoid(pos_score).mean() 
              + log_sigmoid(-neg_score).mean())
    
    return loss
```

### 5.2 负样本构建

GraphST 使用**特征打乱 (feature shuffling)** 构建负样本：

```
原始特征矩阵 X:          打乱后 X_shuffle:
    [g₁, g₂, ..., g_d]₁     [g₃, g₇, ..., g₁]₁
    [g₁, g₂, ..., g_d]₂     [g₅, g₂, ..., g₄]₂
    [g₁, g₂, ..., g_d]₃     [g₁, g₉, ..., g₃]₃
    ...                      ...

保持图结构不变，只打乱节点特征
→ 模型学到：特征 + 结构 = 完整信息
```

### 5.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 隐藏层维度 | 32 | 编码器中间层 |
| 输出维度 | 16 | 嵌入维度 |
| 学习率 | 0.01 | Adam 优化器 |
| Epochs | 1000 | 训练轮数 |
| 负采样方式 | 特征打乱 | 保持图结构不变 |

---

## 6. 下游任务

### 6.1 支持的任务

| 任务 | 说明 | 方法 |
|------|------|------|
| **空间域分割** | 识别组织结构区域 | 对嵌入进行聚类 |
| **细胞-空间映射** | 将 scRNA-seq 细胞映射到空间位置 | Encoder_map |
| **空间可变基因** | 识别具有空间模式的基因 | 利用嵌入分析 |
| **批次校正** | 整合多个切片数据 | 对齐嵌入空间 |
| **轨迹推断** | 推断组织发育/疾病进展 | 嵌入 + 伪时间分析 |

### 6.2 空间域分割

```python
# GraphST 的使用示例
import GraphST

# 1. 初始化模型
model = GraphST.GraphST(adata)

# 2. 构建图
model.build_graph(radius=200)  # 半径 200μm

# 3. 预训练 (DGI)
model.train()

# 4. 获取嵌入
embeddings = model.get_embedding()

# 5. 聚类 → 空间域
adata.obs['domain'] = model.cluster(n_clusters=10)
```

---

## 7. 代码结构速览

```
GraphST/
├── GraphST/                    # 核心代码包
│   ├── __init__.py
│   ├── GraphST.py              # 主类: 训练 + 预测
│   ├── model.py                # 模型定义 (221行)
│   │   ├── Encoder             # 图编码器 (普通/稀疏)
│   │   ├── Discriminator       # Bilinear 判别器
│   │   ├── AvgReadout          # 加权平均池化
│   │   ├── Encoder_sc          # 单细胞自编码器
│   │   └── Encoder_map         # 细胞-spot 映射
│   ├── preprocess.py           # 数据预处理
│   └── utils.py                # 工具函数
│
├── Data/                       # 示例数据
├── setup.py                    # 安装配置
└── README.md
```

---

## 8. 关键概念 Q&A

### Q1: GraphST 和 Novae 的区别？

| 方面 | Novae | GraphST |
|------|-------|---------|
| **预训练方法** | SwAV (自分配对比) | DGI (图-补丁对比) |
| **图卷积** | GAT v2 (注意力) | GCN (简单图卷积) |
| **负样本** | 无显式负样本 | 特征打乱作为负样本 |
| **预训练目标** | 原型分配一致性 | 最大化互信息 |
| **输出粒度** | 细胞/spot 嵌入 | 图嵌入 + 局部嵌入 |
| **扩展模块** | 无 | Encoder_sc, Encoder_map |

### Q2: 什么是 Deep Graph Infomax (DGI)？

DGI 是一种图级别的自监督学习方法：

```
核心思想：让局部节点嵌入能"预测"整图嵌入
  → 如果 hᵢ 包含足够多的全局信息，D(hᵢ, s) 分数高
  
数学原理：最大化互信息 I(hᵢ; s)
  → 通过 Jensen-Shannon 散度的变分下界优化
  
实现方式：
  正样本: (原始图节点, 整图)
  负样本: (特征打乱后的节点, 整图)
  判别器: Bilinear
```

### Q3: Encoder_sc 和 Encoder_map 的具体用途？

- **Encoder_sc**：当空间数据分辨率较低时（如 Visium 每个 spot 包含多个细胞），用自编码器从混合信号中分解出单细胞级别的信息
- **Encoder_map**：当有独立测序的 scRNA-seq 数据时，学习一个映射矩阵 M，推断每种细胞类型在空间中的分布

### Q4: 什么是 Bilinear 判别器？

```python
# 公式: D(hᵢ, s) = hᵢᵀ · W · s
# W 是可学习的双线性变换矩阵

# 相比简单的内积或余弦相似度：
# - 内积: D = hᵢᵀ · s (无参数)
# - Bilinear: D = hᵢᵀ · W · s (可学习)
# 
# W 可以学习到不同的"对齐方式"
```

---

## 9. 延伸阅读

### 核心论文

1. **Y Long et al.** *GraphST: Graph-based Self-supervised Learning for Spatial Transcriptomics.* Nature Communications, 2023. [阅读](https://www.nature.com/articles/s41467-023-40173-4)

### 相关技术

- **Deep Graph Infomax**：[Deep Graph Infomax](https://arxiv.org/abs/1809.10341), ICLR 2019
- **GCN**：[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

### 学习路线

```
空间转录组模型对比:

1. 基于序列 (Transformer):
   └── NicheFormer

2. 基于图 (GNN):
   ├── Novae   ← SwAV (无负样本, 有原型)
   ├── GraphST ← DGI  (有负样本, 无原型) ← 这一篇
   ├── SpaceFlow ← DGI + 空间正则化
   └── SPADE    ← 多模态对比 (ST + H&E)
```

---

> 💡 **学习建议**：GraphST 是 DGI 在空间转录组上的直接应用，架构简洁清晰。建议集中理解 DGI 的训练机制（正负样本构造 + Bilinear 判别器），再关注 Encoder_sc 和 Encoder_map 两个扩展模块的设计动机。
