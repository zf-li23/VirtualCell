# Novae 学习笔记

> Novae 是一个基于**图的视觉自监督学习 (SwAV on Graphs)** 的空间转录组基础模型，通过将基因表达构建为细胞图并在图上进行 SwAV 自监督学习，实现空间域的无监督分割和细胞微环境分析。

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
| **论文** | [Novae: A Graph-based Foundation Model for Spatial Transcriptomics](https://www.nature.com/articles/s41592-025-02899-6), Nature 2025 |
| **架构** | 图神经网络 (GNN) — GAT v2 + SwAV Head |
| **预训练任务** | SwAV 自监督学习 (Swapping Assignments between Views) |
| **输入** | 空间转录组细胞图 (节点 = 细胞/spot, 边 = 空间邻接) |
| **输出** | 细胞/spot 嵌入 + 空间域分割 |
| **图增强** | 基因子集采样 + 噪声 + 丢失 |
| **项目** | [novae](https://github.com/prism-oncology/novae) |

### 核心思想

> "将空间转录组数据建模为**图结构**——节点是细胞/spot，边表示空间邻近关系——然后通过图神经网络和 SwAV 自监督学习，让模型自动学习组织中的**空间域 (spatial domains)**，无需任何人工标注。"

---

## 2. 模型架构

### 2.1 整体架构

```
原始空间转录组数据 (基因表达 + 空间坐标)
        │
        ▼
┌───────────────────────────┐
│       CellEmbedder        │  ← 基因嵌入矩阵 → 线性投影 → 归一化
│   基因表达 → 节点特征      │     x @ F.normalize(genes_embeddings)
└───────────┬───────────────┘
            │
┌───────────┴───────────────┐
│    GraphAugmentation       │  ← 训练时数据增强
│   - Panel Subset (子集采样) │     panel_subset + noise + dropout
│   - Background Noise       │
│   - Sensitivity Noise      │
└───────────┬───────────────┘
            │
┌───────────┴───────────────┐
│       GraphEncoder         │  ← GAT v2 (边特征维度=1)
│   - GAT v2 Convolution     │     + AttentionAggregation
│   - Attention Aggregation  │
└───────────┬───────────────┘
            │
┌───────────┴───────────────┐
│        SwavHead            │  ← SwAV 自监督学习
│   - 原型投影 (Prototypes)   │     Sinkhorn-Knopp 算法
│   - Sinkhorn-Knopp 算法    │     队列式原型选择
│   - 多裁剪 (Multi-crop)    │
└───────────┬───────────────┘
            │
    细胞/spot 嵌入 + 空间域标签
```

### 2.2 CellEmbedder (细胞嵌入器)

```python
# embed.py
class CellEmbedder(L.LightningModule):
    """
    将基因表达向量 → 可学习的细胞嵌入
    
    核心公式：
        embedding = x @ F.normalize(genes_embeddings)
    
    其中：
    - x:        原始基因表达向量 [n_genes]
    - genes_embeddings: 可学习的基因嵌入矩阵 [n_genes, d_model]
    - @:        矩阵乘法 (加权求和)
    """
    def __init__(self, n_genes, d_model):
        # 可学习的基因嵌入矩阵
        self.genes_embeddings = nn.Embedding(n_genes, d_model)
        # 线性投影 (初始化为单位矩阵)
        self.linear = nn.Linear(n_genes, d_model, bias=False)
        nn.init.eye_(self.linear.weight)  # 初始化后 = 恒等映射
    
    def forward(self, data):
        # 归一化的基因嵌入
        norm_emb = F.normalize(self.genes_embeddings.weight, dim=1)
        # 细胞表达 × 基因嵌入 → 细胞特征
        data.x = data.x @ norm_emb  # [n_cells, d_model]
        return data
```

**关键设计**：每个基因学习一个嵌入向量，细胞的表达值通过矩阵乘法与基因嵌入组合，得到细胞特征。

#### PCA 初始化

```python
# CellEmbedder 支持加载预计算的 PCA 结果作为基因嵌入的初始化
# 类似于 scGPT 的 gene_embeddings 初始化思路
```

### 2.3 GraphEncoder (图编码器)

```python
# encode.py
class GraphEncoder(L.LightningModule):
    """
    图神经网络编码器：GAT v2 + Attention Aggregation
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=4):
        # GAT v2 卷积层 (支持边特征)
        self.convs = nn.ModuleList([
            GATv2Conv(in_dim if i==0 else hidden_dim, 
                      hidden_dim, edge_dim=1)
            for i in range(n_layers)
        ])
        
        # 注意力池化
        self.pool = AttentionAggregation(hidden_dim, out_dim)
        
        # 可选：组织学特征融合
        self.mlp_fusion = nn.Sequential(
            nn.Linear(out_dim + histo_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
```

**GAT v2** 相比 GAT v1 的改进：
- 注意力计算改为：`a(LeakyReLU(W·[h_i || h_j || e_ij]))`
- 边特征 (edge_dim=1) 编码空间距离或权重
- ELU 激活函数

### 2.4 AttentionAggregation (注意力池化)

```python
# aggregate.py
class AttentionAggregation(nn.Module):
    """
    门控注意力池化：对每个节点学习一个注意力权重，加权求和
    
    gate_nn:   线性层 → 1 → softmax → 注意力权重 (用于加权)
    nn:        线性层 → out_dim → 值变换
    """
    def __init__(self, in_dim, out_dim):
        self.gate_nn = nn.Linear(in_dim, 1)    # 注意力分数
        self.nn = nn.Linear(in_dim, out_dim)   # 值变换
    
    def forward(self, x, batch):
        # x: [n_nodes, in_dim]
        # batch: 节点属于哪个图的索引
        
        # 计算注意力分数
        gate = torch.softmax(self.gate_nn(x), dim=0)
        
        # 值变换 + 加权求和
        out = self.nn(x) * gate
        out = scatter_sum(out, batch, dim=0)  # 按图聚合
        
        return out
```

---

## 3. 核心创新

### 3.1 SwAV 自监督学习在空间转录组图上的应用

SwAV (Swapping Assignments between Views) 原是计算机视觉中的自监督学习方法。Novae 将其创新性地应用到空间转录组图上：

```
图增强视角 1 (View 1):
    ┌─────────────────┐
    │  原始图的一部分   │  ← Panel Subset + Noise
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  GraphEncoder   │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  Prototype      │  ← 分配到 K 个原型 (prototype)
    │  Assignment q1  │     Sinkhorn-Knopp 算法
    └────────┬────────┘
             │
             │ 交换分配 ("Swap"): q1 → p2, q2 → p1
             │
    ┌────────┴────────┐
    │  交叉熵损失      │  ← CrossEntropy(q1, p2) + CrossEntropy(q2, p1)
    └─────────────────┘

图增强视角 2 (View 2):
    ┌─────────────────┐
    │  原始图的不同部分  │  ← 不同的 Panel Subset + Noise
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  GraphEncoder   │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  Prototype      │
    │  Assignment q2  │
    └─────────────────┘
```

**交换分配的核心思想**：对同一个图的两个增强视角，要求各自视角的预测（p）与另一视角的分配（q）一致。

### 3.2 Sinkhorn-Knopp 算法

```python
# swav.py
def sinkhorn_knopp(Q, num_iters=3, epsilon=0.05):
    """
    Sinkhorn-Knopp 迭代归一化算法
    
    输入: Q [batch_size, n_prototypes] 相似度矩阵
    输出: 均匀分配后的概率矩阵
    
    步骤:
    1. 对 Q 应用 exp(Q / epsilon)
    2. 迭代的行归一化 + 列归一化
    3. 最终得到近似均匀的分配
    """
    Q = torch.exp(Q / epsilon)
    for _ in range(num_iters):
        Q = Q / Q.sum(dim=1, keepdim=True)  # 行归一化
        Q = Q / Q.sum(dim=0, keepdim=True)  # 列归一化
    return Q / Q.shape[0]  # 总归一化
```

**为什么需要 Sinkhorn-Knopp？**
- 防止模型将所有样本分配到同一个原型（崩溃）
- 强制分配的熵最大化（均匀分布在整个原型集合上）
- 只需 3 次迭代就能收敛

### 3.3 队列式原型选择

```python
# swav.py
# 对每个 slide 分别维护一个队列
# 在 Sinkhorn-Knopp 分配时，使用队列中的样本计算全局统计量
# 避免大 batch 需求，提高训练稳定性
```

### 3.4 可解释的空间域分割

Novae 不仅生成细胞嵌入，还可以直接进行空间域分割：

```python
# 零样本域分割
def predict_domains(embeddings, n_domains):
    # 对原型进行层次聚类
    # 使用 AgglomerativeClustering (cosine distance)
    # 将原型合并为 n_domains 个空间域

# 原型到域的映射
# Prototype 1, 2, 3 → Domain A (如 "肿瘤区域")
# Prototype 4, 5   → Domain B (如 "基质区域")
```

### 3.5 图增强策略

Novae 的空间转录组数据增强是专门设计的：

| 增强类型 | 说明 | 效果 |
|----------|------|------|
| **Panel Subset** | 随机保留一部分基因 | 模拟不同基因子集，增强鲁棒性 |
| **Background Noise** | 添加指数分布噪声 | 模拟背景信号的随机性 |
| **Sensitivity Noise** | 添加乘性高斯噪声 | 模拟检测灵敏度的变化 |
| **Gene Dropout** | 将某些基因设为零 | 模拟基因丢失事件 |

---

## 4. 图数据构建

### 4.1 从空间坐标到图

```
空间坐标 (x, y)
    │
    ▼
┌─────────────────────────┐
│  KNN 构建邻接图          │
│  每个节点 → k 个近邻     │
└───────────┬─────────────┘
            │
            ▼
    图: G = (V, E)
    V: 节点 = 细胞/spot
    E: 边 = 空间邻近关系
    边特征: 空间距离 (或权重)
```

### 4.2 节点特征

```
节点特征 = CellEmbedder 输出的细胞嵌入
    = 原始表达向量 @ 基因嵌入矩阵
```

### 4.3 图批处理

Novae 支持对整个组织切片进行图级别的批处理：

```
每个组织切片 = 一个独立的图
多个切片 = 多个图组成的 batch

PyG (PyTorch Geometric) 的 Batch 对象：
    data.x:     所有节点特征拼接
    data.edge_index: 所有边拼接 (偏移量自动处理)
    data.batch: 每个节点属于哪个图的索引
```

---

## 5. Tokenization 与输入编码

Novae 使用 **图结构** 而非序列结构，其"Tokenization"与传统模型有本质不同：

### 5.1 CellEmbedder（节点特征编码）

```python
# 基因表达 → 可学习细胞嵌入
gene_embeddings = nn.Parameter(torch.randn(n_genes, d_model))
# 线性投影: x @ gene_embeddings.T
h = x @ F.normalize(gene_embeddings)  # → 节点特征
```

### 5.2 图结构中的"Token"

| 维度 | 序列模型 | 图模型 (Novae) |
|------|---------|---------------|
| Token | 单个基因 | 整个细胞/spot |
| 上下文 | 线性基因顺序 | 空间邻接图 |
| 位置编码 | 位置嵌入 | 图拓扑结构 |

### 5.3 图增强（训练时的数据增强）

```python
# 训练时对图进行增强
# 1. Panel Subset: 随机采样部分基因
# 2. Background Noise: 添加背景噪声
# 3. Sensitivity Noise: 灵敏度噪声
view1 = augment(graph, subset_ratio=0.8, noise_level=0.1)
view2 = augment(graph, subset_ratio=0.8, noise_level=0.1)
```

---

## 6. 预训练

### 5.1 预训练流程

```python
def training_step(self, batch):
    # 1. 两个增强视角
    view1 = self.augment(batch)  # 图增强
    view2 = self.augment(batch)  # 不同的增强
    
    # 2. 编码
    z1 = self.encoder(view1)     # [n_cells, d_model]
    z2 = self.encoder(view2)
    
    # 3. 投影到原型空间
    p1 = self.swav_head(z1)      # [n_cells, n_prototypes]
    p2 = self.swav_head(z2)
    
    # 4. Sinkhorn-Knopp 分配
    q1 = sinkhorn_knopp(p1)
    q2 = sinkhorn_knopp(p2)
    
    # 5. 交换分配损失
    loss = cross_entropy(q1, p2) + cross_entropy(q2, p1)
    
    return loss
```

### 5.2 原型数量

| 超参数 | 值 | 说明 |
|--------|-----|------|
| n_prototypes | 3000 ~ 5000 | 原型数量，越多越能捕捉细粒度模式 |
| temperature | 0.1 | Softmax 温度 |
| sinkhorn_eps | 0.05 | Sinkhorn-Knopp 的 epsilon 参数 |

### 5.3 零样本域分割

```python
def map_leaves_domains(prototypes, n_domains):
    """
    对原型进行层次聚类，映射到空间域
    
    1. 计算所有原型的嵌入
    2. AgglomerativeClustering(cosine) 聚为 n_domains 类
    3. 每个原型分配到对应的域
    4. 新样本通过其最近的原型分配到域
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_domains,
        metric='cosine',
        linkage='average'
    )
    domain_labels = clustering.fit_predict(prototypes)
    return domain_labels
```

---

## 7. 下游任务

### 6.1 支持的任务

| 任务 | 类型 | 说明 |
|------|------|------|
| **空间域分割** | 无监督 / 零样本 | 自动识别组织区域 (如肿瘤、基质、免疫) |
| **细胞嵌入提取** | 表示学习 | 可用于下游分类或可视化 |
| **空间可变基因检测** | 分析 | 识别具有空间模式的基因 |
| **微环境聚类** | 无监督 | 识别重复出现的细胞邻域类型 |

### 6.2 空间域分割示例

```python
import novae

# 加载预训练模型
model = novae.Novae.from_pretrained("novae")

# 对切片数据编码
embeddings = model.encode(adata)

# 预测空间域 (零样本)
adata.obs["domain"] = model.predict_domains(
    adata, n_domains=10  # 指定分割成 10 个区域
)

# 可视化
import scanpy as sc
sc.pl.spatial(adata, color="domain")
```

---

## 8. 代码结构速览

```
novae/
├── novae/
│   ├── module/
│   │   ├── embed.py          # CellEmbedder (179行)
│   │   │   └── 基因嵌入矩阵 → 细胞特征
│   │   ├── encode.py         # GraphEncoder (67行)
│   │   │   ├── GAT v2 卷积层
│   │   │   └── AttentionAggregation + MLP 融合
│   │   ├── augment.py        # GraphAugmentation (87行)
│   │   │   ├── Panel Subset
│   │   │   ├── 指数噪声 + 乘性高斯噪声
│   │   │   └── 基因丢失
│   │   ├── swav.py           # SwavHead (281行)
│   │   │   ├── 原型投影
│   │   │   ├── Sinkhorn-Knopp 算法
│   │   │   ├── 队列式选择
│   │   │   └── 层次聚类域映射
│   │   └── aggregate.py      # AttentionAggregation (61行)
│   │       └── 门控注意力池化
│   │
│   ├── data/                  # 数据加载
│   │── scripts/               # 训练脚本
│   └── tests/                 # 单元测试
│
├── docs/                      # 文档
├── docker/                    # Docker 配置
├── pyproject.toml             # 项目配置
└── README.md
```

---

## 9. 关键概念 Q&A

### Q1: Novae 与 NicheFormer 的区别？

| 方面 | NicheFormer | Novae |
|------|------------|-------|
| **架构** | Transformer (序列) | GNN (图) |
| **预训练** | MLM (掩码预测) | SwAV (对比学习) |
| **空间表示** | 邻居上下文序列 | 图结构上的消息传递 |
| **数据模式** | 单细胞 + 空间 | 仅空间 |
| **是否支持 H&E** | 否 | 是 (通过 MLP 融合) |
| **域分割** | 需要微调 | 零样本 |

### Q2: SwAV 和对比学习的区别？

```
对比学习 (如 SimCLR):
    正样本对: 相同图的不同增强视图
    负样本对: 不同图的增强视图
    损失: InfoNCE (需要显式负样本)

SwAV (Swapping Assignments):
    正样本对: 相同图的不同增强视图
    无显式负样本
    损失: 交换分配 (CrossEntropy between assignments)
    通过 Sinkhorn-Knopp 保证分配的多样性
```

SwAV 的优势：不需要构造负样本对，计算效率更高。

### Q3: 基因嵌入矩阵 (genes_embeddings) 的作用？

- 本质是每个基因学习一个 d_model 维的嵌入向量
- 细胞表达 = 基因表达值对基因嵌入的**加权求和**
- 类似于 NLP 中的词嵌入 (word embeddings)
- 可以加载预训练嵌入（如 scGPT 的 gene embeddings）

### Q4: GAT v2 相比 GAT v1 的改进？

```
GAT v1: a(LeakyReLU(W·h_i || W·h_j))
GAT v2: a(LeakyReLU(W·[h_i || h_j || e_ij]))

改进：
1. 边特征 (e_ij) 参与注意力计算
2. 更通用的表达形式 (GAT v1 是 GAT v2 的特例)
3. 更好的表示能力
```

### Q5: 为什么需要图增强 (Graph Augmentation)？

与 CV 中图像增强的原理相同：
- **Panel Subset** → 模拟不同基因检测面板，增强泛化性
- **噪声** → 提高鲁棒性，防止过拟合
- **丢失** → 模拟实际数据中的缺失值
- 综合效果：让模型学到**基因表达模式的本质**而非噪声

---

## 10. 延伸阅读

### 相关技术

- **SwAV**：[Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882), NeurIPS 2020
- **Sinkhorn-Knopp**：[Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895)
- **GAT v2**：[How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491)
- **PyTorch Geometric**：[PyG Documentation](https://pytorch-geometric.readthedocs.io)

### 技术路线对比

```
空间转录组模型分类:

1. 基于序列 → Transformer:
   └── NicheFormer ← 单细胞 + 空间

2. 基于图 → GNN:
   └── Novae      ← SwAV 自监督 ← 这一篇
```

---

> 💡 **学习建议**：Novae 是空间转录组图模型的代表。建议先理解 SwAV 自监督学习的基本原理（尤其是 Sinkhorn-Knopp 算法），再关注它如何将"图"作为数据的天然表示方式。与 NicheFormer 的序列方法对比，有助于理解两种空间建模策略的优劣。
