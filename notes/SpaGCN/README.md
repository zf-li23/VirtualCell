# SpaGCN 学习笔记 🧬

## 概述

**SpaGCN** (Spatial Graph Convolutional Network) 是一种基于图卷积网络的空间转录组学方法，由 Jian Hu 等人开发（发表于 *Nature Methods*, 2021）。其核心创新在于**同时整合基因表达、空间位置和组织学（H&E 染色图像）信息**，通过构建无向加权图并运用 GCN + 深度嵌入聚类算法来识别空间域（spatial domains）和空间可变基因（SVGs）。

与 GraphST、SpaceFlow 等同为 GNN-based 方法不同的是，SpaGCN 独特地引入了**组织学特征**（通过 H&E 图像的 RGB 像素强度），将组织形态学信息融入到空间域识别中。

---

## 核心架构

### 1. 整体流程

SpaGCN 的工作流程分为三个核心步骤：

```
原始数据 (基因表达 + 空间坐标 + H&E 图像)
        │
        ▼
   步骤 1: 构建空间邻接图 (Adjacency Matrix)
        │   - 基于空间距离 + H&E 颜色相似度
        │   - 计算加权邻接矩阵 adj
        │
        ▼
   步骤 2: 图卷积 + 自优化聚类
        │   - GCN 聚合邻域表达信息
        │   - DEC (Deep Embedding Clustering) 迭代优化
        │
        ▼
   步骤 3: 空间域识别 & SVGs 分析
        │   - 输出每个 spot 的聚类标签
        │   - 差异表达分析找富集基因
        │
        ▼
   空间域可视化 & 生物学解释
```

### 2. 模型架构

SpaGCN 的深度学习模型是一个**简化的图卷积自编码器 + 深度嵌入聚类**架构：

```
┌─────────────────────────────────────────────────┐
│            simple_GC_DEC 模型                     │
├─────────────────────────────────────────────────┤
│                                                   │
│   输入: PCA 降维后的基因表达 (nfeat)              │
│        + 空间加权邻接矩阵 (adj_exp)               │
│                                                   │
│   ┌─────────────┐                                │
│   │ GraphConv   │  (nfeat → nhid)                │
│   │ W·X @ A    │  单层 GCN：AGGREGATE + COMBINE │
│   └─────┬───────┘                                │
│         ▼                                        │
│   ┌─────────────┐                                │
│   │ Student's t │  计算嵌入到聚类中心的距离       │
│   │  分布 Q     │  q_ij = softmax(1/(1+||z_i-μ_j||²))│
│   └─────┬───────┘                                │
│         ▼                                        │
│   ┌─────────────┐                                │
│   │  Target P   │  辅助目标分布（高置信度提升）   │
│   └─────┬───────┘                                │
│         ▼                                        │
│   Loss = KL(P || Q)  ← 自优化聚类损失             │
└─────────────────────────────────────────────────┘
```

#### GraphConvolution 层

这是最基础的 GCN 层，直接实现了 Kipf & Welling 的图卷积:

```
forward(input, adj):
    support = W @ input          # 线性变换
    output = adj @ support       # 邻接矩阵聚合（spmm）
    return output + bias         # 加偏置
```

特点：
- 简化的实现，没有使用归一化的拉普拉斯矩阵
- 使用 `torch.spmm` 进行稀疏矩阵乘法
- 权重初始化采用均匀分布 `U(-1/sqrt(d_in), 1/sqrt(d_in))`

#### 深度嵌入聚类 (DEC)

受 [Xie et al., ICML 2016](https://arxiv.org/abs/1511.06335) 启发，SpaGCN 使用 DEC 进行无监督聚类：

1. **初始化聚类中心**：通过 Louvain 或 K-means 获得初始聚类标签
2. **软分配**：使用 Student's t 分布计算每个 spot 到各聚类中心的软分配概率 q
3. **目标分布**：计算辅助目标分布 p（提升高置信度分配，压低低置信度）
4. **KL 散度优化**：最小化 P 和 Q 之间的 KL 散度

```python
# 软分配公式
q = 1.0 / ((1.0 + ||z_i - μ_j||² / α) + 1e-8)
q = q^((α+1)/2) / sum(q)  # softmax 归一化

# 目标分布
p = q² / sum(q, dim=0)
p = p / sum(p, dim=1, keepdim=True)

# 损失
loss = mean(sum(p * log(p / q), dim=1))
```

早停机制：当标签变化比例低于 `tol` 阈值时停止训练。

---

## 关键技术细节

### 3.1 空间邻接图构建

这是 SpaGCN 最独特的部分——融合了组织学信息：

```python
def calculate_adj_matrix(x, y, x_pixel, y_pixel, image, beta=49, alpha=1, histology=True):
    if histology:
        # 1. 提取每个 spot 周围 beta×beta 区域的 H&E 颜色均值
        g = [mean(mean(image[spot-beta/2:spot+beta/2])) for each spot]
        
        # 2. 将 RGB 三通道加权为单色值 c3
        #    权重=各通道方差（高方差通道贡献更多信息）
        c0, c1, c2 = R, G, B 通道均值
        c3 = (c0*var(c0) + c1*var(c1) + c2*var(c2)) / (var(c0)+var(c1)+var(c2))
        
        # 3. 标准化后缩放到与空间坐标相同尺度
        c4 = (c3 - mean(c3)) / std(c3)
        z = c4 * max(std(x), std(y)) * alpha
        
        # 4. 构建三维坐标: X = [x, y, z]
    else:
        X = [x, y]
    
    # 5. 计算所有 spot 两两之间的欧氏距离
    return pairwise_distance(X)
```

**关键参数**：
- `beta`：控制提取 H&E 颜色的邻域范围（默认 49）
- `alpha`：控制颜色信息的权重（默认 1.0）
- `histology=True`：启用组织学信息融合

### 3.2 高斯核邻接权重

构建好距离矩阵后，使用高斯核将其转换为边权重：

```python
adj_exp = exp(-adj² / (2 * l²))
```

其中 `l` 是一个关键的超参数，控制高斯核的带宽。SpaGCN 通过二分搜索自动寻找最优 `l`：

```python
def search_l(p, adj, start=0.01, end=1000, tol=0.01):
    # p 是期望的"邻域贡献比例"
    # calculate_p(adj, l) = mean(sum(adj_exp, axis=1)) - 1
    # 即每个 spot 从邻域接收的平均权重之和
    # 通过二分搜索找到满足 |calculate_p(adj, l) - p| < tol 的 l
```

### 3.3 多切片联合分析 (multiSpaGCN)

SpaGCN 支持同时分析多个组织切片：

```python
class multiSpaGCN:
    def train(self, adata_list, adj_list, l_list):
        # 1. 对每个切片独立计算邻接矩阵
        for i in range(len(l_list)):
            adj_exp = exp(-adj² / (2 * l²))
            adj_exp_all[start:start+n, start:start+n] = adj_exp
        
        # 2. 拼接所有切片的表达数据
        adata_all = AnnData.concatenate(*adata_list, join='inner')
        
        # 3. PCA 降维 + 统一模型训练
        # 使用 same simple_GC_DEC model
```

### 3.4 聚类后处理：空间细化 (Refine)

SpaGCN 提供了基于空间邻域的聚类结果细化方法：

```python
def refine(sample_id, pred, dis, shape="hexagon"):
    # 对每个 spot，检查其最近邻的 num_nbs 个 spots
    # 如果当前 spot 的类别在邻域中占比 < num_nbs/2
    # 且存在某个其他类别占比 > num_nbs/2
    # 则将该 spot 重新分配给该多数类别
    
    # shape="hexagon": num_nbs=6 (适用于 Visium)
    # shape="square": num_nbs=4 (适用于 ST)
```

这是一种基于"空间平滑性假设"的后处理，假设相邻 spots 应该属于同一空间域。

---

## 下游分析能力

### 空间可变基因 (SVGs) 识别

SpaGCN 识别出空间域后，对每个域进行差异表达分析：

```python
def rank_genes_groups(input_adata, target_cluster, nbr_list, label_col):
    # 1. 对比目标域 vs 邻近域（而非全组织）
    # 2. Wilcoxon 秩和检验
    # 3. 计算 fold change、in-group/out-group fraction
    # 4. 输出 p_val_adj、fold_change、fraction 等统计量
```

特点：对比组只包含目标域及其空间邻近域，而非全组织，这能更精确地捕获域特异性基因。

### Meta Gene 构建

SpaGCN 可以迭代构建"元基因"——通过加减特定基因来表达特定空间域的特征：

```python
def find_meta_gene(input_adata, pred, target_domain, start_gene):
    # 迭代过程：
    # 1. 在目标域 vs 非目标域之间做差异表达分析
    # 2. 找到上调最显著的基因 (add_g) 和下调最显著的基因 (adj_g)
    # 3. 更新 meta_gene = prev_meta + add_g - adj_g
    # 4. 直到早停条件触发（非目标 spots 数不再减少或均值差异不再增大）
```

---

## 代码结构

| 文件 | 功能 |
|------|------|
| `SpaGCN.py` | 主类 `SpaGCN` 和 `multiSpaGCN`，封装训练和预测流程 |
| `models.py` | `simple_GC_DEC`（单层 GCN + DEC）和 `GC_DEC`（双层 GCN + DEC）模型 |
| `layers.py` | `GraphConvolution` 层实现 |
| `calculate_adj.py` | 邻接矩阵构建（含组织学信息融合） |
| `util.py` | 预处理、参数搜索、差异表达分析、细化、meta gene 等工具函数 |
| `ez_mode.py` | 简化版使用接口，一键执行完整流程 |
| `calculate_moran_I.py` | Moran's I 空间自相关统计量计算 |

---

## 使用方法

### 标准模式

```python
import SpaGCN

# 1. 计算邻接矩阵（融合组织学）
adj = calculate_adj_matrix(x=x_pixel, y=y_pixel, 
                          image=img, beta=49, alpha=1, histology=True)

# 2. 搜索最佳 l 值
l = search_l(p=0.5, adj=adj, start=0.01, end=1000)

# 3. 搜索最佳聚类分辨率
res = search_res(adata, adj, l, target_num=7, start=0.4, step=0.1)

# 4. 训练 SpaGCN
clf = SpaGCN()
clf.set_l(l)
clf.train(adata, adj, init_spa=True, init="louvain", res=res, 
          tol=5e-3, lr=0.05, max_epochs=200)

# 5. 预测
y_pred, prob = clf.predict()

# 6. 细化
refined_pred = refine(sample_id=adata.obs.index.tolist(), 
                      pred=y_pred, dis=adj_2d, shape="hexagon")
```

### EZ Mode（简化版）

```python
from SpaGCN import detect_spatial_domains_ez_mode

y_pred = detect_spatial_domains_ez_mode(
    adata=adata, img=img,
    x_array=x_array, y_array=y_array,
    x_pixel=x_pixel, y_pixel=y_pixel,
    n_clusters=7, histology=True, s=1, b=49, p=0.5
)
```

---

## 与同类方法的比较

| 特性 | SpaGCN | GraphST | SpaceFlow |
|------|--------|---------|-----------|
| 学习范式 | GCN + DEC 自优化聚类 | DGI 对比学习 | DGI + 空间正则化 |
| 组织学融合 | ✅ H&E 颜色直方图 | ❌ 不融合 | ❌ 不融合 |
| 图构建 | 全连接加权图（高斯核） | 邻域图（KNN） | Alpha Complex |
| 聚类方法 | 端到端 DEC 聚类 | GMM 或 Leiden | Leiden |
| 多切片支持 | ✅ multiSpaGCN | ✅ 对齐后联合分析 | ❌ |
| 输出 | 空间域 + SVGs + Meta gene | 空间域 + 增强表达 | pSM + 空间域 |

---

## 优缺点

**优点**：
- ✅ 组织学信息融合是其独特优势，尤其对 H&E 染色质量高的数据
- ✅ 端到端的聚类框架，无需额外的聚类步骤
- ✅ 提供 SVGs 识别和 Meta gene 构建的完整分析链路
- ✅ 支持多切片联合分析
- ✅ EZ Mode 降低了使用门槛

**局限性**：
- ❌ 单层 GCN 的表达能力有限，难以捕获复杂的非线性关系
- ❌ 全连接邻接矩阵的计算复杂度为 O(n²)，大规模数据时内存开销大
- ❌ 需要手动指定期望的聚类数（通过 target_num 搜索 res）
- ❌ 对 l 参数敏感，需要仔细搜索
- ❌ 没有显式的批次校正机制

---

## 关键参数速查

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `beta` | 49 | H&E 颜色提取邻域范围 |
| `alpha` | 1.0 | 颜色信息权重 |
| `p` | 0.5 | 期望的邻域贡献比例 |
| `l` | 自动搜索 | 高斯核带宽 |
| `res` | 自动搜索 | Louvain 分辨率 |
| `num_pcs` | 50 | PCA 降维维数 |
| `lr` | 0.005 | 学习率 |
| `max_epochs` | 2000 | 最大训练轮数 |
| `tol` | 1e-3 | 早停容忍度 |

---

## 参考资源

- 论文：Hu et al., *Nature Methods*, 2021. [DOI: 10.1038/s41592-021-01255-8](https://www.nature.com/articles/s41592-021-01255-8)
- GitHub: [https://github.com/jianhuupenn/SpaGCN](https://github.com/jianhuupenn/SpaGCN)
- 原始 DEC 论文：Xie et al., "Unsupervised Deep Embedding for Clustering Analysis", ICML 2016
- GCN 基础：Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
