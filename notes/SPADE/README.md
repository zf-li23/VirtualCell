# SPADE 学习笔记

> SPADE (Spatially-Aware Domain Expert) 是一个基于 **Mixture-of-Data-Experts** 和多模态对比学习的空间转录组分析方法，通过将空间转录组 (ST) 与 H&E 病理图像对齐，同时利用**多个 ImageEncoder 集成**实现鲁棒的空间域分割。

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理](#4-数据预处理)
5. [训练](#5-训练)
6. [下游任务](#6-下游任务)
7. [代码结构速览](#7-代码结构速览)
8. [关键概念 Q&A](#8-关键概念-qa)
9. [延伸阅读](#9-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [SPADE: Spatial Transcriptomics and Pathology Alignment Using a Mixture of Data Experts for an Expressive Latent Space](https://arxiv.org/abs/2506.21857v1), arXiv 2025 |
| **架构** | CLIP-style 对比学习 (Image + Spot Embedding) + Mixture-of-Experts |
| **训练任务** | ST-H&E 对齐 (对称交叉熵 + 软目标) |
| **图像编码器** | 多个 (ResNet50/101/152, ViT, CLIP-ViT) — 可配置 |
| **Spot 编码器** | ProjectionHead (MLP) |
| **核心方法** | 多专家集成 (Data Experts), 软标签交叉熵 |

### 核心思想

> "空间转录组的每个 spot 有两种模态数据：**基因表达**和对应的**H&E 组织学图像**。SPADE 通过 CLIP 风格的对比学习对齐这两种模态，并利用多个 ImageEncoder（不同架构、不同预训练数据）作为**数据专家**进行集成，弥补单模型的知识局限。"

---

## 2. 模型架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     双模态对比学习架构                        │
│                                                             │
│  ST 模态 (基因表达)           H&E 模态 (组织学图像)           │
│                                                             │
│  Spot 表达向量                  H&E 图像补丁                  │
│       │                            │                        │
│       ▼                            ▼                        │
│  ┌──────────┐               ┌──────────┐                   │
│  │ Projection│              │ Image    │ ← 多个可用         │
│  │ Head     │               │ Encoder  │   - ResNet50/101   │
│  │ (MLP)    │               │ (timm)   │   - ViT / ViT-L   │
│  └────┬─────┘               │          │   - CLIP-ViT      │
│       │                     └────┬─────┘                   │
│       ▼                          ▼                          │
│  ┌──────────┐               ┌──────────┐                   │
│  │ Spot Emb │               │ Image Emb│                   │
│  │ (d=256)  │               │ (d=256)  │                   │
│  └────┬─────┘               └────┬─────┘                   │
│       │                          │                          │
│       └──────────┬───────────────┘                          │
│                  │                                          │
│                  ▼                                          │
│         ┌────────────────┐                                  │
│         │  对比学习损失   │  ← 对称交叉熵 + 软目标         │
│         │  (CLIP-style)  │     (soft targets)              │
│         └────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ImageEncoder (图像编码器)

SPADE 支持多种图像编码器架构，通过 `timm` 库加载：

```python
# modules.py
class ImageEncoder(nn.Module):
    """
    支持多种 CNN/ViT 骨干网络
    通过 timm 库加载预训练权重
    """
    def __init__(self, model_name="resnet50", freeze_backbone=True):
        # 可选模型:
        # - resnet50, resnet101, resnet152
        # - vit_base_patch16_224, vit_large_patch16_224
        # - clip_vit_base_patch16_224 (OpenAI CLIP)
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # 移除分类头
        )
        
        # 冻结骨干网络 (可选)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
```

**支持的编码器架构**：

| 模型 | 参数量 | 输出维度 | 预训练数据 | 特点 |
|------|--------|---------|-----------|------|
| ResNet50 | 25M | 2048 | ImageNet | 经典 CNN |
| ResNet101 | 45M | 2048 | ImageNet | 更深 ResNet |
| ResNet152 | 60M | 2048 | ImageNet | 最深 ResNet |
| ViT-B/16 | 86M | 768 | ImageNet-21K | 视觉 Transformer |
| ViT-L/16 | 307M | 1024 | ImageNet-21K | 大 ViT |
| CLIP-ViT-B/16 | 86M | 512 | 4亿图文对 | 多模态预训练 |

### 2.3 ProjectionHead (投影头)

```python
# modules.py
class ProjectionHead(nn.Module):
    """
    将嵌入投影到对比学习空间 (d_model → 256)
    
    架构:
        Linear → GELU → Linear → Dropout → Residual → LayerNorm
    """
    def __init__(self, in_dim, out_dim=256):
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 残差连接
        self.residual = nn.Identity()
        
        # 层归一化
        self.ln = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        # Linear → GELU → Linear → Dropout → Residual → LayerNorm
        h = self.linear1(x)
        h = self.activation(h)
        h = self.linear2(h)
        h = self.dropout(h)
        h = h + self.residual(x)  # 残差 (需要维度匹配)
        h = self.ln(h)
        return h
```

### 2.4 CLIPModel (对比学习模型)

```python
# models.py
class CLIPModel(nn.Module):
    """
    CLIP 风格的双塔模型
    """
    def __init__(self, image_encoder, spot_projection, image_projection):
        self.image_encoder = image_encoder
        self.spot_projection = spot_projection   # 投影 ST 表达
        self.image_projection = image_projection # 投影图像特征
    
    def forward(self, spot_features, images):
        # ST 模态
        spot_emb = self.spot_projection(spot_features)
        spot_emb = F.normalize(spot_emb, dim=1)
        
        # 图像模态
        img_features = self.image_encoder(images)
        img_emb = self.image_projection(img_features)
        img_emb = F.normalize(img_emb, dim=1)
        
        # 计算相似度矩阵
        logits = spot_emb @ img_emb.T * temperature
        
        return logits
```

---

## 3. 核心创新

### 3.1 对称交叉熵 + 软目标

SPADE 在对比学习中使用了**软目标 (soft targets)**：

```
标准对比学习 (CLIP):
    
    损失: 交叉熵，以"匹配的 spot-图像对"为硬标签 (one-hot)
    
    例如: 10 个 spot, 10 个图像
    正确配对: spot_i ↔ image_i (标签 = [0,...,1,...,0])
    
    问题: 完全不考虑"spot_i 和 image_j (j≠i) 是否也相似"？

SPADE 的改进 (软目标):
    
    损失: 交叉熵，但标签是"软"的
    
    软标签基于:
    - 空间距离: 空间上靠近的 spot-图像配对比远的更相似
    - 表达相似度: 表达模式相近的 spot 应有更相似的图像

    例如: spot_i 和 spot_j 在空间上相邻
    那么 image_i 和 image_j 都可能与 spot_i 有部分匹配
    标签: [1, 0.5, 0.2, ..., 0] (软分布)
```

**为什么用软目标？**
- 空间转录组中，相邻 spot 的组织学图像高度相似
- 硬标签 (one-hot) 强制模型忽略这种相似性
- 软标签允许模型学习**空间平滑性**

### 3.2 Mixture-of-Data-Experts (多数据专家集成)

```python
# SPADE 支持同时使用多个 ImageEncoder
# 每个代表一个"数据专家" (data expert)

experts = [
    ImageEncoder("resnet50"),       # 专家 1: 自然图像
    ImageEncoder("vit_base"),       # 专家 2: 通用视觉
    ImageEncoder("clip_vit_base"),  # 专家 3: 多模态
]

# 每个专家独立进行对比学习
# 最终通过集成 (平均/投票) 得到最终预测
```

**每个专家的特点**：

| 专家 | 预训练数据 | 知识领域 | 优势 |
|------|-----------|---------|------|
| ResNet50 | ImageNet | 通用物体识别 | 纹理特征强 |
| ViT | ImageNet-21K | 大规模视觉 | 全局结构好 |
| CLIP-ViT | 4亿图文对 | 多模态对齐 | 语义理解好 |

### 3.3 多模态对齐 (ST ↔ H&E)

SPADE 将 CLIP 的思想从图像-文本对齐扩展到**ST-组织学图像对齐**：

```
CLIP (OpenAI):
    图像 ↔ 文本描述
    "一个穿红裙子的女孩" ↔ 图像

SPADE (本模型):
    ST 表达模式 ↔ H&E 组织学图像
    "高表达 CD3, CD4, CD8 → T细胞浸润区域" ↔ 图像中深染区域
```

---

## 4. 数据预处理

### 4.1 ST 数据预处理

```
原始 ST 数据
    │
    ▼
┌───────────────────────┐
│ 基因表达标准化         │
│ - normalize_total     │
│ - log1p              │
│ - 可选: 选择 HVG     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Spot 特征向量          │
│ (每个 spot → 一个向量) │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ ProjectionHead         │
│ 投影到 256 维嵌入空间  │
└───────────────────────┘
```

### 4.2 H&E 图像预处理

```
全切片病理图像 (WSI)
    │
    ▼
┌─────────────────────────────┐
│ 1. 提取 spot 对应的小图补丁  │
│    以每个 spot 坐标为中心    │
│    裁剪 224×224 像素的补丁   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. 图像标准化                │
│    - resize 到 224x224      │
│    - 归一化: mean=[0.485,...]│
│    - 归一化: std=[0.229,...] │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. ImageEncoder              │
│    提取视觉特征并投影        │
└─────────────────────────────┘
```

### 4.3 数据配对

```
同一组织切片的两种模态数据：

     Spot ID  │  基因表达向量     │  H&E 图像补丁
    ──────────┼─────────────────┼────────────────
    spot_1    │  [g₁, g₂, ...]  │  img_patch_1.jpg
    spot_2    │  [g₁, g₂, ...]  │  img_patch_2.jpg
    ...       │  ...            │  ...
    spot_N    │  [g₁, g₂, ...]  │  img_patch_N.jpg

    配对原则: 同一 spot 的两种模态 → 正样本对
             不同 spot 的模态     → 负样本对
```

---

## 5. 训练

### 5.1 训练流程

```python
def train_step(model, spot_features, images, labels):
    """
    spot_features: [batch, d_spot] 基因表达
    images: [batch, 3, 224, 224] H&E 图像
    labels: [batch, batch] 软标签矩阵
    """
    # 1. 编码两种模态
    logits = model(spot_features, images)  # [batch, batch]
    # logits[i][j] = spot_i 和 image_j 的相似度
    
    # 2. 对称交叉熵损失
    # 图像→ST 方向
    loss_i2s = cross_entropy(logits.T, labels)
    # ST→图像 方向
    loss_s2i = cross_entropy(logits, labels)
    
    # 3. 总损失
    loss = (loss_i2s + loss_s2i) / 2
    
    return loss
```

### 5.2 软标签构建

```python
def build_soft_labels(spatial_coords, k=5, sigma=50):
    """
    基于空间距离构建软标签
    
    参数:
    - k: 每个 spot 考虑多少个邻居
    - sigma: 高斯核宽度
    
    输出:
    - soft_labels: [n_spots, n_spots] 软标签矩阵
    """
    # 1. 计算空间距离
    dists = pairwise_distances(spatial_coords)
    
    # 2. 高斯核权重
    weights = torch.exp(-dists² / (2 * sigma²))
    
    # 3. 只保留 top-k 邻居
    mask = torch.topk(weights, k=k, dim=1).values
    weights[weights < mask[:, -1:]] = 0
    
    # 4. 归一化 → 概率分布
    soft_labels = weights / weights.sum(dim=1, keepdim=True)
    
    return soft_labels
```

### 5.3 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 嵌入维度 | 256 | 对比学习空间维度 |
| ImageEncoder | resnet50 | 默认骨干网络 |
| Temperature | 0.07 | Softmax 温度 |
| 学习率 | 1e-4 | AdamW |
| Batch size | 128 | 取决于 GPU 显存 |

---

## 6. 下游任务

### 6.1 支持的任务

| 任务 | 说明 | 方法 |
|------|------|------|
| **空间域分割** | 识别组织结构区域 | ST 嵌入 + 聚类 |
| **跨模态检索** | 由 ST 找相似图像 / 由图像找相似 ST | 对比学习的余弦相似度 |
| **零样本域注释** | 通过图像匹配进行域注释 | 图像嵌入最近邻 |
| **多切片整合** | 对齐不同切片的嵌入空间 | 共享的嵌入空间 |

### 6.2 空间域分割

```python
# 使用 SPADE 进行空间域分割
from SPADE import CLIPModel, ImageEncoder

# 1. 加载模型
model = CLIPModel(
    image_encoder=ImageEncoder("resnet50"),
    spot_projection=ProjectionHead(n_genes, 256),
    image_projection=ProjectionHead(2048, 256)
)

# 2. 训练
model.fit(spot_data, images, soft_labels)

# 3. 提取 ST 嵌入
spot_embeddings = model.encode_spot(spot_data)

# 4. 聚类 → 空间域
from sklearn.cluster import KMeans
domains = KMeans(n_clusters=10).fit_predict(spot_embeddings)
```

---

## 7. 代码结构速览

```
SPADE/
├── models.py                 # CLIPModel (50行)
│   └── CLIPModel
│       ├── image_encoder
│       ├── spot_projection
│       └── image_projection
│
├── modules.py                # 模型组件 (185行)
│   ├── ImageEncoder          # 图像编码器骨干
│   └── ProjectionHead        # 投影头
│
├── config.py                 # 配置文件
├── dataset.py                # 数据集定义
├── experts_routing.py        # 多专家路由逻辑
├── main.py                   # 训练入口
│
├── data_preprocessing/       # 数据预处理
└── downstream/               # 下游任务
```

---

## 8. 关键概念 Q&A

### Q1: SPADE 和 ImageEncoder 集成的关系？

SPADE 的核心设计是**不是使用一个固定的图像编码器，而是支持多个不同的编码器同时训练和推理**：

```
单一编码器方法:
    一个模型 (如 ResNet50) 提取图像特征
    → 受限于该模型的预训练数据

SPADE 的多专家方法:
    ResNet50 + ViT + CLIP-ViT 并行工作
    → 综合不同模型的知识
    → 最终结果通过集成获得
```

### Q2: 为什么需要对齐 ST 和 H&E？

```
生物学动机:
    - H&E 染色是临床最常用的病理学方法
    - 病理学家通过 H&E 图像识别组织结构
    - ST 提供了分子层面的信息

技术动机:
    - 对齐后，可以用 ST 的嵌入"解释"H&E 图像
    - 反之，可以从 H&E 图像推断分子特征
    - 对于只有 H&E 的临床样本，可以预测 ST 模式
```

### Q3: 软目标相比硬标签的优势？

| 方面 | 硬标签 (One-hot) | 软标签 (Soft) |
|------|-----------------|---------------|
| **信息量** | 只有"配/不配" | 包含"有多配" |
| **空间平滑性** | 忽略 | 编码空间邻近信息 |
| **对噪声鲁棒性** | 低 | 高 |
| **训练稳定性** | 可能不稳定 | 更稳定 |

### Q4: CLIP 对比学习的数学形式？

```
给定 batch 的 N 个 spot-图像对:

相似度矩阵: S[i][j] = spot_emb_i · image_emb_j (余弦)

图像→ST 方向: 
    L_i2s = -1/N Σᵢ log(exp(S[i][i]/τ) / Σⱼ exp(S[i][j]/τ))

ST→图像方向:
    L_s2i = -1/N Σᵢ log(exp(S[i][i]/τ) / Σⱼ exp(S[j][i]/τ))

总损失:
    L = (L_i2s + L_s2i) / 2

SPADE 将 softmax 中的 one-hot 标签替换为软标签分布。
```

### Q5: SPADE 和 SpaGCN 都用了 H&E 图像，区别是什么？

| 方面 | SPADE | SpaGCN |
|------|-------|--------|
| **图像使用方式** | 深度 CNN/ViT 编码 | RGB 直方图匹配 |
| **对齐方式** | CLIP 对比学习 | 图像作为 GCN 的边权重 |
| **架构** | 双塔对比学习 | GCN + 图像特征 |
| **图像信息粒度** | 深层语义特征 | 浅层颜色统计特征 |

---

## 9. 延伸阅读

### 核心论文

1. **E Redekop et al.** *SPADE: Spatially-Aware Domain Expert for spatial transcriptomicshttps://arxiv.org/abs/2506.21857v1.* arXiv, 2025. [阅读](https://arxiv.org/abs/2506.21857v1)

### 相关技术

- **CLIP**：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021
- **timm**：[PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) — 预训练模型库
- **ResNet**：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **ViT**：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

### 学习路线

```
空间转录组模型（按模态）：

1. 仅基因表达:
   GraphST → SpaceFlow

2. 基因表达 + H&E 图像:
   SpaGCN (简单颜色特征) 
   ↓
   SPADE (深度视觉特征, CLIP 对齐)  ← 这一篇
```

---

> 💡 **学习建议**：SPADE 将 CLIP 的双塔对比学习思想应用到空间转录组领域，是一个很好的**跨模态学习案例**。建议先理解 CLIP 的原理，再关注 SPADE 的两个关键改进：(1) 用空间邻近性构建软标签，(2) 多图像编码器集成。代码实现非常简洁（models.py 仅 50 行），适合深入阅读。
