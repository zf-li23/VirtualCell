---
status: done
filled: 2026-05-28
---

# GeneJEPA 学习笔记

> GeneJEPA (Gene Joint-Embedding Predictive Architecture) 是一个基于 I-JEPA 范式的自监督单细胞转录组基础模型。它不再重构嘈杂的表达计数，而是学习在表示空间中预测掩码基因集的潜在结构。采用 Perceiver 架构编码可变长度的基因集、连续值 Fourier 特征编码器、EMA 教师目标与方差-协方差正则化（VICReg），在 Tahoe-100M 图谱上训练。

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
| **论文** | [GeneJEPA: A Predictive World Model of the Transcriptome](https://doi.org/10.1101/2025.10.14.682378) |
| **发布日期** | 2025-10 |
| **出版** | bioRxiv（预印本） |
| **架构** | Perceiver-JEPA (Joint-Embedding Predictive Architecture) |
| **预训练任务** | 表示空间预测（预测掩码区域在表示空间中的结构） |
| **输入** | 基因 ID + log1p 标准化表达值 |
| **输出** | 细胞嵌入（768维） |
| **词表** | 60,000 基因 |
| **参数规模** | ~650M（768维, 24层, 512 latents） |
| **预训练数据** | Tahoe-100M（1亿细胞图谱） |
| **代码** | [GitHub: Genentech/GeneJEPA](https://github.com/Genentech/GeneJEPA) |
| **许可** | Apache 2.0 |

### 核心思想

> **GeneJEPA 的独特角度**：它不试图重建原始的、噪声丰富的基因表达值，而是学习在表示空间中"理解"转录组的潜在结构。通过预测被掩码基因集在教师网络输出的表示，模型被迫学习基因之间的功能关系和细胞状态的内在规律——这是一种"预测世界模型"的范式，而非"重建数据"的范式。

---

## 2. 模型架构

### 2.1 整体架构

GeneJEPA 基于 **I-JEPA**（Image Joint-Embedding Predictive Architecture）框架，包含三个核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                       GeneJEPA 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入: 基因ID + log1p表达值                                   │
│        │                                                     │
│        ▼                                                     │
│  ┌───────────────── Tokenizer ──────────────────┐            │
│  │  基因身份 Embedding + Fourier 表达值编码       │            │
│  │  identity_embed(id)  +  value_encoder(fourier)│            │
│  └──────────────────────────────────────────────┘            │
│        │                                                     │
│        ├──────── 可见基因集 ──────────┐                       │
│        │                              │                       │
│        ▼                              ▼                       │
│  ┌─────────────────┐         ┌─────────────────┐             │
│  │ 学生编码器        │         │ 教师编码器 (EMA)  │            │
│  │ (Perceiver)      │         │ (Perceiver)      │            │
│  │ 仅上下文基因      │         │ 全部基因          │            │
│  └────────┬────────┘         └────────┬────────┘             │
│           │                           │                       │
│           ▼                           ▼                       │
│  ┌─────────────────┐         ┌─────────────────┐             │
│  │ 上下文表示 z_ctx │         │  目标表示 z_tgt  │            │
│  └────────┬────────┘         └────────┬────────┘             │
│           │                           │                       │
│           ▼                           │                       │
│  ┌─────────────────┐                  │                       │
│  │ MLP 预测器       │────────────────►│                       │
│  │ 预测掩码表示      │     VICReg 损失  │                       │
│  └─────────────────┘     (sim+var+cov) │                       │
│                                         │                       │
│  ┌──────────────────────────────────────┘                       │
│  │ EMA: 教师 ← α * 教师 + (1-α) * 学生                         │
│  └─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件详解

#### Tokenizer（scRNATokenizer）

将基因身份和表达值融合为统一的 token 表示：

```python
# 基因身份嵌入
identity_embedding = nn.Embedding(gene_vocab_size, d_identity)(indices)

# Fourier 表达值编码：多频率 sin/cos 编码
fourier_args = values.unsqueeze(-1) * fourier_freqs  # [T, num_freqs]
fourier = cat([sin(fourier_args), cos(fourier_args)], dim=-1)

# MLP 映射
value_embedding = MLP(fourier)  # → d_value

# 拼接 + 投影
token = final_proj(cat([identity_embedding, value_embedding], dim=-1))
```

**关键参数**：
- `d = 768`（总维度）
- `identity_value_split_ratio = 0.5`（各占一半）
- `fourier_num_frequencies = 64`（64 个频率）
- `fourier_freqs`：对数间隔，范围 [0.1, 100.0]

#### Perceiver 编码器（GenePerceiverEncoder）

核心创新——使用固定数量的 latent tokens 处理可变长度的基因输入：

```python
class GenePerceiverEncoder(nn.Module):
    def __init__(self, config):
        # 可学习的 latent tokens [L, d]  L=512
        self.latents = nn.Parameter(torch.randn(config.latents_L, config.d))
        # 交叉注意力：latents (query) ↔ 基因 token (key/value)
        # 自注意力：24 层 Transformer 处理 latent 序列
        self.latent_blocks = nn.Sequential(*[
            LatentTransformerBlock(config) for _ in range(config.blocks_D)  # D=24
        ])
```

**计算复杂度对比**：

| 方法 | 复杂度 | 说明 |
|------|--------|------|
| 标准 Transformer | $O(N^2)$ | 随基因数 N 平方增长 |
| Perceiver | $O(LN + L^2)$ | L=512 固定，$L^2$ 主导，与 N 线性相关 |

#### 预测器（MLPPredictor）

轻量级 MLP，从上下文表示预测教师输出的掩码区域表示。

#### 教师网络（EMA）

教师编码器与学生编码器结构相同，但参数通过 EMA 从学生更新：

```python
teacher_params = ema_decay * teacher_params + (1 - ema_decay) * student_params
```

- `ema_start_decay = 0.992`
- `ema_end_decay = 0.9995`
- `ema_warmup_steps = 2000`

### 2.3 稳定性门控（Stability Gate）

GeneJEPA 设计了一个**稳定性门控机制**来防止训练初期 collapse：

```python
# 监测教师输出的散度（1 - 平均余弦相似度）
dispersion = 1 - mean_cosine_similarity(teacher_outputs)

# 散度 > 阈值时累积"稳定分数"
if dispersion > dispersion_threshold:
    stable_score *= patience_decay  # 衰减
else:
    stable_score += 1  # 累积

# 稳定分数 > 耐心阈值 或 超过最大步数 时开放门控
if stable_score > patience_steps or global_step > safety_flip_after:
    is_teacher_stable = True
```

---

## 3. 核心创新

### 3.1 JEPA 范式首次进入单细胞生物学

与 scGPT、scFoundation 等基于掩码重建的模型不同，GeneJEPA 采用 **I-JEPA 范式**：

| 范式 | 代表模型 | 预测目标 | 优势 | 劣势 |
|------|---------|---------|------|------|
| **生成式 (MAE)** | scGPT, scFoundation | 掩码基因的表达值 | 直观 | 容易过拟合噪声 |
| **对比学习 (CL)** | Tabula, scCLIP | 正负样本区分 | 简单有效 | 依赖负样本质量 |
| **JEPA** | **GeneJEPA** | 掩码区域的表示 | 抗噪, 抽象 | 训练更复杂 |

### 3.2 Perceiver 架构解决变长输入问题

GeneJEPA 的 Perceiver 设计使得**计算复杂度与输入基因数解耦**：

- 无论细胞检测到 500 个还是 5000 个基因，latent 处理的计算量恒定
- 支持**测试时缩放**：推理时读取更多基因以获得更丰富的细胞状态信息，而计算量不变

### 3.3 Fourier 连续值编码

与 scGPT 将表达值分箱为离散 token 不同，GeneJEPA 使用多频率 Fourier 特征编码连续值：

```python
# 对数间隔的 64 个频率
freqs = logspace(log10(0.1), log10(100.0), steps=64)
# 每个表达值生成 64×2 = 128 维特征
fourier = [sin(value * freq), cos(value * freq)]  # for each freq
```

这保留了表达值的精细数值信息，避免了离散化带来的信息损失。

### 3.4 VICReg 损失防止 Collapse

GeneJEPA 使用 VICReg 风格的损失函数，包含三个项：

$$\mathcal{L} = \underbrace{s \cdot \text{sim}(z_{\text{pred}}, z_{\text{tgt}})}_{\text{相似性}} + \underbrace{\lambda_v \cdot \text{var}(z_{\text{pred}})}_{\text{方差}} + \underbrace{\lambda_c \cdot \text{cov}(z_{\text{pred}})}_{\text{协方差}}$$

| 组件 | 作用 | 默认权重 |
|------|------|---------|
| **相似性 (sim)** | 拉近预测和目标表示 | 1.0 |
| **方差 (var)** | 防止表示 collapse 到常数 | 25.0 |
| **协方差 (cov)** | 鼓励表示维度解耦 | 1.0 |

---

## 4. 数据预处理

### 4.1 训练数据

GeneJEPA 在 **Tahoe-100M** 图谱上训练：

| 属性 | 值 |
|------|-----|
| 细胞总数 | 1 亿 |
| 数据来源 | TahoeBio 内部 Perturb-seq 数据集 |
| 基因词汇 | 60,000 |
| 格式 | Hugging Face Datasets |
| 访问 | 需 HF Hub 认证 |

### 4.2 数据流式处理

通过 Hugging Face Datasets 流式加载，支持大规模数据的高效训练：

```python
from datasets import load_dataset
dataset = load_dataset("tahoebio/tahoe-100m", streaming=True, split="train")
```

DataModule 自动处理：
- **流式批处理**：不将全数据加载到内存
- **变长基因集**：每个细胞检测到的基因数量不同
- **Manifest 元数据**：通过 HF Hub 拉取训练/验证划分

---

## 5. Tokenization 与输入编码

### 5.1 输入格式

GeneJEPA 使用**稀疏张量**表示每个细胞：

```python
# 每个细胞表示为 (indices, values, offsets)
indices = [10, 42, 7, 3, 9,   1, 2]        # 基因 ID（拼接）
values  = [0.1, 0.5, 0.3, 2.1, -0.2, 0.0, 1.1]  # log1p 表达（拼接）
offsets = [0, 5, 7]                            # 每个细胞的边界
# 细胞1: 基因 10,42,7,3,9 | 细胞2: 基因 1,2
```

### 5.2 基因身份编码

60,000 个基因通过 `nn.Embedding` 映射到 384 维向量（d 的一半）。

### 5.3 表达值编码

表达值通过 Fourier 编码 + MLP 映射到 384 维向量：

```python
# Fourier 编码：64 个对数间隔频率的 sin + cos
fourier_dim = 64 * 2 = 128
# MLP: 128 → 768 → 384
value_embedding = nn.Sequential(
    Linear(128, 768), GELU(), Linear(768, 384)
)
```

### 5.4 融合投影

```python
combined = cat([identity_embedding(384d), value_embedding(384d)], dim=-1)
final = LayerNorm → Linear(768→768) → GELU → LayerNorm(768)
```

### 5.5 特殊设计

- **无位置编码**：基因集是无序的（bag-of-genes 表示），Perceiver 的交叉注意力天然无序
- **无 [CLS]/[SEP] 等特殊 token**：直接使用 latent tokens 聚合信息

---

## 6. 预训练

### 6.1 预训练数据

| 属性 | 值 |
|------|-----|
| **数据集** | Tahoe-100M |
| **规模** | 1 亿细胞 |
| **基因词汇** | 60,000 |
| **训练样本** | 1,000,000（每 epoch 采样） |
| **验证样本** | 10,000 |

### 6.2 掩码策略

- **掩码比例**: 45%
- **最小上下文基因数**: 512
- **每掩码块最少基因数**: 16
- **多层目标**: 支持多个不同粒度的预测目标

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 模型维度 d | 768 |
| 层数 D | 24 |
| Latents L | 512 |
| Attention heads | 12 |
| 学习率 | 1e-4 |
| Weight decay | 2e-4 |
| Warmup ratio | 5% |
| Batch size | 92 |
| 最大 epoch | 50 |
| 梯度累积 | 2 |
| 梯度裁剪 | 1.0 |
| 优化器 | AdamW (β₁=0.9, β₂=0.98) |
| 混合精度 | bfloat16 |

### 6.4 分块交叉注意力

为了在交叉注意力中避免 $O(L \times N)$ 的完整注意力矩阵内存消耗，GeneJEPA 实现了**在线 softmax 分块注意力**：

```python
# Pass 1: 逐块计算每个 query 的 max score
for k_chunk in k_chunks:
    max_scores = max(max_scores, (q @ k_chunk.T).max())

# Pass 2: 逐块累积 exp(score - max) @ values
for k_chunk, v_chunk in zip(k_chunks, v_chunks):
    exp_scores = exp(q @ k_chunk.T - safe_max_scores)
    weighted_values += exp_scores @ v_chunk
    normalizer += exp_scores.sum(-1, keepdim=True)

output = weighted_values / (normalizer + 1e-6)
```

这使得模型可以在单 GPU 上处理高达 20,000+ 基因的输入。

---

## 7. 下游任务

GeneJEPA 的主要下游能力：

| 任务 | 方式 | 说明 |
|------|------|------|
| **细胞嵌入** | 直接推理 | 使用 `model.get_embedding()` 获取 768 维嵌入 |
| **细胞类型注释** | 嵌入 + 分类器 | 在嵌入上训练线性探针 |
| **药物响应** | 嵌入 + 回归 | 从嵌入预测药物响应 |
| **扰动推理** | 嵌入空间推理 | 利用嵌入比较扰动前后状态 |
| **测试时缩放** | 推理时增加基因数 | 保持 latent 固定，增加输入基因提升质量 |

### 7.1 嵌入提取

```python
module = JepaLightningModule.load_from_checkpoint(ckpt_path)
model = module.model.eval()

with torch.no_grad():
    emb = model.get_embedding(
        indices, values, offsets,
        use_teacher=True  # 使用教师网络获得更高质量嵌入
    )  # [batch, 768]
```

### 7.2 与 Tahoe-100M 的关系

GeneJEPA 使用 Tahoe-100M 进行预训练，而 Tahoe-100M 本身是 TahoeBio 的大规模 Perturb-seq 图谱。这意味着 GeneJEPA 的嵌入天然编码了丰富的扰动响应信息，可用于零样本扰动分析。

---

## 8. 代码结构速览

```
genejepa/
├── __init__.py          # 包初始化
├── configs.py           # 配置参数
│   ├── ModelConfig      # 模型架构参数
│   ├── TrainingConfig   # 训练参数（VICReg 权重等）
│   ├── DataConfig       # 数据加载参数
│   └── ExperimentConfig # 实验管理参数
├── tokenizer.py         # scRNATokenizer — 基因身份 + Fourier 表达编码
├── models.py            # 核心模型定义
│   ├── LatentTransformerBlock  # Transformer 块
│   ├── GenePerceiverEncoder    # Perceiver 编码器
│   ├── MLPPredictor            # 预测器
│   └── GenePerceiverJEPA       # 完整 JEPA 模型
├── data.py              # Tahoe100MDataModule — 流式数据加载
├── train.py             # JepaLightningModule — PyTorch Lightning 训练
├── callbacks.py         # 验证回调（线性探针评估、嵌入质量监控）
│   ├── LinearProbeMLP
│   ├── EmbeddingQualityValidator
│   └── SupervisedValidatorCallback
```

---

## 9. 关键概念 Q&A

### Q1: GeneJEPA 和 scGPT 的核心区别是什么？

| 维度 | GeneJEPA | scGPT |
|------|----------|-------|
| **学习范式** | JEPA（表示预测） | GPT（生成式掩码重建） |
| **损失函数** | VICReg（sim+var+cov） | MSE + 二分类损失 |
| **架构** | Perceiver（固定 compute） | Transformer（随 N 线性） |
| **输入编码** | Fourier 连续编码 | 值域分箱（离散化） |
| **计算复杂度** | $O(LN + L^2\cos\theta)$ | $O(N^2)$ |
| **训练数据** | Tahoe-100M（1亿） | CELLxGENE（~3600万） |
| **测试时缩放** | ✅ 支持 | ❌ 固定输入 |
| **参数规模** | ~650M | ~50M（基础版） |

### Q2: GeneJEPA 的稳定性门控解决了什么问题？

JEPA 训练中一个常见的困难是**表示 collapse**——即所有细胞的嵌入收敛到相同的常数，损失表面看似降低但表示无信息。GeneJEPA 通过：

1. **VICReg 的方差项**惩罚表示维度的标准差低于阈值
2. **稳定性门控**在教师网络输出散度不足时，减小相似性损失的权重

双重机制防止 collapse。

### Q3: 什么是"预测世界模型"范式？

GeneJEPA 不是学习重建数据（如 scGPT 预测掩码表达），而是学习**预测细胞状态的潜在结构**。类比：
- **重建范式**：就像学画画时临摹照片（scGPT）
- **JEPA 范式**：就像学画画时理解物体的三维结构（GeneJEPA）

后者更关注内在的因果和结构关系，而非表面像素（表达值）的精确匹配。这使得：
- 对噪声更鲁棒
- 学到的表示更抽象、更通用
- 更关注基因间的功能关系

---

## 10. 延伸阅读

- [I-JEPA](https://arxiv.org/abs/2301.08243) — Meta AI 提出的图像 JEPA 框架，GeneJEPA 的基石
- [VICReg](https://arxiv.org/abs/2105.04906) — 方差-协方差正则化的原始论文
- [Perceiver IO](https://arxiv.org/abs/2107.14795) — DeepMind 的 Perceiver 架构
- [Tahoe-100M](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v3) — GeneJEPA 的预训练数据来源
- [scGPT](https://www.nature.com/articles/s41592-024-02201-0) — 生成式单细胞基础模型（对比参考）
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
