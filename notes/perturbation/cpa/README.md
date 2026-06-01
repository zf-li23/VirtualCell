---
status: done
filled: 2026-05-28
---

# CPA 学习笔记

> CPA (Compositional Perturbation Autoencoder) 是一个基于变分自编码器的组合扰动响应预测框架。它通过加性解耦的潜在空间设计，将扰动效应分解为基线状态、扰动分量和协变量分量的组合，支持分布外预测、剂量-响应曲线估计和跨细胞类型的扰动效应迁移。

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
| **论文** | [Predicting cellular responses to complex perturbations in high-throughput screens](https://www.embopress.org/doi/full/10.15252/msb.202211517) |
| **发布日期** | 2023-07 |
| **出版** | Molecular Systems Biology (EMBO Press) |
| **架构** | 变分自编码器 (VAE) + 加性解耦潜在空间 |
| **预训练任务** | 扰动后表达重构（自监督） |
| **输入** | scRNA-seq（基因表达 + 扰动 + 剂量 + 细胞类型 + 批次） |
| **输出** | 扰动后表达预测 + 解耦潜在表示 |
| **词表** | HVG 选择的基因（通常 1000-5000） |
| **参数规模** | ~10-30M |
| **预训练数据** | Combosciplex, Norman, Kang 等 Perturb-seq 数据集 |
| **代码** | [GitHub: theislab/cpa](https://github.com/theislab/cpa) |
| **许可** | MIT |

### 核心思想

> CPA 将扰动响应预测问题建模为**组合式的潜在空间加法**：细胞状态 = 基线状态 + 扰动效应（受剂量调控）+ 协变量效应（如细胞类型）。这种解耦结构使得 CPA 可以通过在潜在空间中操控特定分量来回答反事实问题——"如果这个细胞接受另一种扰动会怎样？"

---

## 2. 模型架构

### 2.1 整体架构

CPA 构建在 **scvi-tools** 框架之上，核心是一个解耦的变分自编码器：

```
输入表达 x (n_genes)
       │
       ▼
┌──────────────────────────────────────┐
│          编码器 (Encoder)              │
│    n_genes → n_latent (默认 128)      │
│    支持变分/确定性模式                │
└──────────────────────────────────────┘
       │
       ▼
z_basal (基线潜在表示)
       │
       ▼  (加性组合)
z = z_basal + z_pert + z_covs
       │           ▲           ▲
       │           │           │
       │    ┌──────┘           │
       │    ▼                  │
       │  ┌───────────┐   ┌───────────┐
       │  │ 扰动网络   │   │ 协变量嵌入 │
       │  │ (Doser)   │   │(Embedding)│
       │  │ 扰动+剂量 │   │ 细胞类型等 │
       │  └───────────┘   └───────────┘
       │
       ▼
┌──────────────────────────────────────┐
│          解码器 (Decoder)              │
│   n_latent → n_genes                  │
│   输出: NB/ZINB/Gaussian 分布参数     │
└──────────────────────────────────────┘
       │
       ▼
   预测表达 px
```

### 2.2 核心公式：加性解耦潜在空间

CPA 的潜在表示遵循**加性组合结构**：

$$z = z_{\text{basal}} + z_{\text{pert}} + z_{\text{covs}}$$

其中：
- $z_{\text{basal}}$：基线细胞状态，由编码器从表达数据推断
- $z_{\text{pert}}$：扰动效应，由扰动网络根据扰动类型和剂量计算
- $z_{\text{covs}}$：协变量效应（如细胞类型、批次），由可学习的 Embedding 计算

这种解耦结构的强大之处在于：

| 操作 | 数学表达 | 应用 |
|------|---------|------|
| 预测扰动效果 | $z_{\text{basal}} + z_{\text{pert}} + z_{\text{covs}}$ | 标准预测 |
| 去除批次效应 | $z_{\text{basal}} + z_{\text{pert}} + z_{\text{covs\setminus batch}}$ | 批次校正 |
| 跨细胞类型迁移 | $z_{\text{basal}} + z_{\text{pert}} + z_{\text{target\_celltype}}$ | 迁移学习 |
| 只预测基底状态 | $z_{\text{basal}} + z_{\text{covs}}$ | 无扰动反事实 |

### 2.3 扰动网络（Doser）

CPA 的扰动网络是其核心创新之一，它接收扰动类型和剂量并输出扰动潜在分量：

```python
z_pert = PerturbationNetwork(perturbation_type, dosage)
```

支持多种**剂量响应曲线**参数化形式：

| 类型 | 公式 | 特点 |
|------|------|------|
| `sigm` | $\sigma(w \cdot d + b)$ | 标准 Sigmoid，饱和响应 |
| `logsigm` | $\log(\sigma(w \cdot d + b) + \epsilon)$ | 对数 Sigmoid，更灵活 |
| `mlp` | $\text{MLP}(d)$ | 通用映射，最灵活 |
| `linear` | $w \cdot d + b$ | 线性响应 |

并通过 **RDKit 化学嵌入**（可选的 `use_rdkit_embeddings`）将药物分子结构信息纳入扰动表示：

```python
mol = Chem.MolFromSmiles(smiles)
fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
```

这使得 CPA 可以预测**训练中未见过的药物**的响应（通过化学结构相似性）。

### 2.4 解码器和损失函数

#### 重构分布

CPA 支持三种重构分布：

| 损失 | 适用场景 | 公式 |
|------|---------|------|
| `gauss` | 对数归一化表达 | $\mathcal{N}(\mu, \sigma^2)$ |
| `nb` | 原始计数 | NegativeBinomial($\mu, \theta$) |
| `zinb` | 零膨胀计数 | ZeroInflatedNB($\mu, \theta, \pi$) |

#### 损失函数

CPA 的完整训练损失包含多个组件：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$$

其中：
- $\mathcal{L}_{\text{recon}}$：重构负对数似然
- $\mathcal{L}_{\text{KL}}$：KL 散度（仅变分模式）
- $\mathcal{L}_{\text{adv}}$：对抗性损失

---

## 3. 核心创新

### 3.1 组合性 + 解耦的潜在表示（核心创新）

CPA 最关键的创新是将扰动响应分解为**加性组合成分**。这与 GEARS 的图传播思想形成鲜明对比——CPA 通过线性加性组合来实现组合性，而非通过图结构传播。

| 能力 | 实现方式 | 应用场景 |
|------|---------|---------|
| 组合扰动 | $z_{\text{pert1}} + z_{\text{pert2}}$ | CRISPR 组合、药物组合 |
| 跨细胞迁移 | 替换 $z_{\text{covs}}$ 为目标细胞类型 | 已知扰动对新细胞类型的预测 |
| 剂量外推 | 修改 Doser 的剂量输入 | 预测不同剂量的效果 |
| 批次校正 | 移除 $z_{\text{batch}}$ 分量 | 整合多个实验批次的表达数据 |

### 3.2 对抗性去偏

CPA 使用**对抗性训练**来确保潜在表示不包含协变量信息。具体来说：

1. 一个对抗性分类器尝试从潜在表示 $z$ 中预测协变量（如细胞类型）
2. 编码器被训练来"欺骗"这个分类器
3. 结果：$z$ 中不再包含协变量信息，提高了跨条件的泛化性

```python
# 对抗性目标
adv_loss = -CE(AdvClassifier(z), true_covariate)
encoder_loss = -adv_loss  # 对抗训练
```

### 3.3 Mixup 增强

CPA 使用 Mixup 增强（$\alpha = 0.2$）对输入和标签进行线性插值：

```python
mixup_lambda = np.random.beta(alpha, alpha)
mixed_x = mixup_lambda * x + (1 - mixup_lambda) * x[index]
```

这提高了模型在分布外预测的鲁棒性。

### 3.4 scvi-tools 生态集成

CPA 构建在 scvi-tools 框架之上，继承了丰富的生态系统功能：

- **标准化的 AnnData 数据处理**（`setup_anndata`）
- **自动超参数优化**（基于 Ray Tune）
- **GPU 加速、分布式训练**
- **与 Scanpy 无缝集成**

---

## 4. 数据预处理

CPA 通过 `setup_anndata` 类方法进行标准化的数据注册：

### 4.1 必需数据

| 字段 | 描述 | 位置 |
|------|------|------|
| 基因表达 | 原始计数或归一化表达 | `adata.X` |
| 扰动信息 | 如 drug_name, CRISPR_target | `adata.obs` |
| 剂量信息 | 数值型剂量 | `adata.obs` |
| 控制标记 | 标记未扰动细胞 | `adata.obs`（如 control_key） |

### 4.2 可选数据

- **细胞类型**：用于跨细胞类型迁移
- **批次信息**：用于批次校正
- **SMILES**：药物分子结构（用于 RDKit 嵌入）
- **HVG 选择**：建议使用 scIB 的 HVG 选择进行多批次整合

### 4.3 预处理流程

```python
# 1. 过滤低质量细胞
sc.pp.filter_cells(adata, min_counts=100)

# 2. 保存原始计数
adata.layers['counts'] = adata.X.copy()

# 3. 归一化 + 对数变换
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 4. HVG 选择（建议跨批次选择）
sc.pp.highly_variable_genes(adata, n_top_genes=5000)

# 5. 注册 AnnData
cpa.CPA.setup_anndata(
    adata,
    perturbation_key='condition',
    dosage_key='dose_val',
    control_key='control',
    categorical_covariate_keys=['cell_type']
)
```

---

## 5. Tokenization 与输入编码

CPA 没有使用复杂的 tokenization 策略。输入直接是：

- **基因表达向量**：HVG 选择后的基因表达值（$\mathbb{R}^{n_{\text{genes}}}$）
- **扰动信息**：独热编码 + 剂量值
- **协变量信息**：独热/类别编码

对于药物分子信息，可选的 RDKit 分子指纹（Morgan Fingerprint, 2048位）作为扰动嵌入的初始化。

---

## 6. 预训练

CPA 的训练过程包含多个阶段：

### 6.1 热身阶段

| 阶段 | 描述 | 持续时间 |
|------|------|---------|
| AE 预训练 | 仅训练自编码器（重构损失） | `n_epochs_pretrain_ae` |
| KL 热身 | 逐渐增加 KL 权重（变分模式） | `n_epochs_kl_warmup` |
| 对抗热身 | 逐渐增加对抗损失权重 | `n_epochs_adv_warmup` |
| Mixup 热身 | 逐渐增加 Mixup 强度 | `n_epochs_mixup_warmup` |

### 6.2 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_latent` | 128 | 潜在空间维度 |
| `n_hidden_encoder` | 256 | 编码器隐藏维度 |
| `n_layers_encoder` | 3 | 编码器层数 |
| `n_hidden_decoder` | 256 | 解码器隐藏维度 |
| `n_layers_decoder` | 3 | 解码器层数 |
| `n_hidden_doser` | 128 | 剂量网络隐藏维度 |
| `n_layers_doser` | 2 | 剂量网络层数 |
| `variational` | False | 是否使用变分模式 |
| `mixup_alpha` | 0.0 | Mixup 强度（0=禁用） |
| `reg_adv` | 1.0 | 对抗损失权重 |

---

## 7. 下游任务

### 7.1 组合扰动预测

CPA 的核心任务——预测药物或基因组合的转录响应：

```python
model.predict(adata, pert_name=["drugA+drugB"])
```

### 7.2 剂量-响应曲线

通过改变 Doser 网络的剂量输入，可以生成完整的剂量-响应曲线：

```python
for dose in [0.1, 0.5, 1.0, 2.0, 5.0]:
    pred = model.predict(adata, pert_name=["drugA"], dosage=dose)
```

### 7.3 跨细胞类型迁移

通过操控协变量分量，可以预测扰动在未见过的细胞类型上的效果：

```python
# 将协变量从细胞类型A改为细胞类型B
pred = model.predict(adata, cell_type=["cell_type_B"])
```

### 7.4 批次校正

通过减掉批次分量，可以在保持生物变异的同时去除批次效应：

```python
corrected_adata = model.remove_batch_effect(adata)
```

### 7.5 解耦的潜在表示分析

CPA 的加性潜在空间支持丰富的可解释性分析：

- **扰动嵌入聚类**：观察不同扰动在潜在空间中的关系
- **细胞类型嵌入**：分析细胞类型相似性
- **剂量-响应轨迹**：沿剂量维度观察潜在表示的变化

---

## 8. 代码结构速览

```
cpa/
├── __init__.py          # 包初始化
├── _api.py              # 高层 API（predict, remove_batch_effect 等）
├── _data.py             # AnnData 数据处理
├── _metrics.py          # 评估指标（Pearson, R², KNN 纯度）
├── _model.py            # CPA 主类 (继承 BaseModelClass)
├── _module.py           # CPAModule — PyTorch 模块定义
│   ├── Encoder          # 编码器（VAE/确定性）
│   ├── PerturbationNetwork # 扰动网络（含 Doser）
│   ├── Decoder          # 解码器（NB/ZINB/Gauss）
│   └── covars_embeddings # 协变量嵌入
├── _task.py             # CPATrainingPlan — 训练逻辑
│   ├── 对抗性训练       # 编码器 vs. 分类器
│   ├── Mixup 增强       # 输入插值
│   └── 多阶段热身       # 递进式训练
├── _tuner.py            # 超参数优化（Ray Tune）
├── _utils.py            # 工具函数（PerturbationNetwork 等）
└── _plotting.py         # 可视化工具
```

---

## 9. 关键概念 Q&A

### Q1: CPA 和 GEARS 有何异同？

| 维度 | CPA | GEARS |
|------|-----|-------|
| **核心思想** | 加性解耦潜在空间 | 图引导的扰动传播 |
| **先验知识** | 可选 RDKit（药物化学） | GO 图 + 共表达网络 |
| **组合性** | 线性加性组合 | GNN 传播融合 |
| **剂量建模** | ✅ Doser 网络 | ❌ |
| **跨细胞类型** | ✅ 协变量嵌入 | ❌（不支持） |
| **批次效应** | ✅ 内置 | ❌ |
| **框架** | scvi-tools | 独立 PyTorch + PyG |

### Q2: CPA 的对抗性训练解决了什么问题？

当编码器学习到利用协变量信息（如细胞类型）来帮助重构时，潜在表示会混入协变量信息。这意味着"细胞类型 A 的扰动响应"和"细胞类型 B 的扰动响应"会混杂在一起。对抗性分类器强制潜在表示不包含协变量信息，从而：

1. 确保扰动分量的**跨细胞类型可迁移性**
2. 提高分布外预测的**泛化性**
3. 使得潜空间更具**生物学可解释性**

### Q3: CPA 后续有哪些重要的扩展工作？

CPA 的加性解耦思想影响了后续多个工作：
- **scPoli** (Nat Methods 2023) — 群体水平的单细胞整合（同组后续工作）
- **PertAdapt** (bioRxiv 2025) — 跨条件适应的扰动预测
- **scLAMBDA** (bioRxiv 2024) — 多基因扰动预测

---

## 10. 延伸阅读

- [GEARS](https://www.nature.com/articles/s41587-023-01905-6) — 同期发表的图神经网络扰动方法
- [scPoli](https://www.nature.com/articles/s41592-023-02035-2) — 同组的群体水平整合方法
- [scVI (scvi-tools)](https://www.nature.com/articles/s41587-025-02857-9) — CPA 的底层框架
- [Systema](https://www.nature.com/articles/s41587-025-02777-8) — 扰动预测系统性评估
