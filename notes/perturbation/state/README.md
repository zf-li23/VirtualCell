# STATE 学习笔记

> STATE（State Transition + State Embedding）是 Arc Institute 开发的**跨上下文扰动响应预测框架**。它由两个组件构成：(1) **State Embedding (SE) 模型**——预训练通用细胞嵌入；(2) **State Transition (ST) 模型**——给定对照表达和扰动条件，预测扰动后的表达。STATE 的核心能力是在**未见过的细胞类型**上泛化，与 Virtual Cell Challenge 的目标高度一致。

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [代码结构速览](#4-代码结构速览)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [Predicting cellular responses to perturbation across diverse contexts with STATE](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2), bioRxiv 2025 |
| **发布日期** | 2025-06 |
| **出版** | bioRxiv（预印本） |
| **架构** | 双组件：State Embedding (SE) + State Transition (ST) |
| **预训练** | SE：通用细胞嵌入；ST：扰动响应预测 |
| **训练数据** | 大规模 Perturb-seq（Replogle, Nadig, Tahoe-100M 等） |
| **代码** | [GitHub](https://github.com/ArcInstitute/state) |
| **许可** | - |

### 核心思想

> **将"细胞状态"与"状态转移"解耦。** 先学习通用的细胞状态表示（SE），再学习在不同扰动条件下状态如何转移（ST）。这样，模型可以在训练时见过的细胞上学习转移规律，然后应用到未见过的细胞类型上。

---

## 2. 模型架构

### 2.1 两阶段设计

```text
第一阶段：State Embedding (SE)
  细胞表达谱 → 编码器 → 通用细胞嵌入
  预训练目标：掩码表达预测 / 对比学习

第二阶段：State Transition (ST)
  [对照表达嵌入 + 扰动条件] → 解码器 → 扰动后表达
```

### 2.2 跨上下文泛化

ST 模型的核心是学习**扰动引起的表达变化模式**，而非特定于细胞类型的表达水平。这使得它可以将从一种细胞类型学到的扰动效应迁移到另一种细胞类型。

### 2.3 TOML 配置的数据划分

支持灵活的数据划分策略（zero-shot / few-shot 场景），通过 TOML 配置文件指定训练/验证/测试的细胞类型划分。

---

## 3. 核心创新

### 3.1 跨上下文泛化

STATE 的设计核心就是**泛化到未见过的细胞类型**——这是 Virtual Cell Challenge 2025 的核心任务。

### 3.2 嵌入与转移解耦

将细胞表示学习与扰动动力学学习分离：
- SE 模型可以灵活替换（如使用 scGPT、Geneformer 等）
- ST 模型专注于学习扰动模式

### 3.3 Arc Institute 出品

与 Virtual Cell Challenge 来自同一机构，代表了扰动预测领域最前沿的探索。

---

## 4. 代码结构速览

```
state/
├── configs/           # 模型和数据配置（TOML）
├── assets/            # 资源
├── LICENSE
├── MODEL_ACCEPTABLE_USE_POLICY.md
├── MODEL_LICENSE.md
└── README.md
```

---

## 5. 延伸阅读

- [Virtual Cell Challenge](https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) — 跨上下文预测基准
- [Perturb-seq](https://www.cell.com/cell/fulltext/S0092-8674(16)31347-6) — 大规模扰动技术
- [Tahoe-100M](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v3) — 大规模扰动图谱
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [Predicting Cellular Responses to Perturbation Across Diverse Contexts with STATE](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2), bioRxiv 2025 |
| **发布日期** | YYYY-MM |
| **出版** | 期刊/会议 |
| **架构** | 如 BERT / GPT / VAE / GNN / 对比学习 |
| **预训练任务** | 如 MLM / 生成式 / 对比学习 / 自回归 |
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

