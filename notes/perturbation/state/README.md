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
