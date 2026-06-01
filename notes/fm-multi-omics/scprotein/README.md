---
status: done
filled: 2026-06-02
id: scprotein
title: scPROTEIN
category: fm-multi-omics
code_url: https://github.com/TencentAILabHealthcare/scPROTEIN
---

# scPROTEIN 学习笔记

> scPROTEIN 是一个两阶段的深度图对比学习框架，专门用于单细胞蛋白质组学数据的嵌入学习。它先通过多任务异质回归模型估计肽段量化的不确定性，再通过图对比学习生成去噪的细胞嵌入，同时解决数据缺失、批次效应和高噪声三大挑战。

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [scPROTEIN: a versatile deep graph contrastive learning framework for single-cell proteomics embedding](https://www.nature.com/articles/s41592-024-02214-9) |
| **发布日期** | 2024-05 |
| **出版** | Nature Methods |
| **架构** | 两阶段：多任务异质回归 + 图对比学习（GCN） |
| **预训练任务** | 不确定性估计 → 图对比学习 |
| **输入** | 原始肽段级（peptide-level）质谱数据 / 蛋白质级表达矩阵 |
| **输出** | 低维细胞嵌入 |
| **参数规模** | ~10M |
| **预训练数据** | SCoPE2, nanopots, plexDIA 等 |
| **代码** | [GitHub: TencentAILabHealthcare/scPROTEIN](https://github.com/TencentAILabHealthcare/scPROTEIN) |
| **许可** | 待确认 |

### 核心思想

> 单细胞蛋白质组学数据有三个致命问题：**大量缺失值、高噪声、批次效应**。scPROTEIN 的核心洞察是——不是所有测量都值得信任。它先用多任务回归模型为每个肽段测量估计不确定性（质量分），再用图对比学习在"高质量"连接上学习鲁棒的细胞嵌入。

---

## 2. 模型架构

### 两阶段流程

```
阶段 1: 不确定性估计
输入: 原始肽段强度矩阵
    │
    ▼
多任务异质回归模型
(为每个肽段预测均值 + 方差)
    │
    ▼
输出: 校正后的蛋白质丰度 + 不确定性分数

阶段 2: 图对比学习
输入: 校正后的蛋白质表达矩阵
    │
    ▼
GCN 编码器 → 细胞关系图
    │
    ├── 节点级对比损失 (semi-loss)
    └── 原型损失 (prototype loss)
    │
    ▼
输出: 去噪的低维细胞嵌入
```

### 阶段 1 详解

多任务异质回归模型为每个肽段 $p$ 建模：

$$y_p \sim \mathcal{N}(\mu_p(x), \sigma_p^2(x))$$

其中 $\mu_p$ 是预测的丰度，$\sigma_p^2$ 是不确定性（异质方差）。高 $\sigma_p$ 的测量在后续阶段被降权。

### 阶段 2 详解

使用 GCN 作为编码器构建细胞-细胞相似图，联合优化：

- **节点级对比损失**（semi-loss）：拉近相似细胞、推远不同细胞
- **原型损失**（prototype loss）：通过可学习的原型向量增强嵌入的聚类结构

---

## 3. 核心创新

### 3.1 不确定性引导的去噪

首次在单细胞蛋白质组学中引入异质回归的不确定性估计，让模型"知道哪些测量不可靠"。

### 3.2 统一的去噪框架

在一个框架内同时解决缺失值、批次效应和噪声——之前的方法通常只能处理其中一两个问题。

### 3.3 原型对比学习

通过原型损失（Prototype Loss）让同一类细胞的嵌入更紧凑，提升了细胞类型识别的准确率。

---

## 4. 数据

| 数据集 | 技术 | 说明 |
|--------|------|------|
| SCoPE2 | 单细胞蛋白质组学 | 单细胞质谱 |
| nanopots | 单细胞蛋白质组学 | 纳米级样品处理 |
| plexDIA | 单细胞蛋白质组学 | 数据独立采集 |

---

## 5. 代码结构速览

```
scPROTEIN/
├── model.py                  # GCN + 对比学习模型定义
├── train_stage1.py           # 阶段 1 训练入口
├── train_stage2.py           # 阶段 2 训练入口
├── utils.py                  # 数据处理工具
└── config.py                 # 配置参数
```

---

## 6. 关键概念 Q&A

### Q1: scPROTEIN 和传统蛋白质组学方法有何不同？

传统方法通常直接对蛋白质表达矩阵进行标准化和聚类，不显式建模测量质量。scPROTEIN 的两阶段设计——先评估每个测量的可信度，再基于可信的连接学习嵌入——是其独特优势。

### Q2: scPROTEIN 属于基础模型吗？

严格来说不是。它是一个两阶段的**对比学习框架**，训练数据来自特定的蛋白质组学实验，不具备大型预训练基础模型的规模。它属于**面向特定模态的深度学习方法**，而非通用基础模型。

---

## 7. 延伸阅读

- [CITE-seq](https://www.nature.com/articles/nmeth.4380) — 同时测量 RNA 和蛋白
- [SCoPE2 数据集](https://www.nature.com/articles/s41592-021-01316-8)
- [scVI](https://www.nature.com/articles/s41592-018-0229-2) — 单细胞 RNA 的 VAE 方法
