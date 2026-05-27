# PertAdapt 学习笔记

> PertAdapt 提出了一种**插件式扰动适配器**（plug-in adapter），可以将单细胞基础模型（如 scFoundation、AIDO.Cell）高效地适配到基因扰动预测任务上，而**无需微调整个模型**。其核心创新是**基因相似性掩码注意力**（基于 GO 图）和**自适应损失函数**，在多个 Perturb-seq 数据集上达到 SOTA。

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
| **论文** | [PertAdapt] |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | **适配器插件** + 冻结骨干 FM（scFoundation / AIDO.Cell） |
| **核心组件** | GO 引导的注意力适配器 + 自适应损失 |
| **代码** | [GitHub](https://github.com/theislab/PertAdapt) |
| **许可** | - |

### 核心思想

> **不要重训整个模型，给现有的基础模型加一个"扰动适配器"。** 通过注入 GO 先验知识的注意力机制，让预训练的通用细胞表示快速适配到扰动预测任务上，高效且省资源。

---

## 2. 模型架构

### 2.1 整体设计

```text
基础模型（冻结）         PertAdapt（可训练）
scFoundation /     →    GO 掩码注意力适配器
AIDO.Cell               ↓
(通用细胞表示)          自适应损失
                        ↓
                   扰动后表达预测
```

### 2.2 GO 引导的基因相似性掩码注意力

基于 Gene Ontology 图计算基因之间的功能相似度，用此相似度构造注意力掩码：

```python
go_sim = compute_gene_similarity(gene_i, gene_j)  # GO 图
attention = softmax(Q @ K.T / sqrt(d) + go_mask)
# go_mask 使得功能相关的基因优先交互
```

### 2.3 自适应损失

平衡扰动敏感基因和不敏感基因的损失权重，防止模型被大量"不变基因"主导：

```python
loss = Σ w_i * MSE(pred_i, truth_i)
# w_i 根据基因的扰动响应程度自适应调整
```

---

## 3. 核心创新

### 3.1 插件式适配器

无需微调整个基础模型（节省大量计算资源），仅在骨干网络上叠加一个小型适配器。

### 3.2 GO 先验注入

利用 Gene Ontology 知识指导注意力机制，使模型关注功能相关的基因对。

### 3.3 多骨干支持

支持 scFoundation 和 AIDO.Cell 作为骨干，可扩展。

### 3.4 评估数据集

在 Norman（K562 双基因）、Replogle（RPE1, K562）、Nadig（HepG2, Jurkat）等多个 Perturb-seq 基准上验证。

---

## 4. 代码结构速览

```
PertAdapt/
├── AIDOCell/            # AIDO.Cell 骨干集成
├── preprocess/          # 数据预处理
├── enviornment.yml      # 环境配置
├── img/                 # 图片资源
└── README.md
```

---

## 5. 延伸阅读

- [scFoundation](https://www.nature.com/articles/s41592-024-02305-7) — 骨干模型
- [GEARS](https://www.nature.com/articles/s41587-023-01905-6) — 基因扰动预测基线
- [Gene Ontology](http://geneontology.org/) — 功能基因知识体系
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [PertAdapt] |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
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
