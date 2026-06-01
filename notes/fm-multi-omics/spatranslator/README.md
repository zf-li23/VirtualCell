---
status: done
filled: 2026-06-02
id: spatranslator
title: SpaTranslator
category: fm-multi-omics
code_url: https://github.com/donghongyu2020/SpaTranslator
---

# SpaTranslator 学习笔记

> SpaTranslator 是一个基于图的生成式对抗网络模型，用于空间组学数据的跨模态预测。它可以从一种空间组学模态（如 RNA）直接生成另一种缺失的模态（如 ATAC 或蛋白质），特别优化了空间邻近性约束，使生成的模态在空间上保持合理。

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [SpaTranslator: A deep generative framework for universal spatial multi-omics cross-modality translation](https://www.biorxiv.org/content/10.1101/2025.11.15.688644v1) |
| **发布日期** | 2025-11 |
| **出版** | bioRxiv |
| **架构** | 图生成对抗网络（GAN） + 翻译器 |
| **预训练任务** | 跨模态生成（对抗训练） |
| **输入** | 一种空间组学模态（通常是 RNA） |
| **输出** | 另一种空间组学模态（ATAC / 蛋白质等） |
| **预训练数据** | MISAR-seq（小鼠脑）、10x Visium RNA-Protein |
| **代码** | [GitHub: donghongyu2020/SpaTranslator](https://github.com/donghongyu2020/SpaTranslator) |
| **许可** | 待确认 |

### 核心思想

> 空间组学实验通常只能测量一种模态（RNA 或 ATAC 或 蛋白质），但生物学家往往希望在同一组织切片上同时看到多种模态。SpaTranslator 的核心是**一个条件生成模型**：给定一种模态的数据和空间位置，预测另一种模态在该位置的表达。它使用图结构编码空间邻域信息，用对抗训练确保生成的模态在生物学上合理。

---

## 2. 模型架构

### 2.1 整体架构

```
输入: 空间 RNA 表达 + 空间坐标
          │
          ▼
  ┌──────────────────────┐
  │  RNA 编码器 (GAT)     │
  │  聚合空间邻居信息      │
  └────────┬─────────────┘
           │
           ▼
  ┌──────────────────────┐
  │   翻译器 (Translator) │
  │   RNA 潜在 → ATAC 潜在 │
  └────────┬─────────────┘
           │
           ▼
  ┌──────────────────────┐
  │  ATAC 解码器 + 判别器  │
  │  生成 ATAC + 对抗训练  │
  └──────────────────────┘
```

### 2.2 核心组件

**编码器（R_encoder / A_encoder）**：使用 GAT 图注意力网络编码空间上下文。每个细胞的表示融合了其空间邻居的信息。

**翻译器（Translator）**：核心组件，将一种模态的潜在表示映射到另一种模态的潜在空间。本质是一个条件生成器。

**判别器（Discriminator）**：对抗训练的一部分，判断生成的模态是否"真实"（在统计分布上与实际测量一致）。

### 2.3 数据增强策略

内置基于 scVI MultiVI 的数据增强策略，用于处理稀疏的实验数据——当配对训练数据不足时，通过单细胞多模态数据增强空间模型的训练。

---

## 3. 核心创新

### 3.1 真正的"翻译"——而非整合

SWITCH 和 GLUE 做的是**整合**（找到共同嵌入空间），SpaTranslator 做的是**翻译**（从一种模态生成另一种）。这意味着即使你只有 RNA 数据，也可以"虚拟地"获得 ATAC 数据。

### 3.2 空间感知的生成

生成的跨模态数据在空间上是平滑的——相邻细胞生成的另一种模态表达也是相似的，这符合生物学直觉。

### 3.3 通用的跨模态能力

支持 RNA → ATAC、RNA → 蛋白质、ATAC → RNA 等多种翻译方向，不限于某一种技术平台。

---

## 4. 数据

| 数据集 | 技术 | 模态对 |
|--------|------|--------|
| 小鼠脑 E11-E18 | MISAR-seq | RNA + ATAC |
| 小鼠脑 | 10x Visium | RNA + 蛋白质 |

---

## 5. 代码结构速览

```
SpaTranslator/
├── spatranslator.py        # 主接口类
├── translation_train.py    # 生成对抗网络训练
├── model/
│   ├── encoder.py          # GAT 编码器
│   ├── translator.py       # 翻译器
│   └── discriminator.py    # 判别器
└── utils/
    └── augment.py          # 数据增强
```

---

## 6. 关键概念 Q&A

### Q1: SpaTranslator 和 SWITCH 有何不同？

| 维度 | SWITCH | SpaTranslator |
|------|--------|---------------|
| **目标** | 对齐 + 联合嵌入 | **翻译/生成** |
| **方法** | VAE + GAT | **GAN + GAT** |
| **输出** | 统一潜在空间 | 生成缺失的模态数据 |
| **使用场景** | 已有两种模态，整合分析 | 仅有一种模态，生成另一种 |

### Q2: 生成的数据可信吗？

SpaTranslator 使用对抗训练和空间平滑性约束确保生成质量。但生成的数据应被视为**计算预测**而非真实测量，在关键结论上仍需实验验证。

### Q3: 和简单的线性插补有何不同？

SpaTranslator 是非线性的、空间感知的生成模型。与简单的线性回归或 KNN 插补相比，它能捕捉更复杂的跨模态关系。

---

## 7. 延伸阅读

- [SWITCH](https://www.nature.com/articles/s43588-025-00891-w) — 空间多模态整合（对比参考）
- [GLUE](https://www.nature.com/articles/s41587-023-01735-6) — 图引导多模态整合
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — GAN 基础
- [MISAR-seq](https://www.nature.com/articles/s41592-023-01958-0) — 空间多组学技术
