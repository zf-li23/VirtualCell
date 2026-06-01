---
status: done
filled: 2026-06-02
id: switch
title: SWITCH
category: fm-multi-omics
code_url: https://github.com/zzli123/SWITCH
---

# SWITCH 学习笔记

> SWITCH (integrative deep learning of spatial multi-omics) 是一个基于图注意力网络和变分自编码器的空间多组学整合框架。它专门优化了空间 RNA + 表观组学（ATAC、CUT&TAG 等）的跨模态对齐和翻译，通过图编码器学习细胞间的拓扑结构并进行联合表征。

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [Integrative deep learning of spatial multi-omics with SWITCH](https://www.nature.com/articles/s43588-025-00891-w) |
| **发布日期** | 2025 |
| **出版** | Nature Computational Science |
| **架构** | VAE + GAT（图注意力编码器） |
| **预训练任务** | 跨模态对齐 + 重构 |
| **输入** | 空间 RNA-seq + 空间表观组学（ATAC, H3K27ac, H3K4me3） |
| **输出** | 联合降维表示、跨模态预测 |
| **预训练数据** | 小鼠胚胎 Spatial-ATAC-RNA-seq、小鼠脑 Spatial CUT&TAG-RNA-seq |
| **代码** | [GitHub: zzli123/SWITCH](https://github.com/zzli123/SWITCH) |
| **许可** | 待确认 |

### 核心思想

> SWITCH 借鉴了 GLUE（图-guided 多模态整合）的思想，但专为**空间组学**优化。核心创新是将细胞间的空间邻接关系编码为图，通过图注意力机制让跨模态整合"感知"空间上下文。这和 GLUE 的区别在于：GLUE 用先验知识图引导整合，SWITCH 用空间邻域图引导。

---

## 2. 模型架构

### 2.1 整体架构

```
输入: 空间 RNA 表达   +   空间 ATAC/表观组数据
          │                     │
          ▼                     ▼
  ┌──────────────┐    ┌──────────────────┐
  │ RNA 编码器    │    │ ATAC/表观编码器    │
  │ (Linear)     │    │ (Linear)          │
  └──────┬───────┘    └────────┬─────────┘
         │                     │
         ▼                     ▼
  ┌──────────────────────────────────────┐
  │    图注意力编码器 (GATEncoder)         │
  │  在空间邻接图上传播信息                │
  │  细胞 i 的表示 = GAT(邻居表示)          │
  └──────────────────────────────────────┘
         │                     │
         ▼                     ▼
  ┌──────────────────────────────────────┐
  │    跨模态对齐 + 联合嵌入               │
  │  (对比学习 / MMD 等)                  │
  └──────────────────────────────────────┘
         │
         ▼
  ┌──────────────────────────────────────┐
  │  RNA 解码器    │    表观解码器         │
  └──────────────────────────────────────┘
```

### 2.2 关键组件

**GATEncoder**：使用图注意力网络（GATConv）聚合空间邻域信息。每个细胞的表示不仅取决于自身的组学特征，还取决于其空间邻居的特征。

**VAE 框架**：变分自编码器结构，每一组学都有独立的编码器和解码器，共享一个联合的潜在空间。

### 2.3 与 GLUE 的关系

SWITCH 和 GLUE 共享类似的高层架构（VAE + 图引导），但核心区别：

| 维度 | GLUE | SWITCH |
|------|------|--------|
| **图类型** | 先验知识图（GO/KEGG） | **空间邻域图** |
| **应用场景** | 非空间多模态整合 | **空间**多模态整合 |
| **图注意力** | 标准 GNN | GAT（注意力权重可解释） |

---

## 3. 核心创新

### 3.1 空间感知的多模态整合

之前的空间多模态方法通常独立处理每个模态后再整合。SWITCH 在整合过程中**显式利用空间邻域图**，让 RNA 和表观组学数据在空间上下文中对齐。

### 3.2 支持多种表观组学模态

不仅支持 scATAC-seq，还支持 CUT&TAG 数据（组蛋白修饰 H3K27ac、H3K4me3），覆盖了转录调控的多个层面。

### 3.3 跨模态预测

在联合嵌入空间训练完成后，SWITCH 可以从一种模态（如 RNA）预测另一种模态（如 ATAC），实现"计算虚拟"的多模态实验。

---

## 4. 数据

| 数据集 | 模态 | 组织 |
|--------|------|------|
| 小鼠胚胎 | Spatial RNA + ATAC | E11-E18 小鼠胚胎 |
| 小鼠脑 | Spatial RNA + CUT&TAG | 小鼠脑组织 |

---

## 5. 代码结构速览

```
SWITCH/
├── switch/
│   ├── model.py           # GATEncoder + VAE 核心模型
│   ├── SWITCH.py          # 主接口类
│   ├── utils.py           # 数据处理
│   └── config.py          # 配置
├── tutorials/             # 使用教程
└── README.md
```

---

## 6. 关键概念 Q&A

### Q1: SWITCH 适合什么场景？

当你有同一个组织切片的两种不同空间组学数据（如空间 RNA + 空间 ATAC），想将它们整合到同一个坐标系和潜在空间时，SWITCH 是专门为这种场景设计的。

### Q2: SWITCH 和 GLUE 应该如何选择？

| 场景 | 推荐方法 |
|------|---------|
| 非空间多模态（不同实验的 scRNA+scATAC） | GLUE |
| 空间多模态（同一组织切片的 RNA+ATAC） | SWITCH |

---

## 7. 延伸阅读

- [GLUE](https://www.nature.com/articles/s41587-023-01735-6) — SWITCH 的架构基础
- [GATConv](https://arxiv.org/abs/1710.10903) — 图注意力网络
- [Spatial-ATAC-RNA-seq](https://www.nature.com/articles/s41586-023-05996-6) — 空间多组学技术
