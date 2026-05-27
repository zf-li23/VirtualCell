---
status: done
filled: 2026-05-27
---

# Tahoe-100M / Tahoe-x1 学习笔记

> Tahoe-100M 和 Tahoe-x1 是**全球最大的单细胞扰动图谱和扰动预训练基础模型**。Tahoe-100M 构建了包含 **1 亿+ 扰动细胞**的数据集（覆盖 ~60,000 个药物扰动实验，50 种细胞系），揭示了扰动响应的**上下文依赖性**；Tahoe-x1 在此基础上训练了首个 **30 亿参数**的扰动预训练单细胞 FM，在癌症相关基准上达到 SOTA。

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [核心创新](#2-核心创新)
3. [数据](#3-数据)
4. [代码结构速览](#4-代码结构速览)
5. [延伸阅读](#5-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文 1** | [Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v3), bioRxiv 2025 |
| **论文 2** | [Tahoe-x1: Scaling Perturbation-Trained Single-Cell Foundation Models to 3B](https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1), bioRxiv 2025 |
| **作者** | Tahoe Therapeutics |
| **架构 (Tx1)** | Decoder-only Transformer（Composer/llm-foundry + FSDP + FlashAttention） |
| **参数 (Tx1)** | 70M → **3B** |
| **预训练数据** | **2.66 亿细胞**: CellxGene (61M) + scBaseCamp (112M) + **Tahoe-100M** (96M 扰动细胞) |
| **关键发现** | 扰动响应具有强烈的**上下文依赖性**——相同扰动在不同细胞系中效果不同 |
| **代码** | [GitHub](https://github.com/MarioniLab/tahoe-100m) |

### 核心思想

> **扰动预测的下一个前沿是规模。** Tahoe-100M 证明了构建大规模扰动图谱的可行性，Tahoe-x1 则证明了在这个图谱上训练的模型可以学到通用的扰动表示，其参数量从 70M 到 3B 展示了清晰的 scaling law。

---

## 2. 核心创新

### 2.1 Tahoe-100M：最大的扰动图谱

- **1 亿+ 扰动细胞**
- **~60,000 个药物扰动实验**
- **50 种不同细胞系**
- 系统揭示了"相同的药物在不同细胞系中产生不同的基因表达变化"

### 2.2 Tahoe-x1：3B 参数扰动 FM

- 首个达到 3B 参数的扰动预训练模型
- 基于 Composer/llm-foundry 框架，支持 FSDP 分片和 FlashAttention
- 训练效率比先前实现提升 **3-30 倍**
- 在 DepMap、MSigDB、状态转移预测等任务上达到 SOTA

### 2.3 上下文依赖性

核心生物学发现：**细胞环境（细胞类型、状态）对扰动响应的影响可能与扰动本身同样重要**。这一发现对精准医学和药物研发有重要启示。

---

## 3. 数据

| 数据集 | 细胞数 | 说明 |
|--------|-------|------|
| CellxGene (2025-01) | ~61M | 公开参考数据 |
| scBaseCamp (2025-02) | ~112M | 多样化正常细胞 |
| Tahoe-100M | ~96M | **扰动实验数据** |
| **总计** | **~266M** | |

---

## 4. 代码结构速览

```
tahoe-100m/
├── configs/          # 模型配置
├── assets/           # 资源文件
├── MANIFEST.in
└── README.md
```

---

## 5. 延伸阅读

- [STATE](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2) — 跨上下文扰动预测
- [Perturb-seq](https://www.cell.com/cell/fulltext/S0092-8674(16)31347-6) — 扰动技术
- [Virtual Cell Challenge](https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0) — 扰动预测基准
