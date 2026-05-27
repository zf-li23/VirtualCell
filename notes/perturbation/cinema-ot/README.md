---
status: done
filled: 2026-05-27
---

# CINEMA-OT 学习笔记

> CINEMA-OT（Causal INdependent Effect Module Attribution - Optimal Transport）是一个基于最优传输理论的因果扰动推断方法。它利用最优传输（OT）将对照细胞与扰动细胞对齐，将表达变化分解为独立的因果效应模块，从而识别受扰动直接影响的关键基因。它是 pertpy 工具包的核心组件之一。

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
| **论文** | [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5) |
| **发布日期** | 2023 |
| **出版** | Nature Methods |
| **架构** | 最优传输 + 因果分解 |
| **预训练任务** | 因果扰动效应推断 |
| **输入** | 对照 + 扰动 scRNA-seq 数据 |
| **输出** | 因果效应模块 / 扰动响应基因 |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** | Perturb-seq 数据 |
| **代码** | [GitHub](https://github.com/scverse/pertpy) |
| **许可** | MIT |

### 核心思想

> CINEMA-OT（Causal INdependent Effect Module Attribution - Optimal Transport）是一个基于最优传输理论的因果扰动推断方法。它利用最优传输（OT）将对照细胞与扰动细胞对齐，将表达变化分解为独立的因果效应模块，从而识别受扰动直接影响的关键基因。它是 pertpy 工具包的核心组件之一。

---

## 2. 模型架构

### 2.1 核心方法

CINEMA-OT 结合了最优传输（OT）和因果分解：

1. **最优传输对齐**: 将扰动细胞的表达分布对齐到对照细胞的分布
2. **因果效应分解**: 将表达变化分解为独立模块（每个模块对应一个因果效应通路）
3. **显著性检验**: 识别显著受扰动的基因和模块

### 2.2 pertpy 集成

CINEMA-OT 作为 pertpy 工具包的一部分提供：

```python
import pertpy as pt
cinemaot = pt.tools.Cinemaot()
results = cinemaot.fit(adata, pert_key="perturbation", cont_key="control")
```

## 3. 核心创新

### 3.1 最优传输用于因果推断

首次将 OT 用于单细胞扰动数据的因果效应推断。

### 3.2 扰动效应分解

将复杂的转录组变化分解为独立的因果模块，提高可解释性。

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务

| 任务 | 方法 |
|------|------|
| 扰动响应基因识别 | 因果效应模块 |
| 扰动机制分析 | 模块功能富集 |
| 跨条件比较 | OT 距离度量 |

### 4.1 pertpy 中的其他相关工具

pertpy 是 scverse 生态的扰动分析工具包，除 CINEMA-OT 外还包括：

- **Milo**: 差异丰度分析，识别扰动后变化的细胞群体
- **DIALOGUE**: 多组学多模态分析
- **差异表达分析**: Wilcoxon、t 检验、Permutation、EdgeR、PyDESeq2 等

## 8. 代码结构速览

CINEMA-OT 在 pertpy 工具包中：
```
pertpy/
├── tools/
│   ├── _cinemaot.py        # CINEMA-OT
│   ├── _milo.py            # Milo
│   ├── _dialogue.py        # DIALOGUE
│   └── _differential_gene_expression/  # 差异表达
```

## 9. 关键概念 Q&A



## 10. 延伸阅读

- [pertpy 文档](https://pertpy.readthedocs.io)
- [Optimal Transport](https://arxiv.org/abs/1803.00567)
