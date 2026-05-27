---
status: done
filled: 2026-05-27
---

# Heimdall 学习笔记

> Heimdall 是一个用于单细胞基础模型的通用词元化（tokenization）工具包和基准测试框架，基于 Hydra 配置框架，系统比较多种策略。

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
| **论文** | [Heimdall](https://www.biorxiv.org/content/10.1101/2025.01.30.635801v1) |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | 模块化 Tokenization 框架（Hydra） |
| **预训练任务** | 单细胞 Tokenization + 基准测试 |
| **输入** | scRNA-seq count 矩阵 |
| **输出** | 词元序列 / 细胞嵌入 |
| **词表** | 可配置基因词汇表 |
| **参数规模** | 可配置 |
| **预训练数据** | scRNA-seq 数据集 |
| **代码** | [GitHub](https://github.com/ma-compbio-lab/Heimdall) |
| **许可** |  |

### 核心思想

> Heimdall 是一个用于单细胞基础模型的通用词元化（tokenization）工具包和基准测试框架，基于 Hydra 配置框架，系统比较多种策略。

---

## 2. 模型架构

### 2.1 框架

1. Tokenizer 模块: 多种词元化策略
2. 模型模块: 训练 Transformer
3. 评估模块: 下游任务评估

### 2.2 Tokenization 策略

- 表达水平分箱
- 基因-表达联合编码
- 上下文感知词元化

### 2.3 配置

Hydra 配置控制所有参数（数据集、tokenizer、模型架构等）。

## 3. 核心创新

首个系统比较单细胞词元化的框架，模块化可扩展。

## 4. 数据预处理

支持 AnnData 格式，高变基因筛选，缓存机制。

## 5. Tokenization 与输入编码

### 5.1 流程

原始表达值 -> 归一化 -> 离散化 -> Token 序列

### 5.2 配置

num_bins, bin_strategy, special_tokens 等可控。

## 6. 预训练

使用 train.py +experiments=cta_pancreas 运行，Wandb 支持。

## 7. 下游任务

细胞类型分类、细胞-细胞交互。

## 8. 代码结构速览

```
Heimdall/
├── Heimdall/
│   ├── config/config.yaml
│   ├── cell_representations.py
│   └── embedding.py
├── train.py
├── tests/
├── scripts/
├── noxfile.py
└── pyproject.toml
```

## 9. 关键概念 Q&A

Q: 重要性？
A: tokenization 决定模型如何理解基因表达。

## 10. 延伸阅读

- [论文](https://www.biorxiv.org/content/10.1101/2025.01.30.635801v1)
- [代码](https://github.com/ma-compbio-lab/Heimdall)
- [文档](https://heimdall-doc.readthedocs.io/en/latest/)
