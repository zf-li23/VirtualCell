---
status: done
filled: 2026-05-27
---

# scELMo 学习笔记

> scELMo（Embeddings from Language Models for Single-Cell）受到 NLP 中 ELMo 的启发，利用大语言模型（GPT-4o、DeepSeek 等）生成的基因功能描述嵌入来增强单细胞数据分析。核心思想是将 LLM 对基因功能的语义理解注入到 scRNA-seq 分析流程中，支持零样本细胞注释、批次校正、聚类分析和扰动分析。提供了一个包含预计算嵌入的公开数据库。

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
| **论文** | [scELMo](https://www.sciencedirect.com/science/article/pii/S266638992500279X) |
| **发布日期** | 2025 |
| **出版** | Patterns (Cell Press) |
| **架构** | LLM 嵌入 + 传统 ML / 轻量微调 |
| **预训练任务** | 零样本细胞注释 / 聚类 / 扰动分析 |
| **输入** | scRNA-seq 数据 + LLM 基因嵌入 |
| **输出** | 细胞类型标签 / 聚类结果 / 嵌入 |
| **词表** | 由 LLM 知识覆盖的基因集 |
| **参数规模** | 无（使用外部 LLM） |
| **预训练数据** | 公开 scRNA-seq 数据集 |
| **代码** | [GitHub](https://github.com/HelloWorldLTY/scELMo) |
| **许可** |  |

### 核心思想

> scELMo（Embeddings from Language Models for Single-Cell）受到 NLP 中 ELMo 的启发，利用大语言模型（GPT-4o、DeepSeek 等）生成的基因功能描述嵌入来增强单细胞数据分析。核心思想是将 LLM 对基因功能的语义理解注入到 scRNA-seq 分析流程中，支持零样本细胞注释、批次校正、聚类分析和扰动分析。提供了一个包含预计算嵌入的公开数据库。

---

## 2. 模型架构

### 2.1 整体架构

scELMo 不是传统的端到端深度学习模型，而是一个将 LLM 知识融入单细胞分析的框架：

1. **LLM 查询层**: 向 GPT-4o/DeepSeek 等 LLM 查询基因功能描述
2. **嵌入生成层**: 将基因描述转化为固定维度嵌入向量
3. **分析层**: 使用嵌入增强传统的 scRNA-seq 分析（聚类、注释等）

### 2.2 嵌入类型

- **基因嵌入**: 基于 NCBI/UniProt 基因描述的 LLM 嵌入
- **药物嵌入**: 基于药物分子描述的 GPT-3.5 嵌入
- **序列嵌入**: 基于 Enformer 模型的 DNA 序列嵌入

## 3. 核心创新

### 3.1 LLM 知识注入

首次系统性地将 LLM 对基因功能的语义理解用于增强单细胞分析，不需额外训练即可获得有生物学意义的基因表示。

### 3.2 零样本能力

基于 LLM 嵌入，可以实现零样本的细胞类型注释和聚类分析，无需标注数据。

### 3.3 嵌入数据库

维护公开的基因/药物嵌入数据库，方便社区直接使用。

## 4. 数据预处理

### 4.1 标准数据预处理

使用 Scanpy 标准流程进行数据预处理。

### 4.2 LLM 查询

向 OpenAI/DeepSeek API 查询基因功能描述，需要 API key。对无法访问 OpenAI 的用户提供 DeepSeek 替代方案。

## 5. Tokenization 与输入编码

### 5.1 基因-嵌入映射

每个基因通过其名称在预计算嵌入表中查询对应的 LLM 嵌入向量。嵌入维度取决于使用的 LLM（如 text-embedding-ada-002 输出 1536 维）。

### 5.2 seq2emb 模块

可选的序列到嵌入模块，基于 Enformer 从基因序列直接生成嵌入，适用于无 LLM API 的场景。

## 6. 预训练

### 6.1 预训练策略

scELMo 本身不需要预训练——它利用已经预训练好的 LLM（GPT-4o 等）的基因知识。额外的嵌入可基于 DeepSeek 或 Enformer 生成。

### 6.2 嵌入下载

预计算的嵌入可通过 scELMo 网站下载，包括 gpt4-o 嵌入和 GPT-3.5 药物嵌入。

## 7. 下游任务

| 任务 | 方法 |
|------|------|
| 零样本细胞注释 | LLM 嵌入 + kNN/SVM |
| 细胞聚类 | 嵌入增强的聚类 |
| 批次校正 | 嵌入对齐 |
| 扰动分析 | 集成 CINEMA-OT / CPA / GEARS |
| In Silico 处理 | LLM 指导的基因扰动模拟 |

## 8. 代码结构速览

```
scELMo/
├── Cell-type Annotation/   # 细胞注释
│   ├── cta_gpt.py          # LLM 查询核心逻辑
│   ├── cta_zeroshot.ipynb  # 零样本注释
│   └── cta_ft.ipynb        # 微调注释
├── seq2emb/                # 序列嵌入
├── Perturbation Analysis/  # 扰动分析
│   ├── cinemaot_example.ipynb
│   ├── cpa_example.ipynb
│   └── gears_example.ipynb
├── Get outputs from LLMs/  # LLM 查询
├── Batch Effect Correction/
└── Clustering/
```

## 9. 关键概念 Q&A

**Q: scELMo 和 GenePT 有什么区别？**
A: scELMo 拓展了 GenePT 的思路，支持更多 LLM（GPT-4o, DeepSeek），增加了 seq2emb 模块和更广泛的下游任务。

**Q: 需要 OpenAI API 吗？**
A: 基本使用需要，但也提供 DeepSeek 替代方案。预计算嵌入可下载直接使用，无需 API。

## 10. 延伸阅读

- [论文](https://www.sciencedirect.com/science/article/pii/S266638992500279X)
- [代码](https://github.com/HelloWorldLTY/scELMo)
- [嵌入数据库](https://sites.google.com/yale.edu/scelmolib)
