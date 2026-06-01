---
status: done
filled: 2026-05-27
id: scgenept
title: scGenePT
category: perturbation
code_url: https://github.com/czi-ai/scGenePT
---
# scGenePT 学习笔记

> scGenePT 是一个利用语言知识增强单细胞扰动预测的模型集合。它通过将 LLM 生成的基因描述嵌入（来自 NCBI、UniProt、GO 等知识源）注入 scGPT 基础模型的架构中，实现了大规模知识的融入。该工作展示了语言知识和基因表达数据结合的巨大潜力，支持多种知识源和多个预训练/微调模型。

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
| **论文** | [scGenePT](https://www.biorxiv.org/content/10.1101/2024.04.07.588418v2) |
| **发布日期** | 2024 |
| **出版** | NeurIPS Workshop / bioRxiv |
| **架构** | scGPT + LLM 基因嵌入 |
| **预训练任务** | 扰动响应预测 |
| **输入** | scRNA-seq + LLM 基因嵌入 |
| **输出** | 扰动后表达预测 |
| **词表** | scGPT 词汇 |
| **参数规模** | 多模型集合 |
| **预训练数据** | 公开 scRNA-seq |
| **代码** | [GitHub](https://github.com/czi-ai/scGenePT) |
| **许可** |  |

### 核心思想

> scGenePT 是一个利用语言知识增强单细胞扰动预测的模型集合。它通过将 LLM 生成的基因描述嵌入（来自 NCBI、UniProt、GO 等知识源）注入 scGPT 基础模型的架构中，实现了大规模知识的融入。该工作展示了语言知识和基因表达数据结合的巨大潜力，支持多种知识源和多个预训练/微调模型。

---

## 2. 模型架构

### 2.1 语言基因嵌入

使用 LLM 从多个知识源编码基因信息：
- **NCBI**: 基因描述
- **UniProt**: 蛋白质功能摘要
- **GO**: 分子功能、生物过程、细胞组分

### 2.2 模型集成

scGenePT 包含多种预训练/微调模型：
- scgenept_go: GO 注释嵌入
- scgenept_ncbi: NCBI 描述嵌入
- scgenept_uniprot: UniProt 摘要嵌入
- 组合模型

## 3. 核心创新

### 3.1 核心创新

- 将大规模语言知识引入单细胞模型
- 多知识源的系统比较
- 开放模型动物园（Model Zoo）

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览



## 9. 关键概念 Q&A



## 10. 延伸阅读

- [GenePT](https://github.com/yiqunchen/GenePT) — 语言基因嵌入的原始工作
- [scGPT](https://github.com/bowang-lab/scGPT) — 基础模型
