---
status: done
filled: 2026-05-27
id: scdrugmap
title: scDrugMap
category: perturbation
code_url: https://github.com/QSong-github/scDrugMap
---
# scDrugMap 学习笔记

> scDrugMap 是跨基础模型的药物响应映射框架，支持 GPT4、Geneformer、scGPT、UCE 等模型上统一预测药物效应。

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
| **论文** | [scDrugMap](https://www.biorxiv.org/content/10.1101/2024.11.15.623839v1) |
| **发布日期** | 2024 |
| **出版** | bioRxiv |
| **架构** | 扰动-药物映射框架 |
| **预训练任务** | 药物响应预测 |
| **输入** | scRNA-seq + 药物信息 |
| **输出** | 细胞类型级药物响应 |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** | Perturb-seq + 药物库 |
| **代码** | [GitHub](https://github.com/QSong-github/scDrugMap) |
| **许可** |  |

### 核心思想

> scDrugMap 是跨基础模型的药物响应映射框架，支持 GPT4、Geneformer、scGPT、UCE 等模型上统一预测药物效应。

---

## 2. 模型架构

### 2.1 流程

1. 扰动数据收集
2. 多基础模型编码
3. 扰动-药物映射
4. 细胞类型级预测

### 2.2 多模型

## 3. 核心创新

跨模型映射、细胞类型级分辨率。

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览

```
scDrugMap/
├── code/
│   ├── gpt4/
│   ├── geneformer/
│   ├── scgpt/
│   └── uce/
```

## 9. 关键概念 Q&A



## 10. 延伸阅读

- [论文](https://www.biorxiv.org/content/10.1101/2024.11.15.623839v1)
- [代码](https://github.com/QSong-github/scDrugMap)
