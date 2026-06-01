---
status: done
filled: 2026-05-27
id: perturbench
title: PerturBench
category: benchmarks
code_url: https://github.com/altoslabs/perturbench
---
# PerturBench 学习笔记

> PerturBench 是针对单细胞扰动预测方法的综合基准测试框架，提供模块化评估管道，涵盖多种数据集、方法和指标，支持扩展。

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
| **论文** | [PerturBench](https://www.nature.com/articles/s41467-024-54159-w) |
| **发布日期** | 2024 |
| **出版** | Nature Communications |
| **架构** | 模块化基准测试框架 |
| **预训练任务** | 扰动预测方法评估 |
| **输入** | Perturb-seq 数据集 |
| **输出** | 标准化性能评估 |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** | 多种 Perturb-seq 数据集 |
| **代码** | [GitHub](https://github.com/altoslabs/perturbench) |
| **许可** | MIT |

### 核心思想

> PerturBench 是针对单细胞扰动预测方法的综合基准测试框架，提供模块化评估管道，涵盖多种数据集、方法和指标，支持扩展。

---

## 2. 模型架构

### 2.1 模块化设计

- 数据集模块: 标准化加载和预处理
- 方法模块: 统一接口封装
- 评估模块: 全面指标计算

### 2.2 覆盖

数据集、方法和指标的全面覆盖。

## 3. 核心创新

模块化设计支持扩展，揭示没有单一方普适方法。

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览

```
perturbench/
├── src/perturbench/
│   ├── modelcore/
│   ├── datasets/
│   └── metrics/
├── configs/
└── README.md
```

## 9. 关键概念 Q&A



## 10. 延伸阅读

- [论文](https://www.nature.com/articles/s41467-024-54159-w)
- [代码](https://github.com/altoslabs/perturbench)
