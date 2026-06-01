---
status: done
filled: 2026-05-28
id: interpretation-perturbation-sc
title: Interpretation, Extrapolation and Perturbation of Single Cells
category: surveys
---
# Interpretation, Extrapolation and Perturbation of Single Cells 学习笔记

> 由 Oliver Stegle（EMBL/德国癌症研究中心）团队撰写的综述，系统回顾了单细胞分析从描述性图谱向因果推断和机制关系建模的转变。涵盖了解释（interpretation）、外推（extrapolation）和扰动（perturbation）三大主题，讨论了技术进步和不断增长的观测与干预数据集如何推动细胞逻辑的定量理解。

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
| **论文** | [Interpretation, Extrapolation and Perturbation of Single Cells](https://www.nature.com/articles/s41576-025-00920-4) |
| **发布日期** | 2026 |
| **出版** | Nature Reviews Genetics |
| **架构** | 综述/前瞻 |
| **预训练任务** | 单细胞因果推断与机制建模综述 |
| **输入** |  |
| **输出** |  |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** |  |
| **代码** | [GitHub]() |
| **许可** |  |

### 核心思想

> 由 Oliver Stegle（EMBL/德国癌症研究中心）团队撰写的综述，系统回顾了单细胞分析从描述性图谱向因果推断和机制关系建模的转变。涵盖了解释（interpretation）、外推（extrapolation）和扰动（perturbation）三大主题，讨论了技术进步和不断增长的观测与干预数据集如何推动细胞逻辑的定量理解。

---

## 2. 模型架构

### 2.1 三大主题

**解释（Interpretation）**:
- 从细胞图谱到机制理解
- 基因调控网络推断
- 细胞状态和身份的定义

**外推（Extrapolation）**:
- 跨条件、跨组织的泛化
- 基础模型的零样本能力
- 数据稀缺场景的预测

**扰动（Perturbation）**:
- 从观察到干预的因果推断
- 遗传扰动和药物扰动预测
- CRISPR 筛选数据分析

## 3. 核心创新

### 3.1 关键见解

- 单细胞领域正从描述性图谱转向因果机制建模
- 基础模型在零样本外推中展现出潜力但仍有局限
- 因果推断方法（如 CINEMA-OT、CPA）与深度学习结合
- 标准化基准对领域进步至关重要

### 3.2 未来方向

- 从单基因扰动到组合扰动
- 从转录组到多模态扰动
- 从体外到体内的扰动建模
- 从相关到因果的转变

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览



## 9. 关键概念 Q&A



## 10. 延伸阅读

- [论文](https://www.nature.com/articles/s41576-025-00920-4)
