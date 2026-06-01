---
status: done
filled: 2026-05-27
id: perturbation-linear-baselines
title: Simple Linear Baselines for Perturbation Prediction
category: perturbation
code_url: https://github.com/ekernf01/perturbation_prediction
---
# Simple Linear Baselines for Perturbation Prediction 学习笔记

> 该论文系统性地验证了简单线性模型在扰动预测中的竞争力。与 PCA Still Rules 类似，本文发现简单的线性基线方法在多种扰动预测任务上可以匹配甚至超越复杂的深度学习方法，挑战了"越复杂越好"的普遍假设。该工作为扰动预测领域的实验设计提供了重要指导。

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
| **论文** | [Simple Linear Baselines for Perturbation Prediction](https://www.nature.com/articles/s41592-025-02772-6) |
| **发布日期** | 2025 |
| **出版** | Nature Methods |
| **架构** | 线性回归 / 线性模型 |
| **预训练任务** | 扰动响应预测 |
| **输入** | scRNA-seq 表达数据 |
| **输出** | 扰动后表达预测 |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** | Perturb-seq 数据集 |
| **代码** | [GitHub](https://github.com/ekernf01/perturbation_prediction) |
| **许可** | MIT |

### 核心思想

> 该论文系统性地验证了简单线性模型在扰动预测中的竞争力。与 PCA Still Rules 类似，本文发现简单的线性基线方法在多种扰动预测任务上可以匹配甚至超越复杂的深度学习方法，挑战了"越复杂越好"的普遍假设。该工作为扰动预测领域的实验设计提供了重要指导。

---

## 2. 模型架构

### 2.1 主要结论

与 PCA Still Rules 一致：**简单线性模型在扰动预测任务上具有高度竞争力**。

### 2.2 实验设置

- 多种线性方法 vs. 深度学习方法
- 多个 Perturb-seq 数据集
- 多种评估指标

## 3. 核心创新

### 3.1 启示

- 应对照简单基线（线性方法）报告结果
- 新方法的改进幅度应相对于强基线而非弱基线
- 该结论推动了 Systema 评估框架的建立

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览



## 9. 关键概念 Q&A



## 10. 延伸阅读

- [PCA Still Rules](https://arxiv.org/abs/2410.13956)
- [Systema](https://www.nature.com/articles/s41587-025-02772-6)
