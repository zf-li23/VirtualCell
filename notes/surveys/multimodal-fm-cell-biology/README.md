---
status: done
filled: 2026-05-28
id: multimodal-fm-cell-biology
title: Towards Multimodal Foundation Models in Molecular Cell Biology
category: surveys
---
# Towards Multimodal Foundation Models in Molecular Cell Biology 学习笔记

> 由 Bo Wang（Vector Institute）、Fabian Theis（Helmholtz Munich）等联合撰写的 Nature Perspective，提出在基因组学、转录组学、表观基因组学、蛋白质组学、代谢组学和空间分析等多种组学数据上预训练多模态基础模型的愿景。这些模型有望实现细胞的分子状态表征，促进细胞、基因和组织的整体图谱构建，赋能从新细胞类型识别到 in silico 扰动预测的广泛应用。

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
| **论文** | [Towards Multimodal Foundation Models in Molecular Cell Biology](https://www.nature.com/articles/s41586-025-08710-y) |
| **发布日期** | 2025 |
| **出版** | Nature |
| **架构** | 多模态基础模型愿景 |
| **预训练任务** | 多模态分子细胞生物学基础模型 |
| **输入** | 多组学数据 |
| **输出** | 统一细胞表征 |
| **词表** |  |
| **参数规模** |  |
| **预训练数据** | 基因组、转录组、表观组、蛋白组、代谢组、空间组 |
| **代码** | [GitHub]() |
| **许可** |  |

### 核心思想

> 由 Bo Wang（Vector Institute）、Fabian Theis（Helmholtz Munich）等联合撰写的 Nature Perspective，提出在基因组学、转录组学、表观基因组学、蛋白质组学、代谢组学和空间分析等多种组学数据上预训练多模态基础模型的愿景。这些模型有望实现细胞的分子状态表征，促进细胞、基因和组织的整体图谱构建，赋能从新细胞类型识别到 in silico 扰动预测的广泛应用。

---

## 2. 模型架构

### 2.1 多模态愿景

论文提出将多种组学数据（DNA、RNA、蛋白质、代谢物、空间信息）整合到统一的基础模型中：

1. **统一编码器**: 处理不同模态的异构数据
2. **跨模态对齐**: 学习不同组学之间的对应关系
3. **联合预训练**: 在多个模态上同时进行自监督学习

### 2.2 技术路线

- 继承 LLM 的成功经验（预训练-微调范式）
- 应对多模态数据对齐的挑战
- 利用全球联盟数据（HCA、HuBMAP、HTAN 等）

## 3. 核心创新

### 3.1 多模态细胞基础模型路线图

首次系统地提出从单模态到多模态的细胞基础模型发展路线。

### 3.2 关键应用

- 新细胞类型识别
- 生物标志物发现
- 基因调控推断
- In silico 扰动预测

### 3.3 面临的挑战

- 数据异质性（不同技术平台、批次）
- 模态缺失（部分模态数据不完整）
- 计算可扩展性
- 生物学验证

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览



## 9. 关键概念 Q&A



## 10. 延伸阅读

- [论文](https://www.nature.com/articles/s41586-025-08710-y)
