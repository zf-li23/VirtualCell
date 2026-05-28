---
status: done
filled: 2026-05-28
---

# Prov-GigaPath / Whole-Slide Foundation Model 学习笔记

> Prov-GigaPath 是一个全切片病理基础模型，由 Providence + Microsoft 合作开发。在来自 Providence 医疗系统的 28,000+ 全切片图像上预训练，使用创新的长上下文 Vision Transformer 架构处理病理全切片图像的超高分辨率。在多种病理学任务上达到 SOTA。

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
| **论文** | [Prov-GigaPath / Whole-Slide Foundation Model](https://www.nature.com/articles/s41586-024-07441-w) |
| **发布日期** | 2024 |
| **出版** | Nature |
| **架构** | 长上下文 ViT + 自监督学习 |
| **预训练任务** | 全切片病理图像表示学习 |
| **输入** | 全切片病理图像（WSI） |
| **输出** | 切片级嵌入 |
| **词表** |  |
| **参数规模** | ~1B |
| **预训练数据** | 28,000+ 全切片图像 |
| **代码** | [GitHub](https://github.com/prov-gigapath/prov-gigapath) |
| **许可** | 研究使用 |

### 核心思想

> Prov-GigaPath 是一个全切片病理基础模型，由 Providence + Microsoft 合作开发。在来自 Providence 医疗系统的 28,000+ 全切片图像上预训练，使用创新的长上下文 Vision Transformer 架构处理病理全切片图像的超高分辨率。在多种病理学任务上达到 SOTA。

---

## 2. 模型架构

### 2.1 核心挑战

全切片图像（WSI）通常有数十亿像素，远超标准 ViT 的处理能力。

### 2.2 解决方案

- 将 WSI 分割为 patch 序列
- 使用长上下文 ViT 处理长序列
- 创新的位置编码处理空间关系

### 2.3 预训练

使用自监督学习在 Providence 医疗系统的真实临床数据上预训练。

## 3. 核心创新

### 3.1 真实临床数据

使用 Providence 医疗系统的真实临床 WSI，而非公开数据集。

### 3.2 长上下文处理

首次在病理领域成功将长上下文 ViT 应用于全切片级别。

### 3.3 Nature 发表

被 Nature 接收，代表了病理基础模型的前沿水平。

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务

| 任务 | 描述 |
|------|------|
| 癌症亚型分类 | 多种癌症类型 |
| 基因突变预测 | 从 WSI 预测突变状态 |
| 生存分析 | 预后预测 |

## 8. 代码结构速览



## 9. 关键概念 Q&A



## 10. 延伸阅读

- [论文](https://www.nature.com/articles/s41586-024-07441-w)
- [代码](https://github.com/prov-gigapath/prov-gigapath)
