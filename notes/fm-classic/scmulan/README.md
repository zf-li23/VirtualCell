---
status: done
filled: 2026-05-27
id: scmulan
title: scMulan
category: fm-classic
code_url: https://github.com/SuperBianC/scMulan
---
# scMulan 学习笔记

> scMulan 是一个多任务生成式预训练语言模型，用于单细胞转录组学分析。它借鉴了 minGPT 架构和 flash-attention 优化，支持零样本细胞类型注释、零样本批次整合和条件细胞生成（模拟 in-silico 扰动）。目前支持心脏、肺、肝脏、骨髓、血液、大脑和胸腺等 7 个人体器官的零样本细胞注释。

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
| **论文** | [scMulan](https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_16) |
| **发布日期** | 2024 |
| **出版** | RECOMB 2024 |
| **架构** | GPT（Causal Transformer）+ Flash Attention |
| **预训练任务** | 生成式预训练 / 条件生成 |
| **输入** | 基因表达序列 + 条件信息 |
| **输出** | 细胞嵌入 / 生成表达谱 |
| **词表** | ~30K 基因 |
| **参数规模** | ~50M |
| **预训练数据** | 多个器官的 scRNA-seq 数据 |
| **代码** | [GitHub](https://github.com/SuperBianC/scMulan) |
| **许可** |  |

### 核心思想

> scMulan 是一个多任务生成式预训练语言模型，用于单细胞转录组学分析。它借鉴了 minGPT 架构和 flash-attention 优化，支持零样本细胞类型注释、零样本批次整合和条件细胞生成（模拟 in-silico 扰动）。目前支持心脏、肺、肝脏、骨髓、血液、大脑和胸腺等 7 个人体器官的零样本细胞注释。

---

## 2. 模型架构

### 2.1 Causal Transformer 架构

scMulan 基于 minGPT（Karpathy 的最小化 GPT 实现）构建，使用因果注意力机制对基因表达序列进行建模。

- **输入序列**: 类似 scGPT 的基因 pair token 序列
- **注意力**: 使用 Flash Attention 加速训练和推理
- **生成方向**: 从左到右的因果生成

### 2.2 条件生成机制

支持条件控制生成，通过特殊的条件 token 来控制生成细胞类型或组织来源。

## 3. 核心创新

### 3.1 多任务预训练

统一了细胞注释、批次整合和 in-silico 扰动模拟等多个任务在一个生成式框架下。

### 3.2 零样本器官注释

支持 7 个人体器官的零样本细胞类型注释，无需参考图谱或微调。

### 3.3 华为 NPU 支持

除了 CUDA 版本，还提供 NPU（Ascend）版本，支持国产硬件部署。

## 4. 数据预处理

### 4.1 数据准备

输入为标准的 AnnData 对象，包含基因表达矩阵和细胞元数据。

### 4.2 标准化

- 对数归一化
- 基因表达分箱
- 基因 ID 对齐到模型词表

## 5. Tokenization 与输入编码

### 5.1 基因 Pair Token

与 scGPT 类似，每个基因编码为 (gene_token, value_token) pair。
- 基因 token: 词表中的基因 ID
- 值 token: 表达值的离散化分箱

### 5.2 条件 Token

额外的条件 token 用于指示批次、组织等元信息。

## 6. 预训练

### 6.1 预训练任务

生成式预训练：给定部分基因的表达，预测剩余基因的表达值。

### 6.2 数据

多器官 scRNA-seq 数据，涵盖人类 7 个主要器官。

### 6.3 基础设施

借鉴 minGPT 和 flash-attention 实现高效的训练。

## 7. 下游任务

| 任务 | 方法 |
|------|------|
| 零样本细胞注释 | 直接推理，无需微调 |
| 批次整合 | 利用细胞嵌入进行整合 |
| In-silico 扰动 | 条件细胞生成 |

参考教程: Tutorial-cell_type_annotation.ipynb, Tutorial-integration.ipynb

## 8. 代码结构速览

```
scMulan/
├── scMulan/
│   ├── scMulan.py        # 主入口（CUDA）
│   ├── scMulan_npu.py    # 华为 NPU 版本
│   └── model/
│       └── model.py      # 生成式预训练模型
├── inference_cuda.py     # CUDA 推理脚本
├── inference_npu.py      # NPU 推理脚本
├── ckpt/                 # 模型权重
└── Data/                 # 数据目录
```

## 9. 关键概念 Q&A

**Q: scMulan 和 scGPT 有什么区别？**
A: 两者都用 GPT 架构，但 scMulan 更强调多任务能力和零样本注释，scGPT 更强调基础模型的可泛化性。

**Q: 零样本注释支持哪些器官？**
A: 心脏、肺、肝脏、骨髓、血液、大脑、胸腺 7 个器官。

## 10. 延伸阅读

- [论文](https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_16)
- [代码](https://github.com/SuperBianC/scMulan)
- [minGPT](https://github.com/karpathy/minGPT)
- [flash-attention](https://github.com/HazyResearch/flash-attention)
