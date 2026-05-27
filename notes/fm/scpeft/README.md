---
status: done
filled: 2026-05-27
---

# scPEFT 学习笔记

> scPEFT 是一个将参数高效微调（PEFT）技术应用于单细胞大语言模型（scLLMs）的框架。不同于全参数微调，scPEFT 在冻结原始 scLLM 参数的同时，仅学习少量低维可插拔适配器（adapters）来估计"模型增量"，实现对 scGPT、scBERT、scFoundation、Geneformer 等多种 scLLM 后端的高效领域适配。发表在 Nature Machine Intelligence。

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
| **论文** | [scPEFT](https://www.nature.com/articles/s42256-025-01050-6) |
| **发布日期** | 2025 |
| **出版** | Nature Machine Intelligence |
| **架构** | PEFT 适配器 + 冻结骨干模型 |
| **预训练任务** | 参数高效微调 / 领域适配 |
| **输入** | scRNA-seq + 任务特定数据 |
| **输出** | 适配后的模型权重 |
| **词表** | 取决于骨干模型 |
| **参数规模** | 骨干冻结 + 少量适配器 |
| **预训练数据** | 下游任务数据 |
| **代码** | [GitHub](https://github.com/coffee19850519/scPEFT) |
| **许可** | MIT |

### 核心思想

> scPEFT 是一个将参数高效微调（PEFT）技术应用于单细胞大语言模型（scLLMs）的框架。不同于全参数微调，scPEFT 在冻结原始 scLLM 参数的同时，仅学习少量低维可插拔适配器（adapters）来估计"模型增量"，实现对 scGPT、scBERT、scFoundation、Geneformer 等多种 scLLM 后端的高效领域适配。发表在 Nature Machine Intelligence。

---

## 2. 模型架构

### 2.1 核心思想

scPEFT 的核心是**冻结预训练 scLLM，仅更新少量适配器参数**，类似于 NLP 中的 LoRA/Adapter 方法。

1. 加载预训练 scLLM（scGPT/Geneformer/scBERT/scFoundation）
2. 冻结所有原始参数
3. 注入低维可学习适配器
4. 仅适配器参数在微调中更新

### 2.2 适配器设计

- 低维降维-升维结构，类似瓶颈层
- 可插拔设计，不同任务使用不同适配器
- 估计 "model delta"——任务适配所需的参数变化量

### 2.3 支持的骨干

scGPT、Geneformer、scBERT、scFoundation（持续扩展中）

## 3. 核心创新

### 3.1 首个 scLLM PEFT 框架

首次将 PEFT 技术系统性地引入单细胞大语言模型领域。

### 3.2 多后端支持

统一接口支持 scGPT、Geneformer、scBERT、scFoundation 四种骨干。

### 3.3 资源高效

相比全参数微调，显存占用降低 10 倍以上，训练时间缩短数倍。

### 3.4 知识保留

冻结预训练参数，保留模型学到的广谱生物学知识，避免灾难性遗忘。

## 4. 数据预处理

### 4.1 数据准备

取决于下游任务（细胞注释、批次校正等）的具体要求，不同骨干模型有不同的预处理需求。

### 4.2 适配器配置

配置适配器维度、位置和类型等超参数。

## 5. Tokenization 与输入编码

### 5.1 取决于骨干模型

scPEFT 复用骨干模型的 tokenization 策略。
- scGPT: 基因 pair token
- Geneformer: rank-value token
- scBERT: 基于 BERT 的编码

## 6. 预训练

### 6.1 预训练策略

scPEFT 本身不进行预训练。

预训练阶段：使用全量数据对骨干模型进行预训练（如 scGPT 的 33M 细胞预训练）。

适配阶段：在冻结骨干上，仅训练适配器参数。

## 7. 下游任务

| 后端 | 下游任务 |
|------|---------|
| scGPT | 细胞类型识别、批次校正、扰动预测、细胞群体发现、Marker 基因检测 |
| scFoundation | 细胞类型识别、扰动预测 |
| Geneformer | 细胞类型识别 |
| scBERT | 细胞类型识别 |

## 8. 代码结构速览

```
scPEFT/
├── geneformer_peft/        # Geneformer 适配器
├── scgpt/                  # scGPT 适配器
│   └── trainer.py          # 训练框架
├── scfoundation/           # scFoundation 适配器
├── tutorial_peft/          # 各后端的微调教程
└── requirements.yaml       # 环境配置
```

## 9. 关键概念 Q&A

**Q: PEFT 相比全参数微调的优势？**
A: 显存占用更低、训练速度更快、避免灾难性遗忘、方便切换任务。

**Q: 适配器的参数量？**
A: 通常仅为原始模型的 1-5%。

**Q: 支持多任务同时适配吗？**
A: 可以通过多个适配器切换实现多任务支持。

## 10. 延伸阅读

- [论文](https://www.nature.com/articles/s42256-025-01050-6)
- [代码](https://github.com/coffee19850519/scPEFT)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
