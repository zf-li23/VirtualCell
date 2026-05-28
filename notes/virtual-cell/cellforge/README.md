---
status: done
filled: 2026-05-28
---

# CellForge 学习笔记

> CellForge 是一个开放式的端到端多智能体（multi-agent）框架，用于单细胞组学计算方法的自主设计。不同于传统的单细胞模型，CellForge 通过多智能体协作和 LLM 驱动的任务分解，自动完成任务分析、方法设计到代码生成的全流程。它本身不是模型，而是一个"设计模型的模型"（meta-tool）。

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
| **论文** | [CellForge]() |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | 多智能体 LLM 协作框架（含 RAG + MCP + OpenHands） |
| **预训练任务** | 自动化方法设计与代码生成 |
| **输入** | 任务描述 + 数据集信息 |
| **输出** | 研究计划 + Mermaid 架构图 + Python 代码 |
| **词表** |  |
| **参数规模** | 基于 GPT-4/Claude/DeepSeek 等 API |
| **预训练数据** | 通过 RAG 检索文献和代码库 |
| **代码** | [GitHub](https://github.com/gersteinlab/CellForge) |
| **许可** | MIT |

### 核心思想

> CellForge 是一个开放式的端到端多智能体（multi-agent）框架，用于单细胞组学计算方法的自主设计。不同于传统的单细胞模型，CellForge 通过多智能体协作和 LLM 驱动的任务分解，自动完成任务分析、方法设计到代码生成的全流程。它本身不是模型，而是一个"设计模型的模型"（meta-tool）。

---

## 2. 模型架构

### 2.1 核心流程

CellForge 的工作流程分为三个阶段：

1. **Task Analysis（任务分析）**: 分析任务描述，评估基线方法，检索相关知识
2. **Method Design（方法设计）**: 多领域专家智能体讨论协作，生成研究计划
3. **Code Generation（代码生成）**: 使用 OpenHands 自动生成 Python 实现代码

### 2.2 专家智能体池

包含多个领域专家：
- **深度学习专家**: 神经网络架构设计
- **生物学专家**: 生物约束与通路知识
- **训练专家**: 训练策略和超参数优化
- **评估专家**: 指标设计和验证策略

### 2.3 基础设施

- RAG 检索: 基于 Qdrant 向量数据库
- MCP 知识库: 连接外部知识源
- Mermaid 图表: 自动生成架构图
- OpenHands: Docker 容器化代码生成

## 3. 核心创新

### 3.1 多智能体方法设计

首次将 LLM 多智能体协作引入单细胞计算方法设计。

### 3.2 端到端自动化

从任务描述 → 研究计划 → 架构图 → Python 代码，全自动完成。

### 3.3 知识增强 RAG

通过 RAG 检索相关文献和开源代码，为方法设计提供知识支撑。

## 4. 数据预处理



## 5. Tokenization 与输入编码



## 6. 预训练



## 7. 下游任务



## 8. 代码结构速览

```
CellForge/
├── cellforge/
│   ├── Task_Analysis/    # 任务分析模块
│   │   ├── baseline_assessor.py
│   │   └── rag.py        # RAG 检索
│   ├── Method_Design/    # 方法设计（核心）
│   │   ├── main.py       # 入口
│   │   ├── experts.py    # 专家智能体
│   │   ├── expert_discussion.py  # 专家讨论
│   │   └── refinement.py # 计划精化
│   └── Code_Generation/  # 代码生成
│       └── auto_start_openhands.py
├── setup.py
├── setup_env.py
└── main.py
```

## 9. 关键概念 Q&A

**Q: CellForge 是模型还是工具？**
A: 是元工具（meta-tool），用于自动设计单细胞分析方法的框架。

**Q: 需要什么 API？**
A: 需要 LLM API（OpenAI/Anthropic/DeepSeek 等）和搜索 API（GitHub/SerpAPI）。

**Q: 可以设计哪些类型的方法？**
A: 主要是扰动预测方法，但框架是通用的。

## 10. 延伸阅读

- [代码](https://github.com/gersteinlab/CellForge)
