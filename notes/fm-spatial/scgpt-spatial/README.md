---
status: done
filled: 2026-05-27
---

# scGPT-spatial 学习笔记

> scGPT-spatial 是 scGPT 模型在空间转录组学领域的持续预训练扩展。在 3000 万细胞/spot 的 SpatialHuman30M 语料库上进行持续预训练，引入了混合专家（MoE）解码器、空间感知采样和基于邻域的重建目标。支持 Visium、Visium HD、Xenium、MERFISH 等多种空间技术平台，实现多模态多切片整合、细胞类型反卷积和缺失基因插补。

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
| **论文** | [scGPT-spatial](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1) |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | GPT + MoE 解码器 + 空间感知机制 |
| **预训练任务** | 持续预训练（空间感知） |
| **输入** | 空间转录组学数据（Visium/Xenium/MERFISH 等） |
| **输出** | 空间细胞嵌入 / 基因表达预测 |
| **词表** | scGPT 词汇 + 空间特异性 token |
| **参数规模** | ~50M+ |
| **预训练数据** | SpatialHuman30M（3000 万细胞/spot） |
| **代码** | [GitHub](https://github.com/bowang-lab/scGPT-spatial) |
| **许可** |  |

### 核心思想

> scGPT-spatial 是 scGPT 模型在空间转录组学领域的持续预训练扩展。在 3000 万细胞/spot 的 SpatialHuman30M 语料库上进行持续预训练，引入了混合专家（MoE）解码器、空间感知采样和基于邻域的重建目标。支持 Visium、Visium HD、Xenium、MERFISH 等多种空间技术平台，实现多模态多切片整合、细胞类型反卷积和缺失基因插补。

---

## 2. 模型架构

### 2.1 持续预训练架构

scGPT-spatial 从 scGPT 的预训练权重初始化，在空间转录组数据上继续训练：

1. **基础编码器**: 复用 scGPT 的因果 Transformer 编码器
2. **MoE 解码器**: 创新性的混合专家解码器，不同专家处理不同的空间模式
3. **空间感知采样**: 在构建训练批次时考虑空间邻域关系

### 2.2 混合专家（MoE）

```python
# MoE 概念代码
class MoEDecoder:
    experts = [SpatialExpert(), GlobalExpert(), LocalExpert()]
    gate = SoftmaxGate()  # 路由到最合适的专家
    def forward(self, x):
        weights = self.gate(x)
        return sum(w * e(x) for w, e in zip(weights, self.experts))
```

### 2.3 邻域重建目标

不仅预测目标 spot 的表达，还预测其空间邻居的表达，强制模型学习空间上下文。

## 3. 核心创新

### 3.1 空间转录组基础模型

首个将单细胞基础模型持续预训练适配到空间转录组的工作。

### 3.2 MoE 解码器

用混合专家机制处理空间数据的异质性——不同空间区域需要不同的解码策略。

### 3.3 SpatialHuman30M

构建了包含 3000 万细胞/spot 的大规模空间转录组预训练语料库。

### 3.4 多平台支持

同时支持 Visium、Visium HD、Xenium、MERFISH 等多种空间技术。

## 4. 数据预处理

### 4.1 空间数据预处理

- 空间坐标对齐
- 组织区域分割
- 基因表达标准化
- 空间邻居图构建

### 4.2 SpatialHuman30M

预训练数据来自多个公开空间转录组数据集，涵盖不同组织和技术平台。

## 5. Tokenization 与输入编码

### 5.1 扩展的 scGPT Tokenization

在 scGPT 的基因 pair token 基础上，添加空间信息 token：
- 空间坐标编码
- 邻域关系编码
- 技术平台标识 token

## 6. 预训练

### 6.1 持续预训练

- 初始化: scGPT 预训练权重
- 数据: SpatialHuman30M（3000 万细胞/spot）
- 目标: 空间感知的邻域重建 + 掩码基因预测

### 6.2 模型权重

V1 权重已公开在 figshare。

## 7. 下游任务

| 任务 | 方法 |
|------|------|
| 多切片整合 | 嵌入对齐 |
| 细胞类型反卷积 | 空间解卷积 |
| 缺失基因插补 | 空间上下文预测 |
| 空间域识别 | 聚类分析 |

提供零样本推理教程。

## 8. 代码结构速览

```
scGPT-spatial/
├── scgpt_spatial/
│   ├── model/
│   │   ├── MoE.py           # 混合专家解码器
│   │   └── flash_layers.py  # Flash Attention
│   ├── tasks/               # 任务定义
│   └── tokenizer/           # 空间 Tokenizer
├── tutorials/               # 零样本推理教程
└── images/                  # 示意图
```

## 9. 关键概念 Q&A

**Q: 和 scGPT 的关系？**
A: scGPT-spatial 是 scGPT 的空间扩展，从 scGPT 权重初始化后持续预训练。

**Q: MoE 的作用？**
A: 不同空间模式（如组织区域边界 vs 内部）需要不同的解码策略，MoE 自动路由。

## 10. 延伸阅读

- [论文](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1)
- [代码](https://github.com/bowang-lab/scGPT-spatial)
- [scGPT 基础](https://github.com/bowang-lab/scGPT)
- [模型权重](https://figshare.com/articles/software/scGPT-spatial_V1_Model_Weights/28356068)
