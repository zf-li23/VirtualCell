---
status: done
filled: 2026-05-28
---

# scPRINT-2 学习笔记

> scPRINT-2 是下一代单细胞 RNA-seq 基础模型，由 Jérémie Kalfon（Cantini Lab）构建。采用新颖的架构、编码/解码方式和训练范式，在超过 3.5 亿细胞、22,000+ 数据集和 16 个物种上预训练。零样本支持表达去噪与插补、细胞嵌入与批次校正、标签预测（细胞类型、疾病、组织等）、基因网络推断和跨物种整合。基于 Lightning + lamin.ai + scDataLoader 生态构建，支持通过 pLLM 生成基因 token。

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
| **论文** | [scPRINT-2](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1) |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | 自定义 Transformer + Lightning + lamin.ai |
| **预训练任务** | 生成式预训练 + 多任务零样本推理 |
| **输入** | 原始 scRNA-seq count 矩阵 |
| **输出** | 去噪表达 / 细胞嵌入 / 标签预测 / 基因网络 |
| **词表** | ~25K 基因 + 本体论编码 |
| **参数规模** | small/medium/large/vlarge 多种规模 |
| **预训练数据** | 3.5 亿细胞，22K+ 数据集，16 个物种 |
| **代码** | [GitHub](https://github.com/cantinilab/scPRINT-2) |
| **许可** | GPL-3.0 |

### 核心思想

> scPRINT-2 是下一代单细胞 RNA-seq 基础模型，由 Jérémie Kalfon（Cantini Lab）构建。采用新颖的架构、编码/解码方式和训练范式，在超过 3.5 亿细胞、22,000+ 数据集和 16 个物种上预训练。零样本支持表达去噪与插补、细胞嵌入与批次校正、标签预测（细胞类型、疾病、组织等）、基因网络推断和跨物种整合。基于 Lightning + lamin.ai + scDataLoader 生态构建，支持通过 pLLM 生成基因 token。

---

## 2. 模型架构

### 2.1 架构概述

scPRINT-2 基于 Lightning 框架构建，核心架构特点包括：

1. **新型编码器**: 将基因表达原始 count 直接编码为 Transformer 可处理的表示
2. **多层分类头**: 支持细胞类型、疾病、性别、年龄、组织、种族等多种标签的层次化分类
3. **本体论集成**: 使用 lamin.ai 数据库将生物学本体（细胞类型、组织等）编码为模型可理解的表示

### 2.2 多种模型规模

提供 small/medium/large/vlarge 四种配置，在 config/ 目录中定义。

### 2.3 配套生态

- **scDataLoader**: 大规模细胞数据加载器
- **GRnnData**: 基因网络数据处理
- **benGRN**: 基因网络推断基准

## 3. 核心创新

### 3.1 超大规模预训练

3.5 亿细胞、22K+ 数据集、16 物种，是目前规模最大的 scRNA-seq 基础模型之一。

### 3.2 多层次零样本

零样本去噪、插补、嵌入、标签预测、基因网络推断和跨物种整合。

### 3.3 pLLM 基因 token

支持使用蛋白质语言模型（pLLM）生成的基因嵌入作为输入特征。

### 3.4 本体论驱动

使用生物学本体（cell type ontology、tissue ontology 等）进行标签编码，支持层次化分类。

## 4. 数据预处理

输入仅需包含物种本体 ID 和基因名（ENSEMBL/HUGO）。使用 scDataLoader 进行批次化处理。接受原始 count，无需预先整合。

## 5. Tokenization 与输入编码

### 5.1 基因 Token

使用预计算的基因嵌入（可从 HuggingFace 下载）。可选通过 pLLM 生成。

### 5.2 本体编码

细胞类型、疾病等标签通过本体树编码为层次化表示。

## 6. 预训练

### 6.1 预训练

在 3.5 亿细胞上训练，使用 wandb 跟踪。

### 6.2 微调

支持新物种微调、细胞类型分类微调、批次校正微调、跨物种 MMD 微调。

## 7. 下游任务

| 任务 | 命令 |
|------|------|
| 去噪 | scprint2 denoise |
| 嵌入与分类 | scprint2 embed |
| 基因网络推断 | scprint2 gninfer |
| 插补 | scprint2 impute |
| 反事实预测 | scprint2 generate |

## 8. 代码结构速览

```
scPRINT-2/
├── scprint2/          # 核心模型
├── config/            # 配置（small/medium/large/vlarge）
├── notebooks/         # 教程 Notebook
├── data/              # 数据
├── docs/              # 文档
├── tests/             # 测试
├── tools/             # 工具
└── Dockerfile         # Docker 支持
```

## 9. 关键概念 Q&A

**Q: 需要 lamin.ai 吗？**
A: 是的，lamin.ai 用于加载生物学本体和基因信息。

**Q: 支持非人类物种吗？**
A: 已预训练 16 个物种，可微调扩展到新物种。

**Q: 需要 GPU 吗？**
A: 强烈推荐，至少需要一块高性能 GPU。

## 10. 延伸阅读

- [论文](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1)
- [代码](https://github.com/cantinilab/scPRINT-2)
- [文档](https://cantinilab.github.io/scPRINT-2)
- [HuggingFace 模型](https://huggingface.co/jkobject/scPRINT)
