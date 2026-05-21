# CASSIA 学习笔记

> CASSIA（Cell Annotation with multi-agent Systems for Single-cell Interpretable Analysis）是一个基于**多智能体大语言模型**的自动化、可解释细胞注释系统。它利用多个 LLM agent 协作完成细胞类型标注，不仅给出标签，还能提供**推理过程和生物学证据**。由 Theis Lab（scVI 团队）开发。

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [核心创新](#2-核心创新)
3. [下游任务](#3-下游任务)
4. [代码结构速览](#4-代码结构速览)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [CASSIA: a multi-agent large language model for automated and interpretable cell annotation](https://www.nature.com/articles/s41467-025-67084-x), Nature Communications 2025 |
| **发布日期** | 2025 |
| **出版** | Nature Communications 2025 |
| **架构** | **多智能体 LLM 系统** |
| **核心组件** | 多个 LLM Agent（GPT-4/开源 LLM）+ Marker Gene 知识库 |
| **输入** | 基因表达谱 + 参考数据 |
| **输出** | 细胞类型标签 + 解释性推理 |
| **代码** | [GitHub](https://github.com/theislab/CASSIA) |
| **许可** | - |

### 核心思想

> **让多个 AI agent 像生物学家小组一样协作注释细胞。** 一个 agent 负责识别 marker gene，另一个负责匹配细胞类型，第三个负责验证——整个过程透明可追溯，用户可以审查每一步的推理。

---

## 2. 核心创新

### 2.1 多智能体架构

CASSIA 使用多个 LLM agent 分工协作：
- **Marker Agent**: 从表达谱中识别差异表达基因
- **Annotation Agent**: 匹配 marker gene 到细胞类型
- **Verification Agent**: 验证注释的一致性
- **Explanation Agent**: 生成人类可读的解释

### 2.2 可解释性

不同于黑箱分类器，CASSIA 提供：
- 使用了哪些 marker gene
- 匹配到哪些参考细胞类型
- 置信度评估
- 推理链（chain-of-thought）

### 2.3 Theis Lab 出品

由单细胞领域最著名的团队之一开发（scVI、scANVI、scPoli 等），保证了方法的专业性和可靠性。

---

## 3. 下游任务

| 任务 | 说明 |
|------|------|
| 自动化细胞类型注释 | 全自动 |
| 交互式注释 | 用户可审查和修正 |
| 稀有细胞类型发现 | 利用 LLM 知识 |

---

## 4. 代码结构速览

```
CASSIA/
├── CASSIA_python/        # Python 实现
├── CASSIA_R/             # R 实现
├── CASSIA_example/       # 示例
├── Benchmark/            # 基准测试
└── AGENTS.md             # Agent 说明
```

---

## 5. 延伸阅读

- [scVI](https://www.nature.com/articles/s41592-018-0229-2) — Theis Lab 经典工作
- [scPoli](https://www.nature.com/articles/s41592-023-02035-2) — Theis Lab 参考映射方法
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [CASSIA: A Multi-agent Large Language Model for Automated and Interpretable Cell Annotation](https://www.nature.com/articles/s41467-025-67084-x), Nature Communications 2025 |
| **发布日期** | YYYY-MM |
| **出版** | 期刊/会议 |
| **架构** | 如 BERT / GPT / VAE / GNN / 对比学习 |
| **预训练任务** | 如 MLM / 生成式 / 对比学习 / 自回归 |
| **输入** | 如基因 token + 表达值 |
| **输出** | 如细胞嵌入 / 基因表达预测 / 类型分类 |
| **词表** | 基因数量 |
| **参数规模** | 如 6.5M / 50M / 300M |
| **预训练数据** | 数据来源与规模 |
| **代码** | [GitHub](链接) |
| **许可** | MIT / Apache 2.0 / 自定义 |

### 核心思想

> **用一句话说明这个模型的角度**：它解决了什么问题？用了什么独特的方法？

---

## 2. 模型架构

### 2.1 整体架构图

```text
用 ASCII 艺术图展示整体流程，例如：

Input: [CLS] g_1(v_1) g_2(v_2) ... g_N(v_N)
         │
    ┌────┴────┐
    │Encoder  │  ← 组件名 + 说明
    └────┬────┘
         │
    ┌────┴────┐
    │ Head    │  ← 输出头
    └────┬────┘
         │
    Output
```

### 2.2 核心组件

#### [组件1 名称]

```python
# 伪代码或关键代码片段
# 说明每个组件的作用
```

#### [组件2 名称]

```python
# 伪代码
```

---

## 3. 核心创新

总结 2-3 个这个模型最重要的创新点：

### 3.1 [创新点1]

说明 + 代码/公式（可选）

### 3.2 [创新点2]

说明

### 3.3 [与同类模型的对比]

| 维度 | 本模型 | 模型A | 模型B |
|------|--------|-------|-------|
| 架构 | | | |
| 预训练数据 | | | |
| 核心能力 | | | |

---

## 4. 数据预处理

### 4.1 输入格式

### 4.2 Pipeline

```python
# 从原始数据到模型输入的完整流程
```

### 4.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| n_top_genes | 2000 | 高变基因数量 |
| max_seq_len | 2048 | 最大输入长度 |
| ... | ... | ... |

---

## 5. Tokenization 与输入编码

### 5.1 基因编码

基因是如何表示为 token 的？

### 5.2 表达值编码

表达值是如何编码的？连续值 / 离散化 / 排序？

### 5.3 特殊 Token

[CLS], [SEP], [MASK], [PAD] 等用途。

### 5.4 位置编码

绝对位置 / 相对位置 / 可学习 / 无位置编码？

---

## 6. 预训练

### 6.1 预训练数据

| 数据来源 | 细胞数量 | 组织覆盖 |
|---------|---------|---------|
| ... | ... | ... |

### 6.2 预训练目标

```python
# 损失函数
loss = ...
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-4 |
| Batch Size | 64 |
| 训练步数 | 100K |
| 优化器 | AdamW |

---

## 7. 下游任务

| 任务 | 方法 | 性能 |
|------|------|------|
| 细胞类型分类 | Fine-tune / Zero-shot | ... |
| 基因网络推断 | 注意力权重 / 微调 | ... |
| 扰动预测 | ... | ... |
| 批次校正 | ... | ... |

---

## 8. 代码结构速览

```
project/
├── model.py          ← 模型定义
├── dataset.py        ← 数据加载
├── train.py          ← 训练脚本
├── config.py         ← 配置
└── evaluate.py       ← 评估
```

### 快速开始

```bash
# 如何下载和运行
git clone <repo>
cd <repo>
pip install -e .
python run.py --help
```

---

## 9. 关键概念 Q&A

### Q1: 这个模型与 X 模型的核心区别是什么？

**A**: ...

### Q2: 这个模型有什么已知的局限性？

**A**: ...

### Q3: 这个模型适合什么场景？

**A**: ...

---

## 10. 延伸阅读

- **[相关论文1]**：说明
- **[相关论文2]**：说明
- **[官方文档]**：链接

---

> *笔记最后更新：YYYY-MM-DD*
