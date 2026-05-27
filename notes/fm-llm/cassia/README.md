---
status: done
filled: 2026-05-27
---

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
| **论文** | [CASSIA] |
| **发布日期** | 2025 |
| **出版** | Nature Communications |
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
