---
status: done
filled: 2026-05-27
---

# scGPT-spatial 学习笔记

> scGPT-spatial 是 scGPT 的空间转录组学扩展，通过在 3000 万细胞/spot 上持续预训练，并引入混合专家（MoE）解码器、空间感知采样和邻域重建目标，使 scGPT 能够处理 Visium、Visium HD、Xenium、MERFISH 等多种空间转录组数据。

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [scGPT-spatial](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1) |
| **发布日期** | 2025 |
| **出版** | bioRxiv |
| **架构** | scGPT 骨干 + MoE 解码器 + 空间注意力 |
| **预训练任务** | 掩码基因预测 + 空间邻域重建 |
| **输入** | 基因表达 + 空间坐标 |
| **输出** | 基因表达预测 / 空间嵌入 / 反卷积结果 |
| **词表** | 约 30,000 个基因 |
| **参数规模** | 约 300M |
| **预训练数据** | SpatialHuman30M（3000 万细胞/spot） |
| **代码** | [GitHub](https://github.com/bowang-lab/scGPT-spatial) |
| **许可** |  |

### 核心思想

> scGPT-spatial 是 scGPT 的空间转录组学扩展，通过在 3000 万细胞/spot 上持续预训练，并引入混合专家（MoE）解码器、空间感知采样和邻域重建目标，使 scGPT 能够处理 Visium、Visium HD、Xenium、MERFISH 等多种空间转录组数据。

---

## 2. 模型架构


### 2.1 整体设计

```text
scGPT 骨干（持续预训练于空间数据）
         │
    ┌────┴────┐
    │ MoE 解码器│ ← 混合专家（多个专用解码头）
    └────┬────┘
         │
    空间任务输出（反卷积/插补/整合）
```

### 2.2 四项关键创新

1. **持续预训练**: 在 3000 万空间数据上继续训练 scGPT
2. **MoE 解码器**: 不同任务使用不同的专家解码器
3. **空间感知采样**: 训练时考虑空间邻近关系
4. **邻域重建目标**: 预测细胞的邻域表达谱

### 2.3 数据

**SpatialHuman30M**: 精心整理的 3000 万细胞/spot 空间转录组语料库，涵盖 Visium、Visium HD、Xenium、MERFISH 四种平台。



## 3. 核心创新


### 3.1 空间转录组的 scGPT 扩展

首个将单细胞基础模型通过持续预训练适配到空间转录组的工作。

### 3.2 混合专家（MoE）解码器

不同空间任务（反卷积、插补、整合）使用专门训练的专家解码器，避免任务间的负迁移。

### 3.3 空间感知训练策略

- 空间感知采样：同一组织的相邻 spot 同时出现
- 邻域重建：预测一个 spot 的表达时，同时考虑其空间邻居



## 4. 数据预处理


标准 scRNA-seq 预处理 + 空间坐标对齐。



## 5. Tokenization 与输入编码


继承 scGPT 的基因 pair tokenization 方案。



## 6. 预训练


- 基于 scGPT 全人类模型初始化
- 在 SpatialHuman30M 上持续预训练
- 优化目标：掩码预测 + 空间邻域重建



## 7. 下游任务


| 任务 | 方法 | 性能 |
|------|------|------|
| 空间反卷积 | MoE 解码器 | SOTA |
| 缺失基因插补 | 生成式预测 | 优异 |
| 多平台整合 | 空间对齐 | 良好 |
| 细胞类型注释 | scGPT 嵌入 | 竞争力 |



## 8. 代码结构速览


```
scGPT-spatial/
├── scgpt_spatial/        # 核心代码
├── tutorials/            # 教程
├── weights/              # 预训练权重
└── README.md
```



## 9. 关键概念 Q&A


**Q: scGPT-spatial 与 Novae 有何不同？**
A: scGPT-spatial 是通用 scFM 的空间适配，Novae 是专门为空间数据设计的图基础模型。前者更通用，后者更专注。

**Q: 什么是"空间感知采样"？**
A: 传统训练随机采样细胞，忽略其空间位置。空间感知采样确保同一组织的相邻 spot 同时进入一个 batch，让模型学习空间局部一致性。



## 10. 延伸阅读


- [scGPT](https://www.nature.com/articles/s41592-024-02201-0) — 骨干模型
- [Novae](https://www.nature.com/articles/s41592-025-02899-6) — 空间图基础模型
- [MoE (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538) — 混合专家


