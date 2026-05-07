# 🧬 VirtualCell — 虚拟细胞大模型学习

> 空间组学 | 虚拟细胞基础模型 | Python | conda env: `zf-li23` | ❌ 无 GPU

---

## 📂 结构

```
VirtualCell/
├── README.md         ← 你在这里
├── scGPT/            # scGPT 论文笔记 + 复现
├── Geneformer/       # Geneformer 论文笔记 + 复现
├── Spatial/          # 空间组学模型笔记 + 复现
└── .gitignore
```

**看论文 → 做笔记 → 拉代码 → 跑通 → 总结**，都在各自的目录里完成。

---

## 🔥 关键论文速览（2023-2025）

### 路线一：单细胞转录组基础模型

| 模型 | 发表 | 参数量 | 架构 | 核心思路 | 推荐 |
|------|------|--------|------|---------|------|
| **Geneformer** | Nature 2023 | 6.5M | Transformer 6层 | 基因按表达量排序做 token | ⭐ 最易上手 |
| **scGPT** | Nature Methods 2024 | ~50M | GPT (Causal) | 基因 pair token，支持多模态 | ⭐ 首选复现 |
| **scFoundation** | bioRxiv 2024 | **3B** | 非对称 AE | 超大模型，引入 GO 先验 | 了解思想 |
| **CellLM** | 2024 | ~110M | BERT | 图结构先验 | 选读 |
| **UCE** | CZI 2024 | - | 对比学习 | 跨物种通用细胞嵌入 | 拓展阅读 |

---

## 🗺️ 学习路线

```
第1步（1-2周）：读论文
  ├─ Geneformer → 读原文，Geneformer/ 里写笔记
  └─ scGPT → 读原文，scGPT/ 里写笔记

第2步（2-3周）：跑代码
  ├─ scGPT → git clone → 跑通 Tutorial
  └─ Geneformer → huggingface 加载 → 跑推理

第3步（1周+）：结合空间组学
  └─ scGPT embedding + squidpy 做空间聚类
```

每篇论文搞懂这三点：
1. **基因怎么编码成 token？**
2. **怎么训练的？（掩码？预测？对比？）**
3. **怎么用到下游任务？**

---

## ⚙️ 环境

```bash
conda activate zf-li23
```

已有：`scanpy` `squidpy` `spatialdata` `anndata` `pytorch` `jax`

可能要装：
```bash
conda run -n zf-li23 pip install transformers datasets huggingface_hub
```

---

## 🚀 快速开始

```python
# Geneformer（最轻量，几秒钟加载）
from transformers import AutoModel
model = AutoModel.from_pretrained("ctheodoris/Geneformer")  
# 模型只有6.5M参数，CPU完全可跑
```

```bash
# scGPT
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT
pip install -e .
jupyter notebook tutorial/Tutorial.ipynb
```

---

## 📥 常用数据集

| 数据集 | 用途 | 获取 |
|--------|------|------|
| PBMC 3k | 入门测试 | `scanpy.datasets.pbmc3k()` |
| CELLxGENE | 千万级参考 | cellxgene.cziscience.com |
| MERFISH | 空间建模 | vizgen.com |
| 10x Visium | 空间转录组 | squidpy 可加载 |

---

## ✅ 第一步建议

```bash
conda activate zf-li23
cd ~/VirtualCell

# 今天就做：
# 1. 读 Geneformer 论文 → 在 Geneformer/ 写 notes.md
# 2. 或者直接拉 scGPT 代码跑起来
```

---

*有问题随时在工作区记录。最后更新：2026-05-01*
