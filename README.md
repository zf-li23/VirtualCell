# 🧬 VirtualCell — 虚拟细胞大模型学习

> 单细胞基础模型 | 虚拟细胞 | 空间组学 | Python | conda env: `zf-li23` | ❌ 无 GPU

---

## 📂 项目结构

```
VirtualCell/
├── README.md              ← 项目说明
├── papers.md              # 论文清单（~180 篇，7 大分类）
├── REPO_MAP.md            # 论文 ↔ GitHub 仓库映射（含验证状态）
├── PLAN.md                # 笔记收集计划
│
├── notes/                 # 📗 论文笔记（7 大分类，~180 篇）
│   ├── _template.md       #    笔记模板（含 frontmatter 标签）
│   ├── fm/                #    单细胞基础模型（核心）
│   ├── fm-llm/            #    FM + LLM
│   ├── perturbation/      #    遗传扰动预测
│   ├── virtual-cell/      #    虚拟细胞
│   ├── benchmarks/        #    评估与 Benchmark
│   ├── pathology/         #    病理基础模型
│   └── surveys/           #    综述与展望
│
├── scripts/               # 🐍 工具脚本
│   ├── generate_note.py   #    从结构化数据生成笔记
│   ├── fill_repo_links.py #    填充仓库链接 + 元数据
│   ├── sync_notes_config.py #  自动发现笔记 + 模板残留检测
│   ├── copy_template.sh   #    模板初始化
│   ├── benchmarks/        #    基准测试
│   ├── tutorials/         #    教程
│   └── utils/             #    工具函数
│
├── repos/                 # 第三方仓库（已 clone 31 个）
├── data/                  # 公开数据集 (PBMC 3k 等)
└── docs-viewer/           # 🖥️ 笔记浏览器 (React + Vite)

当前状态：**26 篇深度笔记** | 31 个已克隆仓库 | 自动部署 GitHub Pages
```

---

## 📊 笔记进度

| 状态 | 数量 | 含义 |
|------|:----:|------|
| `✅ done` | **26** | 已撰写深度内容（含完整架构、创新分析、代码速览） |
| `🔶 metadata` | **24** | 仅有论文元数据（标题/年份/期刊等） |
| `⬜ template` | **128** | 空白模板，等待填写 |

### 已完成的 26 篇笔记一览

| 分类 | 笔记 | 
|------|------|
| **FM (16)** | Geneformer, scGPT, scFoundation, scBERT, scPoli, scPRINT, Nicheformer, Novae, UCE, SATURN, LangCell, GeneCompass, EpiAgent, Visual-Omics FM, xTrimoGene, scLong, CellFM |
| **FM+LLM (4)** | Cell2Sentence, CELLama, CASSIA, scChat |
| **Perturbation (3)** | Tahoe-100M/x1, STATE, PertAdapt |
| **Virtual Cell (2)** | Virtual Cell Challenge, The Virtual Cell |
| **Benchmark (1)** | Virtual Cell Challenge |

---

## 🏗️ 笔记系统

### Frontmatter 标签

每篇笔记文件头包含 YAML frontmatter，用于自动发现和管理：

```yaml
---
status: done    # template | metadata | done
filled: 2026-05-27
---
```

### 撰写新笔记

```bash
# 1. 准备结构化数据 (JSON)
# 2. 一条命令生成完整笔记（自动校验残留 + 更新前端配置）
python scripts/generate_note.py data/note.json

# 3. 构建验证
cd docs-viewer && npm run build
```

### 模板残留检测

```bash
python scripts/sync_notes_config.py --validate
```

---

## 🔥 关键论文速览（2023-2025）

### 路线一：单细胞转录组基础模型

| 模型 | 发表 | 参数量 | 架构 | 核心思路 | 推荐 |
|------|------|--------|------|---------|------|
| **Geneformer** | Nature 2023 | 6.5M | Transformer 6层 | 基因按表达量排序做 token | ⭐ 最易上手 |
| **scGPT** | Nature Methods 2024 | ~50M | GPT (Causal) | 基因 pair token，支持多模态 | ⭐ 首选复现 |
| **scFoundation** | Nature Methods 2024 | **3B** | 非对称 AE | 超大模型，引入 GO 先验 | 了解思想 |
| **scBERT** | Nat Mach Intell 2022 | ~110M | BERT | 最早将 BERT 引入单细胞 | 经典必读 |
| **UCE** | Nature 2024 | - | 对比学习 | 跨物种通用细胞嵌入 | 拓展阅读 |

---

## 🗺️ 学习路线

```
第1步（1-2周）：读论文 + 写笔记
  ├─ 浏览 REPO_MAP.md 了解可用仓库
  └─ 用 generate_note.py 撰写笔记

第2步（2-3周）：跑代码
  ├─ scGPT → repos/scGPT/ 下跑通 Tutorial
  └─ Geneformer → repos/Geneformer/ 下跑推理

第3步（1周+）：结合空间组学
  └─ scGPT embedding + squidpy 做空间聚类
```

每篇笔记搞懂这三点：
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
# scGPT (已 clone 到 repos/)
cd repos/scGPT
conda run -n zf-li23 pip install -e .
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

---

## 🌐 部署到 GitHub Pages

笔记浏览器已适配 GitHub Pages 自动部署，查看详细指南：

- **[docs-viewer/DEPLOY.md](docs-viewer/DEPLOY.md)** — 完整部署指南

快速 4 步上线：

1. 打开 GitHub 仓库 → **Settings** → **Pages** → **Source** 选 `GitHub Actions`
2. `git push origin main`
3. 等 1-2 分钟 GitHub Actions 跑完
4. 访问 `https://<你的用户名>.github.io/VirtualCell/`

本地预览：

```bash
cd docs-viewer
npm install
npm run dev          # 开发 → localhost:5173
BASE=/VirtualCell/ npm run build && npx vite preview  # 预览生产构建
```

---

*有问题随时在工作区记录。最后更新：2026-05-18*
