# OpenBioMed 学习笔记 🔬

## 概述

**OpenBioMed** 是由 PharMolix 和清华大学智能产业研究院（AIR）联合开发的多模态生物医学 AI Agent 平台。与前面学习的单细胞或空间转录组基础模型不同，OpenBioMed 是一个**综合性工具包**，涵盖了**药物发现、蛋白质工程、单细胞组学分析**等多个领域，提供了 **45+ 个端到端技能（Skills）** 和 **20+ 种深度学习工具**。

OpenBioMed 的核心理念是：通过将先进的 AI 模型（如 BioMedGPT、PharMolixFM、LangCell 等）封装为可组合的工具，借助 Claude Code 的 Agent 能力，为生物医学研究者提供**自动化工作流**解决方案。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    OpenBioMed 平台                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │              45+ Skills (技能层)                 │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│   │  │药物发现   │  │蛋白质工程 │  │单细胞组学分析 │  │  │
│   │  │11个技能   │  │13个技能   │  │13个技能      │  │  │
│   │  └──────────┘  └──────────┘  └──────────────┘  │  │
│   │  ┌──────────┐  ┌──────────┐                    │  │
│   │  │数据检索   │  │工具技能   │                    │  │
│   │  │6个技能    │  │2个技能    │                    │  │
│   │  └──────────┘  └──────────┘                    │  │
│   └─────────────────────────────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│   ┌─────────────────────────────────────────────────┐  │
│   │            20+ Tools (工具层)                    │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│   │  │分子工具   │  │蛋白质工具 │  │单细胞工具    │  │  │
│   │  │QA/ADMET  │  │折叠/对接 │  │LangCell     │  │  │
│   │  │可视化    │  │突变分析  │  │scGPT/GF    │  │  │
│   │  └──────────┘  └──────────┘  └──────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
│                         │                               │
│                         ▼                               │
│   ┌─────────────────────────────────────────────────┐  │
│   │         Foundation Models (基础模型层)            │  │
│   │  BioMedGPT  │  PharMolixFM  │  LangCell         │  │
│   │  MutaPLM    │  MolT5/BioT5  │  DrugFM           │  │
│   └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 核心技术：LangCell（与细胞注释相关的核心模型）

LangCell 是 OpenBioMed 中**单细胞组学方向**的核心模型，也是当前工作区中唯一一个**多模态细胞基础模型**（将单细胞转录组与自然语言对齐）。

### 模型架构

LangCell 是一个**双编码器 + 跨模态匹配**架构：

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangCell 架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐         ┌──────────────────┐             │
│   │   Cell Encoder   │         │   Text Encoder   │             │
│   │   (BertModel)    │         │  (MedBertModel)  │             │
│   │   Geneformer     │         │  + cross-attn    │             │
│   └────────┬─────────┘         └────────┬─────────┘             │
│            │                            │                       │
│            ▼                            ▼                       │
│   ┌──────────────────┐         ┌──────────────────┐             │
│   │  LangCell_Pooler │         │  LangCell_Pooler │             │
│   │  Linear(768→256) │         │  Linear(768→256) │             │
│   │  L2 Normalize    │         │  L2 Normalize    │             │
│   └────────┬─────────┘         └────────┬─────────┘             │
│            │                            │                       │
│            └──────────┬─────────────────┘                       │
│                       ▼                                         │
│   ┌──────────────────────────────────────┐                      │
│   │      余弦相似度 (dot product)        │                      │
│   │      sim = cell_emb @ text_emb.T     │  ← 对比学习对齐      │
│   └──────────────────┬───────────────────┘                      │
│                      │                                          │
│                      ▼                                          │
│   ┌──────────────────────────────────────┐                      │
│   │   CTM Head (Cross-Transformer Match) │                      │
│   │   text_encoder(encoder_hidden_states │                      │
│   │     = cell_last_h, mode='multimodal')│  ← 细粒度匹配        │
│   │   Linear(768 → 2) → Softmax          │                      │
│   └──────────────────┬───────────────────┘                      │
│                      │                                          │
│                      ▼                                          │
│   ┌──────────────────────────────────────┐                      │
│   │   最终预测 = 0.1×sim + 0.9×ctm      │                      │
│   │   argmax over class candidates       │                      │
│   └──────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 核心组件详解

1. **Cell Encoder**：基于 Geneformer 的 BertModel
   - 输入：基因表达序列（Geneformer 的 Rank Value Encoding）
   - 编码器：标准的 BERT Transformer
   - Pooler：自定义 `LangCell_Pooler`，将 [CLS] token 的 hidden state 投影到 256 维并做 L2 归一化

2. **Text Encoder**：基于生物医学文本的 MedBertModel
   - 增加了特殊 token：`[DEC]`（bos_token）和 `[ENC]`（additional_special_token）
   - 支持 cross-attention 模式（用于 CTM）
   - Pooler：同 Cell Encoder，投影到 256 维

3. **对比学习对齐**（相似度计算）
   - 细胞嵌入和文本嵌入通过余弦相似度进行对比
   - 温度系数 τ = 0.05
   - `sim = cell_emb @ text_embs.T / 0.05`
   - Softmax 得到概率分布

4. **CTM（Cross-Transformer Matching）**
   - 将细胞编码器的最后一层隐藏状态作为 Text Encoder 的 cross-attention 输入
   - Text Encoder 在 `mode='multimodal'` 下融合细胞和文本信息
   - CTM Head 输出二分类 logits（匹配/不匹配），取匹配概率
   - 对每个候选类别分别计算 CTM 分数

5. **最终预测**
   - 加权融合：`0.1 × sim_logit + 0.9 × ctm_logit`
   - CTM 权重远高于简单相似度，说明细粒度匹配更为关键

### LangCell 特性

- **Zero-shot 细胞注释**：无需微调即可对未见过的细胞类型进行注释
- **Few-shot 能力**：只需少量标注数据即可获得良好性能
- **多模态理解**：利用文本描述中的知识（如细胞标记基因的文献描述）增强细胞理解
- **缓存机制**：缓存文本嵌入，避免重复编码

---

## 其他基础模型

OpenBioMed 集成了多种生物医学基础模型：

### BioMedGPT 系列

| 模型 | 参数量 | 特点 |
|------|--------|------|
| BioMedGPT-LM-7B | 7B | 基于 Llama-2 的生物医学语言模型 |
| BioMedGPT-10B | 10B | 多模态，对齐分子/蛋白质/语言 |
| BioMedGPT-R1-17B | 17B | 基于 DeepSeek-R1-Distill-Qwen-14B 的推理模型 |
| BioMedGPT-Mol | - | 分子理解与生成多模态模型 |

BioMedGPT 对齐了三种模态：
```
分子结构 (SMILES/Graph) ──┐
                          ├──→ 统一语义空间 ←── 自然语言
蛋白质序列 (氨基酸) ──────┘
```

### PharMolixFM

- **全原子分子基础模型**（All-Atom Foundation Model）
- 统一建模小分子、抗体、蛋白质
- 使用非自回归多模态生成模型
- 能力：分子对接、结构药物设计、多肽设计、构象生成
- 对接性能（给定口袋）RMSD < 2Å 达 83.9

### MutaPLM

- **蛋白质突变解释与工程**模型
- 输入：蛋白质序列 + 突变位置信息
- 输出：突变效应的文本解释
- 应用：突变设计（AAV、GFP 等高通量优化）

---

## 技能体系详解

### 💊 药物发现 (11个技能)

| 技能 | 核心工具 | 说明 |
|------|----------|------|
| drug-candidate-discovery | MolCRAFT + 对接 | 靶向药物分子生成 |
| drug-lead-analysis | GraphMVP | 药物相似性分析 |
| target-based-lead-design | MolCRAFT + 对接 | 基于结构的先导化合物设计 |
| admet-prediction | GraphMVP 集成 | ADMET 性质预测 |
| retrosynthesis-planning | AiZynthFinder | 逆合成路线规划 |
| text-based-molecule-editing | MolT5/BioT5 | 基于自然语言的分子编辑 |

### 🧬 蛋白质工程 (13个技能)

| 技能 | 核心工具 | 说明 |
|------|----------|------|
| protein-mutation-analysis | MutaPLM + ESMFold | 突变效应分析 |
| functional-protein-design | CodeFP + GO | 功能蛋白设计 |
| antibody-structure-prediction | tFold | 抗体结构预测 |
| antibody-design-iggm | IgGM | 抗体设计 |
| structure-prediction | Boltz-2 | 复合物结构预测 |
| binding-affinity-prediction | Prodigy | 结合亲和力预测 |

### 🔬 单细胞组学 (13个技能)

| 技能 | 核心模型 | 说明 |
|------|----------|------|
| single-cell-geneformer | Geneformer | scRNA-seq 分析 |
| single-cell-langcell | LangCell | 细胞注释（零样本/少样本） |
| single-cell-scgpt | scGPT | 参考映射、嵌入提取 |
| spatial-stofm | SToFM | 空间转录组分析 |
| scanpy-analysis | Scanpy | 标准 scRNA-seq 流程 |
| multi-omics-scvi | scVI/scANVI/totalVI | 多组学整合 |
| cellxgene-query | CELLxGENE Census | 查询 6100万+ 细胞 |
| atac-seq-processing | MACS2 | ATAC-seq 分析 |

---

## 数据处理流程

OpenBioMed 定义了清晰的数据模态抽象：

```
Molecule (SMILES/分子图) ──→ 性质预测/QA/可视化
Protein (氨基酸序列) ──────→ 折叠/对接/突变分析
Pocket (结合口袋) ────────→ 结构药物设计
Text (自然语言) ──────────→ QA/报告生成
Cell (基因表达序列) ──────→ 细胞注释 (LangCell)
```

每种数据类型都对应了特定的 featurizer、collator 和 model pipeline。

---

## 与当前工作区其他模型的关系

| 模型 | 在 OpenBioMed 中的角色 |
|------|----------------------|
| **Geneformer** | LangCell 的细胞编码器基座模型 |
| **scGPT** | 独立的单细胞技能 |
| **LangCell** | **OpenBioMed 核心模型**，双编码器细胞-文本匹配 |
| **BioMedGPT** | 跨模态生物医学问答 |
| **PharMolixFM** | 全原子分子基础模型 |
| **MutaPLM** | 蛋白质突变解释 |

---

## 优缺点

**优点**：
- ✅ 涵盖面极广：药物、蛋白质、单细胞、数据检索全覆盖
- ✅ Agent 驱动：借助 Claude Code 实现自动化工作流
- ✅ 45+ 现成技能，可直接用于科研
- ✅ 多模态对齐（分子+蛋白质+文本+细胞）
- ✅ LangCell 支持零样本细胞注释

**局限性**：
- ❌ 依赖外部工具和服务（Claude Code、数据库 API）
- ❌ 安装配置复杂（conda + pip + 多个外部工具）
- ❌ 各技能的成熟度不统一（部分标记为 MVP）
- ❌ 计算资源需求高（BioMedGPT-R1 需要 17B 参数显存）
- ❌ 对 Python 版本和 CUDA 版本有严格限制

---

## 安装指南

```bash
# 基础环境
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117

# PyG
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0

# 安装包
pip install -e .

# 可视化（可选）
conda install -c conda-forge pymol-open-source

# LangCell 依赖
pip install geneformer
```

---

## 参考资源

- GitHub (主仓库): [https://github.com/PharMolix/OpenBioMed](https://github.com/PharMolix/OpenBioMed)
- 在线平台: [http://openbiomed.pharmolix.com](http://openbiomed.pharmolix.com)
- HuggingFace: [https://huggingface.co/PharMolix](https://huggingface.co/PharMolix)
- LangCell 论文: Zhao et al., 2024. [arXiv: 2405.06708](https://arxiv.org/abs/2405.06708)
- BioMedGPT 论文: Luo et al., 2024. *IEEE JBHI*
- PharMolixFM 论文: Luo et al., 2025. [arXiv: 2503.21788](https://arxiv.org/abs/2503.21788)
