## 🏆 优先级 S — 必读经典

这些是被广泛引用/使用的基础模型，建议优先收集：

| # | 工作 | 发表 | 理由 | 笔记目录 |
|---|------|------|------|---------|
| 1 | **CellPLM** | ICLR 2024 | 最早将预训练推广到多组织的工作之一 | cell-plm |
| 2 | **scELMo** | bioRxiv 2023 | "ELMo 思路进单细胞"的代表作 | scelmo |
| 3 | **scMulan** | RECOMB 2024 | 多任务生成式预训练语言模型 | scmulan |
| 4 | **scPEFT** | Nat Mach Intell 2025 | 参数高效微调框架，实用价值高 | scpeft |
| 5 | **scGPT-spatial** | bioRxiv 2025 | scGPT 的空间扩展 | scgpt-spatial |
| 6 | **scNET** | Nat Methods 2025 | 基因表达 + PPI 网络融合 | scnet |
| 7 | **Tabula** | NeurIPS 2025 | 表格数据自监督范式 | tabulam |

## 🔥 优先级 A — 重要扰动/基准

这些代表关键方向的 critical paper 或重要基准：

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 8 | **CINEMA-OT** | Nat Methods 2023 | 因果扰动推断经典方法 |
| 9 | **scLAMBDA** | bioRxiv 2024 | 多基因扰动预测 |
| 10 | **Systema** | Nat Biotech 2025 | 扰动预测评估框架 |
| 11 | **"PCA still rules"** | bioRxiv 2024 | **重要的批评性论文**——质疑深度学习扰动预测 |
| 12 | **Perturbation linear baselines** | Nat Methods 2025 | 同样是批评性论文，揭示线性基线不比深度模型差 |
| 13 | **Zero-shot limitations** | Genome Bio 2025 | 揭示 scFM 零样本局限的重要评估 |
| 14 | **Metric Mirages** | bioRxiv 2024 | **关键批评**——细胞嵌入评估中的度量幻觉 |
| 15 | **Biology-driven insights** | Genome Bio 2025 | 系统性分析 scFM 学到了什么生物知识 |
| 16 | **scGenePT** | bioRxiv 2024 | 用 LLM 文本嵌入做扰动预测 |
| 17 | **PerturBench** | arXiv 2025 | 扰动分析基准 |
| 18 | **scDrugMap** | Nat Comms 2025 | 药物响应基准 |
| 19 | **HEIMDALL** | bioRxiv 2025 | Tokenization 系统性框架 |

## 🚀 优先级 B — 前沿新方向

2025-2026 最新热点：

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 20 | **scPRINT-2** | bioRxiv 2025 | scPRINT 下一代（已有仓库） |
| 21 | **OmniCell** | bioRxiv 2025 | 统一单细胞+空间的尝试 |
| 22 | **CellTok** | bioRxiv 2025 | Cell2Sentence 升级版 |
| 23 | **CellForge** | bioRxiv 2025 | Agentic 虚拟细胞设计 |
| 24 | **VCWorld** | bioRxiv 2025 | 生物世界模型 |
| 25 | **Scouter** | Nat Comput Sci 2025 | LLM 嵌入做扰动预测 |
| 26 | **TranscriptFormer** | bioRxiv 2025 | 跨物种生成式细胞图谱 |

## 📊 优先级 C — 病理基础模型

| # | 工作 | 发表 |
|---|------|------|
| 27 | **CONCH** | Nat Med 2024 | 视觉-语言病理 FM |
| 28 | **UNI** | Nat Med 2024 | 通用病理 FM |
| 29 | **Whole-slide FM (Prov-GigaPath)** | Nature 2024 |

## 📝 优先级 D — 综述（有总结价值）

现有 surveys 下 24 篇全是模板，建议优先写：
- **"Transformers in single-cell omics"** (Nat Methods 2024) — 被引最多
- **"How to build the virtual cell with AI"** (Cell 2024) — 虚拟细胞纲领
- **"Towards multimodal foundation models in molecular cell biology"** (Nature 2025)
- **"Harnessing foundation models in single-cell omics"** (Nat Rev Mol Cell Bio 2024)
- **"Interpretation, extrapolation and perturbation of single cells"** (Nat Rev Genet 2026)

---

## 🆕 优先级 E — 下一步推荐（基于 papers.md 对比）

经过对 `papers.md` 的全面对比，以下是我们尚未覆盖但值得优先收集的重要项目：

### E1 — FM 基础模型（有代码仓库，推荐优先）

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 1 | **GeneJepa** 🆕 | bioRxiv 2025 | 转录组的预测世界模型，概念新颖 |
| 2 | **Stack** 🆕 | bioRxiv 2026 | In-Context Learning 单细胞范式 |
| 3 | **PULSAR** 🆕 | bioRxiv 2025 | 多尺度多细胞生物学基础模型 |
| 4 | **CAPTAIN** | bioRxiv 2025 | 多模态 RNA + 蛋白（BMS） |
| 5 | **ChromFound** 🆕 | NeurIPS 2025 | 染色质可及性通用 FM |
| 6 | **scConcept** 🆕 | bioRxiv 2025 | 对比预训练，技术无关表示 |
| 7 | **Cell-GraphCompass** 🆕 | NSR 2024 | 图结构基础模型 |
| 8 | **Bidirectional Mamba** 🆕 | bioRxiv 2025 | Mamba 架构进单细胞 |
| 9 | **scHyena** 🆕 | bioRxiv 2023 | Hyena 架构全长度 RNA |
| 10 | **SCARF** 🆕 | BioRxiv 2025 | scATAC + scRNA 联合 FM |
| 11 | **EpiFoundation / EpiAgent** | bioRxiv 2025 | 表观基因组 FM |
| 12 | **xTrimoGene** | NeurIPS 2023 | 高效表示学习 |
| 13 | **Scaling Dense Representations** 🆕 | bioRxiv 2024 | 全转录组尺度上下文 |
| 14 | **GeneCompass** | Cell Res 2024 | 知识引导跨物种 FM |
| 15 | **CancerFoundation** | bioRxiv 2024 | 耐药机制解密 |

### E2 — FM + LLM（LLM 辅助单细胞分析）

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 1 | **OKR-Cell** 🆕 | bioRxiv 2026 | 开放世界知识辅助跨模态预训练 |
| 2 | **spEMO** 🆕 | Nat Biomed Eng 2026 | 多模态空间 + 病理分析 |
| 3 | **CellHermes** 🆕 | bioRxiv 2025 | 多模态组学理解 |
| 4 | **TISSUENARRATOR** 🆕 | bioRxiv 2025 | 空间转录组 + LLM 生成 |
| 5 | **scReader** | arXiv 2024 | LLM 解读 scRNA-seq |
| 6 | **sciLaMA** | ICML 2025 | LLM 先验知识注入 |
| 7 | **TEDDY** 🆕 | ICML Workshop 2025 | 基础模型家族 |
| 8 | **Scaling LLM for sc** | bioRxiv 2025 | LLM 扩展到单细胞 |

### E3 — 扰动预测与评估

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 1 | **STATE** 🆕 | bioRxiv 2025 | Arc Institute，跨条件扰动预测 |
| 2 | **scFME (PertEval)** 🆕 | ICML 2024 | 扰动预测方法评估 |
| 3 | **Diffusion Debiasing** 🆕 | PNAS 2025 | 扩散模型去偏扰动预测 |
| 4 | **SpatialProp** 🆕 | bioRxiv 2025 | 空间扰动建模 |
| 5 | **In Silico Discovery** | Nat Comput Sci 2025 | 大规模扰动发现（已有模板） |

### E4 — 基准与评估（新增批评/评估论文）

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 1 | **Systematic Comparison of Perturbation Models** 🆕 | bioRxiv 2026 | 最全面扰动模型对比 |
| 2 | **Unified Framework for Benchmarking scFMs** 🆕 | bioRxiv 2026 | scFM 统一部署与基准 |
| 3 | **Defining Open Problems in sc Analysis** 🆕 | Nat Biotech 2025 | 单细胞开放问题定义 |
| 4 | **Benchmarking Perturbation Algorithms** 🆕 | Nat Methods 2025 | 可泛化扰动预测基准 |
| 5 | **Batch Effects Remain Barrier** 🆕 | bioRxiv 2025 | **关键发现**——批次效应仍是根本障碍 |
| 6 | **Deep Learning Models Do Outperform** 🆕 | bioRxiv 2025 | 对 PCA still rules 的反驳 |
| 7 | **BMFM-RNA** 🆕 | arXiv 2025 | 开放转录组 FM 构建框架 |
| 8 | **AnnDictionary** | Nat Comms 2025 | LLM 注释基准（已有模板） |

### E5 — 虚拟细胞

| # | 工作 | 发表 | 理由 |
|---|------|------|------|
| 1 | **Virtual Cells: Predict, Explain, Discover** 🆕 | bioRxiv 2025 | 虚拟细胞新纲领 |
| 2 | **The Virtual Cell** 🆕 | Nat Methods 2025 | 同一方向的方法论文 |
| 3 | **AI Virtual Organoids (AIVO)** 🆕 | Bioactive Materials 2026 | 前沿交叉方向 |
| 4 | **Virtual Cell Challenge** 🆕 | Cell 2025 | Arc Institute 基准竞赛 |

---

> 📌 完成状态：**S/A/B/C/D 共 34 篇全部完成**。下一步建议从 E1 和 E4 入手（有代码仓库的模型 + 高影响力的批评/基准论文）。
