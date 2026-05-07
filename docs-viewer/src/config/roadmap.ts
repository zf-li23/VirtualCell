export interface RoadmapItem {
  noteId: string
  label: string
  optional?: boolean
}

export interface RoadmapRoute {
  title: string
  description: string
  items: RoadmapItem[]
}

/**
 * 学习路线图配置 — 覆盖基础、核心、空间、LLM、扰动、前沿等方向。
 *
 * 选入标准：
 * 1. ⭐ 路线早期奠基工作（chronological significance）
 * 2. 🏆 高性能 / 大规模模型（performance / scale）
 * 3. 🎯 代表性 / 范式开创模型（representative / paradigm-shifting）
 * 4. 🔥 广泛使用的高引用模型（widely used / cited）
 */
export const roadmapRoutes: RoadmapRoute[] = [
  // ──────────────────────────────────────────────
  // 1️⃣ 溯源：早期探索 (2022-2023)
  // ──────────────────────────────────────────────
  {
    title: '🕰️ 溯源：早期探索 (2022-2023)',
    description: '奠基工作，理解细胞模型从何而来——从 BERT 迁移到单细胞的第一步',
    items: [
      { noteId: 'scbert', label: '⭐ scBERT — 首个将 BERT 引入 scRNA-seq 的开山之作' },
      { noteId: 'geneformer', label: '⭐🔥 Geneformer — Rank Encoding + 30M 细胞，最广泛使用的基础模型' },
      { noteId: 'scpoli', label: '⭐ scPoli — 条件 VAE + 参考映射，开创迁移学习范式' },
      { noteId: 'xtrimogene', label: '🏆 xTrimoGene — 非对称 AE 架构先驱，scFoundation 的前身 (NeurIPS 2023)' },
    ],
  },
  // ──────────────────────────────────────────────
  // 2️⃣ 奠基：BERT 与表示学习
  // ──────────────────────────────────────────────
  {
    title: '🏛️ 奠基：BERT 与表示学习',
    description: '深度理解细胞嵌入学习——跨物种、跨数据集的通用表示',
    items: [
      { noteId: 'cell-atlas-fm', label: '⭐ UCE / Cell Atlas — 跨物种通用细胞嵌入 (Nature 2024)' },
      { noteId: 'saturn', label: '⭐ SATURN — DNA 序列引导的基因嵌入，跨物种零样本 (Nat Methods 2024)' },
      { noteId: 'genecompass', label: '🏆 GeneCompass — GO/KEGG 知识注入，跨物种调控推断 (Cell Res 2024)' },
      { noteId: 'scprint', label: '🔥 scPRINT — 5000 万细胞预训练，注意力权重→基因调控网络 (Nat Comms 2025)' },
    ],
  },
  // ──────────────────────────────────────────────
  // 3️⃣ 浪潮：GPT 与生成式革命
  // ──────────────────────────────────────────────
  {
    title: '🌊 浪潮：GPT 与生成式革命',
    description: '生成式预训练 + 大规模模型——当前单细胞 FM 的主流范式',
    items: [
      { noteId: 'scgpt', label: '⭐🔥 scGPT — GPT + 基因对 Tokenization，最高引用量的单细胞 FM (Nat Methods 2024)' },
      { noteId: 'scfoundation', label: '🏆 scFoundation — 非对称 AE + Performer 线性注意力，300M~1B 参数 (Nat Methods 2024)' },
      { noteId: 'sclong', label: '🏆 scLong — 十亿参数，捕捉长距离基因上下文 (2024)' },
      { noteId: 'cellfm', label: '🔥 CellFM — 1 亿细胞预训练，最大的公开单细胞 FM 之一 (Nat Comms 2025)' },
    ],
  },
  // ──────────────────────────────────────────────
  // 4️⃣ 空间：从单细胞到组织微环境
  // ──────────────────────────────────────────────
  {
    title: '🧫 空间：从单细胞到组织微环境',
    description: '将细胞放入空间上下文——空间转录组基础模型',
    items: [
      { noteId: 'nicheformer', label: '⭐ Nicheformer — BERT + 空间邻域上下文，sc→ST 桥梁 (Nat Methods 2025)' },
      { noteId: 'novae', label: '⭐🔥 Novae — 图自监督 SwAV 空间域分割，实际应用最多的空间 FM (Nat Methods 2025)' },
      { noteId: 'visual-omics-fm', label: '🎯 Visual-Omics FM — H&E + ST 跨模态融合 (Nat Methods 2025)' },
      { noteId: 'scgpt', label: 'scGPT-spatial (延展) — scGPT 空间持续预训练', optional: true },
    ],
  },
  // ──────────────────────────────────────────────
  // 5️⃣ 对话：细胞与语言模型
  // ──────────────────────────────────────────────
  {
    title: '💬 对话：细胞与语言模型',
    description: 'LLM 时代的细胞生物学——用自然语言理解细胞',
    items: [
      { noteId: 'cell2sentence', label: '⭐ Cell2Sentence — 将细胞"翻译"成句子，用 LLM 建模 (ICML 2024)' },
      { noteId: 'cellama', label: '🎯 CELLama — LLM 驱动的细胞嵌入，同时支持 sc 和 ST (2024)' },
      { noteId: 'cassia', label: '🎯 CASSIA — 多智能体 LLM，自动细胞注释 (Nat Comms 2025)' },
      { noteId: 'scchat', label: '🔥 scChat — LLM Co-Pilot 用于单细胞分析 (2024)' },
    ],
  },
  // ──────────────────────────────────────────────
  // 6️⃣ 扰动：虚拟实验
  // ──────────────────────────────────────────────
  {
    title: '🧪 扰动：虚拟细胞实验',
    description: '预测基因扰动后的细胞状态——药物发现与机制研究',
    items: [
      { noteId: 'tahoe-100m', label: '🏆 Tahoe-100M — 十亿级扰动图谱，最大规模 (2025)' },
      { noteId: 'state', label: '🎯 STATE — 跨上下文扰动响应预测 (2025)' },
      { noteId: 'pertadapt', label: '🎯 PertAdapt — 将 scFM 适配到扰动预测 (2025)' },
      { noteId: 'scprint', label: '⭐ scPRINT（扰动篇）— 注意力权重推断 GRN', optional: true },
    ],
  },
  // ──────────────────────────────────────────────
  // 7️⃣ 前沿：新模态与新架构
  // ──────────────────────────────────────────────
  {
    title: '🔬 前沿：新模态与新架构',
    description: '突破转录组的边界——表观组、蛋白组、新架构',
    items: [
      { noteId: 'epiagent', label: '⭐ EpiAgent — scATAC-seq 首个表观基因组 FM (Nat Methods 2025)' },
      { noteId: 'visual-omics-fm', label: '🎯 视觉-组学融合 — H&E ↔ ST 双向预测', optional: true },
      { noteId: 'novae', label: 'Novae (图架构) — SwAV 在图结构上的应用', optional: true },
    ],
  },
  // ──────────────────────────────────────────────
  // 8️⃣ 愿景：虚拟细胞
  // ──────────────────────────────────────────────
  {
    title: '🌌 愿景：虚拟细胞',
    description: '概念性 / 展望——所有基础模型努力的终极目标',
    items: [
      { noteId: 'virtual-cell-challenge', label: '🎯 Virtual Cell Challenge — Turing Test for Cells (Cell 2025)' },
      { noteId: 'the-virtual-cell', label: '⭐ The Virtual Cell — 定义虚拟细胞蓝图 (Nat Methods 2025)' },
      { noteId: 'scgpt', label: '回顾：scGPT — 从数据到虚拟细胞的桥梁', optional: true },
    ],
  },
]
