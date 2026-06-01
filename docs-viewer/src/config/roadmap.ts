export interface RoadmapItem {
  noteId: string
  label: string
  category: string
  optional?: boolean
}

export interface RoadmapRoute {
  title: string
  description: string
  category: string
  items: RoadmapItem[]
}

/**
 * 学习路线图配置 — 按 11 个分类组织，标注学习依赖关系。
 */
export const roadmapRoutes: RoadmapRoute[] = [
  {
    title: '🏛️ FM + 经典语言模型',
    description: 'BERT/GPT 等经典 Transformer 架构在单细胞上的适配，构成技术主干',
    category: 'FM + 经典语言模型',
    items: [
      { noteId: 'scbert', label: '⭐ scBERT — 首个 BERT 进单细胞的开山之作', category: 'FM + 经典语言模型' },
      { noteId: 'geneformer', label: '⭐🔥 Geneformer — Rank Encoding + 30M 细胞', category: 'FM + 经典语言模型' },
      { noteId: 'scgpt', label: '⭐🔥 scGPT — GPT + 基因对 Tokenization，最高引用', category: 'FM + 经典语言模型' },
      { noteId: 'scfoundation', label: '🏆 scFoundation — 非对称 AE + Performer，300M~1B', category: 'FM + 经典语言模型' },
      { noteId: 'cellfm', label: '🔥 CellFM — 1 亿细胞预训练，最大公开 FM 之一', category: 'FM + 经典语言模型' },
    ],
  },
  {
    title: '🧫 FM + 空间组学',
    description: '将细胞放入组织微环境上下文',
    category: 'FM + 空间组学',
    items: [
      { noteId: 'nicheformer', label: '⭐ Nicheformer — BERT + 空间邻域 (Nat Methods 2025)', category: 'FM + 空间组学' },
      { noteId: 'novae', label: '⭐🔥 Novae — GNN + SwAV 空间域分割 (Nat Methods 2025)', category: 'FM + 空间组学' },
      { noteId: 'visual-omics-fm', label: '🎯 Visual-Omics FM — H&E + ST 跨模态融合', category: 'FM + 空间组学' },
    ],
  },
  {
    title: '🌌 FM + 世界模型',
    description: '预测细胞状态内在动态规律的下一代范式',
    category: 'FM + 世界模型',
    items: [
      { noteId: 'genejepa', label: '🎯 GeneJEPA — Perceiver + VICReg 表示预测世界模型', category: 'FM + 世界模型' },
    ],
  },
  {
    title: '🔗 FM + 跨物种/通用嵌入',
    description: '跨物种对齐与通用细胞表示',
    category: 'FM + 跨物种/通用嵌入',
    items: [
      { noteId: 'cell-atlas-fm', label: '⭐ UCE — 跨物种通用细胞嵌入 (Nature 2024)', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'saturn', label: '⭐ SATURN — DNA 序列引导跨物种零样本', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'genecompass', label: '🏆 GeneCompass — GO/KEGG 知识注入跨物种 FM', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'transcriptformer', label: '🏆 TranscriptFormer — 跨物种生成式细胞图谱', category: 'FM + 跨物种/通用嵌入' },
    ],
  },
  {
    title: '🕸️ FM + 图与网络',
    description: '利用图结构（PPI/GO/共表达）建模细胞调控网络',
    category: 'FM + 图与网络',
    items: [
      { noteId: 'scnet', label: '⭐ scNET — PPI 网络 + 表达数据融合', category: 'FM + 图与网络' },
      { noteId: 'gears', label: '⭐🔥 GEARS — GO 图传播扰动效应 (Nat Biotech 2023)', category: 'FM + 图与网络' },
      { noteId: 'cell-plm', label: '🎯 CellPLM — 细胞-细胞图编码器', category: 'FM + 图与网络' },
    ],
  },
  {
    title: '💬 FM + LLM',
    description: '大语言模型与单细胞数据的交叉融合',
    category: 'FM + LLM',
    items: [
      { noteId: 'cell2sentence', label: '⭐ Cell2Sentence — 细胞→句子→LLM (ICML 2024)', category: 'FM + LLM' },
      { noteId: 'scelmo', label: '🎯 scELMo — LLM 基因描述嵌入增强单细胞分析', category: 'FM + LLM' },
      { noteId: 'cassia', label: '🔥 CASSIA — 多智能体自动注释 (Nat Comms 2025)', category: 'FM + LLM' },
      { noteId: 'scchat', label: '🔥 scChat — LLM Co-Pilot 交互式分析', category: 'FM + LLM' },
    ],
  },
  {
    title: '🧪 遗传扰动',
    description: '预测基因/药物扰动后的细胞状态改变',
    category: '遗传扰动',
    items: [
      { noteId: 'cpa', label: '⭐ CPA — 加性解耦 VAE + 剂量响应 (2023)', category: '遗传扰动' },
      { noteId: 'cinema-ot', label: '⭐ CINEMA-OT — 因果最优传输扰动推断', category: '遗传扰动' },
      { noteId: 'systema', label: '🔥 Systema — 扰动预测系统性评估框架', category: '遗传扰动' },
      { noteId: 'state', label: '🎯 STATE — 跨上下文扰动预测 (Arc Institute)', category: '遗传扰动' },
      { noteId: 'tahoe-100m', label: '🏆 Tahoe-100M — 十亿级扰动图谱', category: '遗传扰动' },
    ],
  },
  {
    title: '📊 评估与 Benchmark',
    description: '关键性评估与批评性视角',
    category: '评估与 Benchmark',
    items: [
      { noteId: 'perturbench', label: '🔥 PerturBench — 扰动分析标准基准', category: '评估与 Benchmark' },
      { noteId: 'metric-mirages', label: '📊 Metric Mirages — 细胞嵌入的度量幻觉', category: '评估与 Benchmark' },
      { noteId: 'zero-shot-limitations', label: '📊 Zero-shot 局限 — scFM 泛化边界在哪', category: '评估与 Benchmark' },
      { noteId: 'virtual-cell-challenge', label: '🎯 Virtual Cell Challenge — 细胞的图灵测试', category: '评估与 Benchmark' },
    ],
  },
  {
    title: '🔬 病理基础模型',
    description: '计算病理学基础模型',
    category: '病理基础模型',
    items: [
      { noteId: 'uni', label: '⭐🔥 UNI — 通用病理 FM (Nature Med 2024)', category: '病理基础模型' },
      { noteId: 'conch', label: '⭐ CONCH — 视觉-语言病理 FM (Nature Med 2024)', category: '病理基础模型' },
      { noteId: 'whole-slide-fm', label: '🏆 Whole-Slide FM — 全切片基础模型 (Nature 2024)', category: '病理基础模型' },
    ],
  },
  {
    title: '🌌 虚拟细胞',
    description: '所有基础模型努力的终极目标',
    category: '虚拟细胞',
    items: [
      { noteId: 'the-virtual-cell', label: '⭐ The Virtual Cell — 定义虚拟细胞蓝图', category: '虚拟细胞' },
      { noteId: 'vcworld', label: '🎯 VCWorld — 知识图谱+LLM 白盒模拟器', category: '虚拟细胞' },
      { noteId: 'cellforge', label: '🎯 CellForge — Agentic 虚拟细胞设计', category: '虚拟细胞' },
    ],
  },
  {
    title: '📚 综述与展望',
    description: '领域全景综述（优先阅读）',
    category: '综述与展望',
    items: [
      { noteId: 'transformers-sc-omics-review', label: '📝 Transformers in sc Omics — 引用最高的综述', category: '综述与展望' },
      { noteId: 'multimodal-fm-cell-biology', label: '📝 多模态 FM — Nature 2025 纲领性展望', category: '综述与展望' },
      { noteId: 'build-virtual-cell-ai', label: '📝 构建虚拟细胞 — 优先事项 (Cell 2024)', category: '综述与展望' },
    ],
  },
]
