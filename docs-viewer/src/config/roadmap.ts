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
    description: 'BERT/GPT/ELMo 等经典 Transformer 架构在单细胞数据上的适配',
    category: 'FM + 经典语言模型',
    items: [
      { noteId: 'scbert', label: '⭐ scBERT — 首个 BERT 迁移到 scRNA-seq 的开山之作', category: 'FM + 经典语言模型' },
      { noteId: 'geneformer', label: '⭐🔥 Geneformer — Rank Encoding + 30M 细胞', category: 'FM + 经典语言模型' },
      { noteId: 'scgpt', label: '⭐🔥 scGPT — GPT + 基因对 Tokenization，最高引用量', category: 'FM + 经典语言模型' },
      { noteId: 'scfoundation', label: '🏆 scFoundation — 非对称 AE + Performer，300M~1B 参数', category: 'FM + 经典语言模型' },
      { noteId: 'scmulan', label: '🏆 scMulan — 多任务生成式预训练语言模型', category: 'FM + 经典语言模型' },
      { noteId: 'cellfm', label: '🔥 CellFM — 1 亿细胞预训练的最大公开 FM 之一', category: 'FM + 经典语言模型' },
      { noteId: 'sclong', label: '🏆 scLong — 十亿参数，长距离基因上下文', category: 'FM + 经典语言模型' },
      { noteId: 'xtrimogene', label: '🏆 xTrimoGene — 非对称 AE 架构先驱', category: 'FM + 经典语言模型' },
      { noteId: 'scelmo', label: '🎯 scELMo — ELMo 风格嵌入的单细胞适配', category: 'FM + 经典语言模型' },
      { noteId: 'heimdall', label: '🎯 Heimdall — Tokenization 系统框架', category: 'FM + 经典语言模型' },
      { noteId: 'scpeft', label: '🔥 scPEFT — 参数高效微调框架', category: 'FM + 经典语言模型' },
      { noteId: 'epiagent', label: '⭐ EpiAgent — 表观基因组首个 FM', category: 'FM + 经典语言模型' },
    ],
  },
  {
    title: '🧫 FM + 空间组学',
    description: '将细胞放入空间上下文——空间转录组基础模型',
    category: 'FM + 空间组学',
    items: [
      { noteId: 'nicheformer', label: '⭐ Nicheformer — BERT + 空间邻域，sc→ST 桥梁', category: 'FM + 空间组学' },
      { noteId: 'novae', label: '⭐🔥 Novae — 图自监督 SwAV 空间域分割', category: 'FM + 空间组学' },
      { noteId: 'scgpt-spatial', label: '🎯 scGPT-spatial — scGPT 空间持续预训练', category: 'FM + 空间组学' },
      { noteId: 'visual-omics-fm', label: '🎯 Visual-Omics FM — H&E + ST 跨模态融合', category: 'FM + 空间组学' },
      { noteId: 'omnicell', label: '🏆 OmniCell — 统一 sc + 空间编码器', category: 'FM + 空间组学' },
    ],
  },
  {
    title: '🌌 FM + 世界模型',
    description: '生成式 / 预测式世界模型范式',
    category: 'FM + 世界模型',
    items: [
      { noteId: 'genejepa', label: '🎯 GeneJEPA — Perceiver + VICReg 表示预测', category: 'FM + 世界模型' },
      { noteId: 'transcriptformer', label: '🏆 TranscriptFormer — 跨物种生成式细胞图谱', category: 'FM + 世界模型' },
      { noteId: 'scprint', label: '🔥 scPRINT — Performer + 5000 万细胞 + GRN 推断', category: 'FM + 世界模型' },
      { noteId: 'scprint-2', label: '🔥 scPRINT-2 — 下一代细胞基础模型', category: 'FM + 世界模型' },
    ],
  },
  {
    title: '🔗 FM + 跨物种/通用嵌入',
    description: '跨物种对齐和通用细胞表示学习',
    category: 'FM + 跨物种/通用嵌入',
    items: [
      { noteId: 'cell-atlas-fm', label: '⭐ UCE — 跨物种通用细胞嵌入 (Nature 2024)', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'saturn', label: '⭐ SATURN — DNA 序列引导跨物种零样本', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'genecompass', label: '🏆 GeneCompass — GO/KEGG 知识注入跨物种 FM', category: 'FM + 跨物种/通用嵌入' },
      { noteId: 'cell-plm', label: '🎯 CellPLM — 细胞间关系编码', category: 'FM + 跨物种/通用嵌入' },
    ],
  },
  {
    title: '🕸️ FM + 图与网络',
    description: '图结构 / 网络生物学方法',
    category: 'FM + 图与网络',
    items: [
      { noteId: 'scnet', label: '⭐ scNET — PPI 网络 + 表达数据融合', category: 'FM + 图与网络' },
      { noteId: 'scpoli', label: '⭐ scPoli — cVAE + 群体水平整合', category: 'FM + 图与网络' },
      { noteId: 'tabulam', label: '🎯 Tabula — 表格自监督学习范式', category: 'FM + 图与网络' },
    ],
  },
  {
    title: '💬 FM + LLM',
    description: '大语言模型与单细胞数据的交叉',
    category: 'FM + LLM',
    items: [
      { noteId: 'cell2sentence', label: '⭐ Cell2Sentence — 细胞→句子→LLM', category: 'FM + LLM' },
      { noteId: 'cellama', label: '🎯 CELLama — LLM 驱动细胞嵌入', category: 'FM + LLM' },
      { noteId: 'celltok', label: '🎯 CellTok — 早期融合多模态 LLM', category: 'FM + LLM' },
      { noteId: 'langcell', label: '🏆 LangCell — 语言-细胞联合预训练', category: 'FM + LLM' },
      { noteId: 'cassia', label: '🔥 CASSIA — 多智能体自动注释', category: 'FM + LLM' },
      { noteId: 'scchat', label: '🔥 scChat — LLM Co-Pilot 分析', category: 'FM + LLM' },
      { noteId: 'scouter', label: '🎯 Scouter — LLM 嵌入做扰动预测', category: 'FM + LLM' },
    ],
  },
  {
    title: '🧪 遗传扰动',
    description: '预测基因/药物扰动后的细胞状态',
    category: '遗传扰动',
    items: [
      { noteId: 'gears', label: '⭐🔥 GEARS — GNN + GO 图传播扰动效应', category: '遗传扰动' },
      { noteId: 'cpa', label: '⭐ CPA — 加性解耦 VAE + 剂量响应', category: '遗传扰动' },
      { noteId: 'cinema-ot', label: '⭐ CINEMA-OT — 因果最优传输扰动推断', category: '遗传扰动' },
      { noteId: 'systema', label: '🔥 Systema — 扰动预测系统性评估框架', category: '遗传扰动' },
      { noteId: 'state', label: '🎯 STATE — 跨上下文扰动预测', category: '遗传扰动' },
      { noteId: 'tahoe-100m', label: '🏆 Tahoe-100M — 十亿级扰动图谱', category: '遗传扰动' },
      { noteId: 'scdrugmap', label: '🎯 scDrugMap — 药物响应基准', category: '遗传扰动' },
      { noteId: 'pca-still-rules', label: '📊 PCA Still Rules — 批评：DL ≠ 更好', category: '遗传扰动' },
      { noteId: 'perturbation-linear-baselines', label: '📊 线性基线 — 简单方法仍具竞争力', category: '遗传扰动' },
    ],
  },
  {
    title: '📊 评估与 Benchmark',
    description: '系统性评估单细胞基础模型的性能与局限',
    category: '评估与 Benchmark',
    items: [
      { noteId: 'perturbench', label: '🔥 PerturBench — 扰动分析标准基准', category: '评估与 Benchmark' },
      { noteId: 'metric-mirages', label: '📊 Metric Mirages — 细胞嵌入评估的度量幻觉', category: '评估与 Benchmark' },
      { noteId: 'zero-shot-limitations', label: '📊 Zero-shot 局限 — 揭示 scFM 泛化边界', category: '评估与 Benchmark' },
      { noteId: 'ssl-effective-use', label: '📊 SSL 有效使用 — 自监督学习指导', category: '评估与 Benchmark' },
      { noteId: 'biology-driven-insights', label: '📊 Biology-driven Insights — scFM 学到了什么', category: '评估与 Benchmark' },
      { noteId: 'deeper-evaluation-scfms', label: '📊 Deeper Evaluation — 深入评估 scFM', category: '评估与 Benchmark' },
      { noteId: 'multimodal-integration-benchmark', label: '📊 多模态整合 — 多组学方法对比', category: '评估与 Benchmark' },
      { noteId: 'virtual-cell-challenge', label: '🎯 Virtual Cell Challenge — Turing Test', category: '评估与 Benchmark' },
    ],
  },
  {
    title: '🔬 病理基础模型',
    description: '计算病理学基础模型',
    category: '病理基础模型',
    items: [
      { noteId: 'uni', label: '⭐🔥 UNI — 通用病理 FM (Nature Med 2024)', category: '病理基础模型' },
      { noteId: 'conch', label: '⭐ CONCH — 视觉-语言病理 FM (Nature Med 2024)', category: '病理基础模型' },
      { noteId: 'whole-slide-fm', label: '🏆 Whole-Slide FM — 真实世界全切片', category: '病理基础模型' },
    ],
  },
  {
    title: '🌌 虚拟细胞',
    description: '所有努力的终极目标——虚拟细胞',
    category: '虚拟细胞',
    items: [
      { noteId: 'the-virtual-cell', label: '⭐ The Virtual Cell — 定义蓝图 (Nat Methods 2025)', category: '虚拟细胞' },
      { noteId: 'vcworld', label: '🎯 VCWorld — 生物世界模型', category: '虚拟细胞' },
      { noteId: 'cellforge', label: '🎯 CellForge — Agentic 虚拟细胞设计', category: '虚拟细胞' },
    ],
  },
  {
    title: '📚 综述与展望',
    description: '领域全景综述和未来方向',
    category: '综述与展望',
    items: [
      { noteId: 'transformers-sc-omics-review', label: '📝 Transformers in sc Omics — 被引最多的综述', category: '综述与展望' },
      { noteId: 'harnessing-fm-omics', label: '📝 Harnessing FM — 基础模型在组学中的应用', category: '综述与展望' },
      { noteId: 'multimodal-fm-cell-biology', label: '📝 多模态 FM — 分子细胞生物学 (Nature 2025)', category: '综述与展望' },
      { noteId: 'interpretation-perturbation-sc', label: '📝 解释/外推/扰动 — 三大任务', category: '综述与展望' },
      { noteId: 'build-virtual-cell-ai', label: '📝 构建虚拟细胞 — 优先事项 (Cell 2024)', category: '综述与展望' },
    ],
  },
]
