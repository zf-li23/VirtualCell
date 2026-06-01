export interface TechNode {
  name: string
  children?: TechNode[]
  noteId?: string
  description?: string
}

/**
 * 技术谱系树 — 按架构路线组织模型
 */
export const techLineage: TechNode[] = [
  {
    name: 'BERT 路线',
    description: '双向 Transformer 编码器 + 掩码语言建模',
    children: [
      { name: 'scBERT', noteId: 'scbert', description: '首个 BERT 进单细胞 (2022)' },
      { name: 'Geneformer', noteId: 'geneformer', description: 'Rank Encoding + 3000 万细胞 (2023)' },
      { name: 'Nicheformer', noteId: 'nicheformer', description: 'BERT + 空间邻域 (2025)' },
      { name: 'UCE', noteId: 'cell-atlas-fm', description: '通用细胞嵌入 + 二元分类器 (2024)' },
      { name: 'Heimdall', noteId: 'heimdall', description: 'Tokenization 框架 (2025)' },
      { name: 'scPEFT', noteId: 'scpeft', description: '参数高效微调框架 (2025)' },
      { name: 'scPRINT', noteId: 'scprint', description: 'Performer + GRN 推断 (2025)' },
      { name: 'scPRINT-2', noteId: 'scprint-2', description: '下一代 + lamin.ai (2025)' },
      { name: 'scPoli', noteId: 'scpoli', description: '条件 VAE 群体整合 (2023)' },
      { name: 'Tabula', noteId: 'tabulam', description: '表格自监督学习 (2025)' },
    ],
  },
  {
    name: 'GPT 路线',
    description: '单向因果 Transformer + 自回归生成',
    children: [
      { name: 'scGPT', noteId: 'scgpt', description: 'GPT + 基因对 Tokenization (2024)' },
      { name: 'scMulan', noteId: 'scmulan', description: '多任务生成式预训练 (2024)' },
      { name: 'scLong', noteId: 'sclong', description: '长上下文 + GO 引导注意力 (2024)' },
      { name: 'CellFM', noteId: 'cellfm', description: '1 亿细胞 BERT 规模 (2025)' },
    ],
  },
  {
    name: '非对称 AE / 高效 Attention 路线',
    description: '编码器-解码器非对称设计 + 线性复杂度注意力',
    children: [
      { name: 'xTrimoGene', noteId: 'xtrimogene', description: '非对称 AE 先驱 (NeurIPS 2023)' },
      { name: 'scFoundation', noteId: 'scfoundation', description: 'Performer 线性注意力 (2024)' },
    ],
  },
  {
    name: 'VAE / 解耦表示路线',
    description: '变分自编码器 + 组合性潜在空间',
    children: [
      { name: 'scPoli', noteId: 'scpoli', description: '条件 VAE + 群体整合 (2023)' },
      { name: 'CPA', noteId: 'cpa', description: '加性解耦 + 剂量响应 (2023)' },
      { name: 'CINEMA-OT', noteId: 'cinema-ot', description: '因果最优传输 (2023)' },
    ],
  },
  {
    name: 'JEPA / 世界模型路线',
    description: '联合嵌入预测架构，预测表示而非重建数据',
    children: [
      { name: 'GeneJEPA', noteId: 'genejepa', description: 'Perceiver + VICReg (2025)' },
    ],
  },
  {
    name: 'GNN / 图网络路线',
    description: '图神经网络 + 关系结构建模',
    children: [
      { name: 'scNET', noteId: 'scnet', description: 'PPI 网络 + 基因表达融合 (2025)' },
      { name: 'GEARS', noteId: 'gears', description: 'GO 图传播扰动效应 (2023)' },
      { name: 'CellPLM', noteId: 'cell-plm', description: '细胞-细胞图编码器 (2024)' },
      { name: 'PertAdapt', noteId: 'pertadapt', description: 'GO 引导注意力适配器 (2025)' },
      { name: 'scLong', noteId: 'sclong', description: 'GO 图引导对比注意力 (2024)' },
      { name: 'Novae', noteId: 'novae', description: 'GATv2 + SwAV 空间图 (2025)' },
    ],
  },
  {
    name: '跨物种 / 对比学习路线',
    description: '跨物种对齐 + 通用细胞表示',
    children: [
      { name: 'SATURN', noteId: 'saturn', description: 'DNA 序列引导对比学习 (2024)' },
      { name: 'GeneCompass', noteId: 'genecompass', description: '知识引导跨物种 FM (2024)' },
      { name: 'TranscriptFormer', noteId: 'transcriptformer', description: '跨物种生成式细胞图谱 (2025)' },
    ],
  },
  {
    name: 'LLM + 细胞路线',
    description: '大语言模型与单细胞数据的交叉融合',
    children: [
      { name: 'Cell2Sentence', noteId: 'cell2sentence', description: '细胞→句子→LLM (ICML 2024)' },
      { name: 'CELLama', noteId: 'cellama', description: 'LLM 驱动细胞嵌入 (2024)' },
      { name: 'CellTok', noteId: 'celltok', description: '早期融合多模态 LLM (2025)' },
      { name: 'scChat', noteId: 'scchat', description: 'LLM Co-Pilot 分析 (2024)' },
      { name: 'CASSIA', noteId: 'cassia', description: '多智能体自动注释 (2025)' },
      { name: 'Scouter', noteId: 'scouter', description: 'LLM 嵌入做扰动预测 (2025)' },
      { name: 'scELMo', noteId: 'scelmo', description: 'LLM 基因描述嵌入 (2023)' },
      { name: 'LangCell', noteId: 'langcell', description: '语言-细胞联合预训练 (ICML 2024)' },
    ],
  },
]
