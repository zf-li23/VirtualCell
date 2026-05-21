export interface NoteMeta {
  id: string
  title: string
  category: string
  /** public/notes/ 下的相对路径（不含 BASE_URL） */
  path: string
}

export const categories = [
  '概览',
  '评估与 Benchmark',
  '虚拟细胞',
  '综述与展望',
  '单细胞基础模型 (FM)',
  'FM + LLM',
  '遗传扰动',
  '病理基础模型',
] as const

export const noteMetas: NoteMeta[] = [
  // --- 概览 ---
  { id: 'overview', title: '总览', category: '概览', path: 'README.md' },

  // --- 单细胞基础模型 (FM) ---
  { id: 'geneformer', title: 'Geneformer (2023 Nature)', category: '单细胞基础模型 (FM)', path: 'fm/geneformer/README.md' },
  { id: 'scgpt', title: 'scGPT (2024 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/scgpt/README.md' },
  { id: 'scfoundation', title: 'scFoundation (2024 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/scfoundation/README.md' },
  { id: 'cell-atlas-fm', title: 'UCE / Cell Atlas FM (2024 Nature)', category: '单细胞基础模型 (FM)', path: 'fm/cell-atlas-fm/README.md' },
  { id: 'nicheformer', title: 'Nicheformer (2025 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/nicheformer/README.md' },
  { id: 'novae', title: 'Novae (2025 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/novae/README.md' },
  { id: 'scbert', title: 'scBERT (2022 Nat Mach Intell)', category: '单细胞基础模型 (FM)', path: 'fm/scbert/README.md' },
  { id: 'scpoli', title: 'scPoli (2023 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/scpoli/README.md' },
  { id: 'scprint', title: 'scPRINT (2025 Nat Comms)', category: '单细胞基础模型 (FM)', path: 'fm/scprint/README.md' },
  { id: 'genecompass', title: 'GeneCompass (2024 Cell Res)', category: '单细胞基础模型 (FM)', path: 'fm/genecompass/README.md' },
  { id: 'saturn', title: 'SATURN (2024 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/saturn/README.md' },
  { id: 'epiagent', title: 'EpiAgent (2025 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/epiagent/README.md' },
  { id: 'visual-omics-fm', title: 'Visual-Omics FM (2025 Nat Methods)', category: '单细胞基础模型 (FM)', path: 'fm/visual-omics-fm/README.md' },
  { id: 'langcell', title: 'LangCell (2024 ICML)', category: 'FM + LLM', path: 'fm-llm/langcell/README.md' },
  { id: 'xtrimogene', title: 'xTrimoGene (2023 NeurIPS)', category: '单细胞基础模型 (FM)', path: 'fm/xtrimogene/README.md' },
  { id: 'sclong', title: 'scLong (2024 bioRxiv)', category: '单细胞基础模型 (FM)', path: 'fm/sclong/README.md' },
  { id: 'cellfm', title: 'CellFM (2025 Nat Comms)', category: '单细胞基础模型 (FM)', path: 'fm/cellfm/README.md' },

  // --- FM + LLM ---
  { id: 'cell2sentence', title: 'Cell2Sentence (2024 ICML)', category: 'FM + LLM', path: 'fm-llm/cell2sentence/README.md' },
  { id: 'cellama', title: 'CELLama (2024 bioRxiv)', category: 'FM + LLM', path: 'fm-llm/cellama/README.md' },
  { id: 'cassia', title: 'CASSIA (2025 Nat Comms)', category: 'FM + LLM', path: 'fm-llm/cassia/README.md' },
  { id: 'scchat', title: 'scChat (2024 bioRxiv)', category: 'FM + LLM', path: 'fm-llm/scchat/README.md' },

  // --- 遗传扰动 ---
  { id: 'tahoe-100m', title: 'Tahoe-100M (2025 bioRxiv)', category: '遗传扰动', path: 'perturbation/tahoe-100m/README.md' },
  { id: 'state', title: 'STATE (2025 bioRxiv)', category: '遗传扰动', path: 'perturbation/state/README.md' },
  { id: 'pertadapt', title: 'PertAdapt (2025 bioRxiv)', category: '遗传扰动', path: 'perturbation/pertadapt/README.md' },

  // --- 虚拟细胞 ---
  { id: 'virtual-cell-challenge', title: 'Virtual Cell Challenge (2025 Cell)', category: '虚拟细胞', path: 'benchmarks/virtual-cell-challenge/README.md' },
  { id: 'the-virtual-cell', title: 'The Virtual Cell (2025 Nat Methods)', category: '虚拟细胞', path: 'virtual-cell/the-virtual-cell/README.md' },
]

