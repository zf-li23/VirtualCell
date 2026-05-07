export interface NoteMeta {
  id: string
  title: string
  category: string
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
  { id: 'overview', title: '总览', category: '概览' },

  // --- 评估与 Benchmark ---
  // (35 papers, 暂不全部加入, 保留 overview)

  // --- 单细胞基础模型 (FM) ---
  { id: 'geneformer', title: 'Geneformer (2023 Nature)', category: '单细胞基础模型 (FM)' },
  { id: 'scgpt', title: 'scGPT (2024 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'scfoundation', title: 'scFoundation (2024 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'cell-atlas-fm', title: 'UCE / Cell Atlas FM (2024 Nature)', category: '单细胞基础模型 (FM)' },
  { id: 'nicheformer', title: 'Nicheformer (2025 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'novae', title: 'Novae (2025 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'scbert', title: 'scBERT (2022 Nat Mach Intell)', category: '单细胞基础模型 (FM)' },
  { id: 'scpoli', title: 'scPoli (2023 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'scprint', title: 'scPRINT (2025 Nat Comms)', category: '单细胞基础模型 (FM)' },
  { id: 'genecompass', title: 'GeneCompass (2024 Cell Res)', category: '单细胞基础模型 (FM)' },
  { id: 'saturn', title: 'SATURN (2024 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'epiagent', title: 'EpiAgent (2025 Nat Methods)', category: '单细胞基础模型 (FM)' },
  { id: 'visual-omics-fm', title: 'Visual-Omics FM (2025 Nat Methods)', category: '单细胞基础模型 (FM)' },
]
