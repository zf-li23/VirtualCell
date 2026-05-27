// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成
// 手动修改将被覆盖！请修改笔记的 frontmatter 后重新运行。

export interface NoteMeta {
  id: string
  title: string
  category: string
  path: string
}

export const categories = [
  '单细胞基础模型 (FM)',
  'FM + LLM',
  '遗传扰动',
  '评估与 Benchmark',
  '虚拟细胞',
  '病理基础模型',
  '综述与展望',
] as const

export const noteMetas: NoteMeta[] = [
  { id: 'overview', title: '总览', category: '概览', path: 'README.md' },

  // --- 单细胞基础模型 (FM) ---
  { id: 'cell-atlas-fm', title: 'UCE / Cell Atlas FM', category: '单细胞基础模型 (FM)', path: 'fm/cell-atlas-fm/README.md' },
  { id: 'cellfm', title: 'CellFM', category: '单细胞基础模型 (FM)', path: 'fm/cellfm/README.md' },
  { id: 'epiagent', title: 'EpiAgent', category: '单细胞基础模型 (FM)', path: 'fm/epiagent/README.md' },
  { id: 'genecompass', title: 'GeneCompass', category: '单细胞基础模型 (FM)', path: 'fm/genecompass/README.md' },
  { id: 'geneformer', title: 'Geneformer', category: '单细胞基础模型 (FM)', path: 'fm/geneformer/README.md' },
  { id: 'nicheformer', title: 'Nicheformer', category: '单细胞基础模型 (FM)', path: 'fm/nicheformer/README.md' },
  { id: 'novae', title: 'Novae', category: '单细胞基础模型 (FM)', path: 'fm/novae/README.md' },
  { id: 'saturn', title: 'SATURN', category: '单细胞基础模型 (FM)', path: 'fm/saturn/README.md' },
  { id: 'scbert', title: 'scBERT', category: '单细胞基础模型 (FM)', path: 'fm/scbert/README.md' },
  { id: 'scfoundation', title: 'scFoundation', category: '单细胞基础模型 (FM)', path: 'fm/scfoundation/README.md' },
  { id: 'scgpt', title: 'scGPT', category: '单细胞基础模型 (FM)', path: 'fm/scgpt/README.md' },
  { id: 'sclong', title: 'scLong', category: '单细胞基础模型 (FM)', path: 'fm/sclong/README.md' },
  { id: 'scpoli', title: 'scPoli', category: '单细胞基础模型 (FM)', path: 'fm/scpoli/README.md' },
  { id: 'scprint', title: 'scPRINT', category: '单细胞基础模型 (FM)', path: 'fm/scprint/README.md' },
  { id: 'visual-omics-fm', title: 'Visual-Omics FM', category: '单细胞基础模型 (FM)', path: 'fm/visual-omics-fm/README.md' },
  { id: 'xtrimogene', title: 'xTrimoGene', category: '单细胞基础模型 (FM)', path: 'fm/xtrimogene/README.md' },

  // --- FM + LLM ---
  { id: 'cassia', title: 'CASSIA', category: 'FM + LLM', path: 'fm-llm/cassia/README.md' },
  { id: 'cell2sentence', title: 'Cell2Sentence', category: 'FM + LLM', path: 'fm-llm/cell2sentence/README.md' },
  { id: 'cellama', title: 'CELLama', category: 'FM + LLM', path: 'fm-llm/cellama/README.md' },
  { id: 'langcell', title: 'LangCell', category: 'FM + LLM', path: 'fm-llm/langcell/README.md' },
  { id: 'scchat', title: 'scChat', category: 'FM + LLM', path: 'fm-llm/scchat/README.md' },

  // --- 遗传扰动 ---
  { id: 'pertadapt', title: 'PertAdapt', category: '遗传扰动', path: 'perturbation/pertadapt/README.md' },
  { id: 'state', title: 'STATE', category: '遗传扰动', path: 'perturbation/state/README.md' },
  { id: 'tahoe-100m', title: 'Tahoe-100M / Tahoe-x1', category: '遗传扰动', path: 'perturbation/tahoe-100m/README.md' },

  // --- 评估与 Benchmark ---
  { id: 'virtual-cell-challenge', title: 'Virtual Cell Challenge', category: '评估与 Benchmark', path: 'benchmarks/virtual-cell-challenge/README.md' },

  // --- 虚拟细胞 ---
  { id: 'the-virtual-cell', title: 'The Virtual Cell', category: '虚拟细胞', path: 'virtual-cell/the-virtual-cell/README.md' },
]
