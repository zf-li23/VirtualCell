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
  { id: 'cell-plm', title: 'CellPLM', category: '单细胞基础模型 (FM)', path: 'fm/cell-plm/README.md' },
  { id: 'cellfm', title: 'CellFM', category: '单细胞基础模型 (FM)', path: 'fm/cellfm/README.md' },
  { id: 'epiagent', title: 'EpiAgent', category: '单细胞基础模型 (FM)', path: 'fm/epiagent/README.md' },
  { id: 'genecompass', title: 'GeneCompass', category: '单细胞基础模型 (FM)', path: 'fm/genecompass/README.md' },
  { id: 'geneformer', title: 'Geneformer', category: '单细胞基础模型 (FM)', path: 'fm/geneformer/README.md' },
  { id: 'genejepa', title: 'GeneJEPA', category: '单细胞基础模型 (FM)', path: 'fm/genejepa/README.md' },
  { id: 'heimdall', title: 'Heimdall', category: '单细胞基础模型 (FM)', path: 'fm/heimdall/README.md' },
  { id: 'nicheformer', title: 'Nicheformer', category: '单细胞基础模型 (FM)', path: 'fm/nicheformer/README.md' },
  { id: 'novae', title: 'Novae', category: '单细胞基础模型 (FM)', path: 'fm/novae/README.md' },
  { id: 'omnicell', title: 'OmniCell', category: '单细胞基础模型 (FM)', path: 'fm/omnicell/README.md' },
  { id: 'saturn', title: 'SATURN', category: '单细胞基础模型 (FM)', path: 'fm/saturn/README.md' },
  { id: 'scbert', title: 'scBERT', category: '单细胞基础模型 (FM)', path: 'fm/scbert/README.md' },
  { id: 'scelmo', title: 'scELMo', category: '单细胞基础模型 (FM)', path: 'fm/scelmo/README.md' },
  { id: 'scfoundation', title: 'scFoundation', category: '单细胞基础模型 (FM)', path: 'fm/scfoundation/README.md' },
  { id: 'scgpt', title: 'scGPT', category: '单细胞基础模型 (FM)', path: 'fm/scgpt/README.md' },
  { id: 'scgpt-spatial', title: 'scGPT-spatial', category: '单细胞基础模型 (FM)', path: 'fm/scgpt-spatial/README.md' },
  { id: 'sclong', title: 'scLong', category: '单细胞基础模型 (FM)', path: 'fm/sclong/README.md' },
  { id: 'scmulan', title: 'scMulan', category: '单细胞基础模型 (FM)', path: 'fm/scmulan/README.md' },
  { id: 'scnet', title: 'scNET', category: '单细胞基础模型 (FM)', path: 'fm/scnet/README.md' },
  { id: 'scpeft', title: 'scPEFT', category: '单细胞基础模型 (FM)', path: 'fm/scpeft/README.md' },
  { id: 'scpoli', title: 'scPoli', category: '单细胞基础模型 (FM)', path: 'fm/scpoli/README.md' },
  { id: 'scprint', title: 'scPRINT', category: '单细胞基础模型 (FM)', path: 'fm/scprint/README.md' },
  { id: 'scprint-2', title: 'scPRINT-2', category: '单细胞基础模型 (FM)', path: 'fm/scprint-2/README.md' },
  { id: 'tabulam', title: 'Tabula', category: '单细胞基础模型 (FM)', path: 'fm/tabulam/README.md' },
  { id: 'transcriptformer', title: 'TranscriptFormer', category: '单细胞基础模型 (FM)', path: 'fm/transcriptformer/README.md' },
  { id: 'visual-omics-fm', title: 'Visual-Omics FM', category: '单细胞基础模型 (FM)', path: 'fm/visual-omics-fm/README.md' },
  { id: 'xtrimogene', title: 'xTrimoGene', category: '单细胞基础模型 (FM)', path: 'fm/xtrimogene/README.md' },

  // --- FM + LLM ---
  { id: 'cassia', title: 'CASSIA', category: 'FM + LLM', path: 'fm-llm/cassia/README.md' },
  { id: 'cell2sentence', title: 'Cell2Sentence', category: 'FM + LLM', path: 'fm-llm/cell2sentence/README.md' },
  { id: 'cellama', title: 'CELLama', category: 'FM + LLM', path: 'fm-llm/cellama/README.md' },
  { id: 'celltok', title: 'CellTok', category: 'FM + LLM', path: 'fm-llm/celltok/README.md' },
  { id: 'langcell', title: 'LangCell', category: 'FM + LLM', path: 'fm-llm/langcell/README.md' },
  { id: 'scchat', title: 'scChat', category: 'FM + LLM', path: 'fm-llm/scchat/README.md' },
  { id: 'scouter', title: 'Scouter', category: 'FM + LLM', path: 'fm-llm/scouter/README.md' },

  // --- 遗传扰动 ---
  { id: 'cinema-ot', title: 'CINEMA-OT', category: '遗传扰动', path: 'perturbation/cinema-ot/README.md' },
  { id: 'cpa', title: 'CPA', category: '遗传扰动', path: 'perturbation/cpa/README.md' },
  { id: 'gears', title: 'GEARS', category: '遗传扰动', path: 'perturbation/gears/README.md' },
  { id: 'pca-still-rules', title: 'PCA Still Rules', category: '遗传扰动', path: 'perturbation/pca-still-rules/README.md' },
  { id: 'pertadapt', title: 'PertAdapt', category: '遗传扰动', path: 'perturbation/pertadapt/README.md' },
  { id: 'perturbation-linear-baselines', title: 'Perturbation Linear Baselines', category: '遗传扰动', path: 'perturbation/perturbation-linear-baselines/README.md' },
  { id: 'scdrugmap', title: 'scDrugMap', category: '遗传扰动', path: 'perturbation/scdrugmap/README.md' },
  { id: 'scgenept', title: 'scGenePT', category: '遗传扰动', path: 'perturbation/scgenept/README.md' },
  { id: 'sclambda', title: 'scLAMBDA', category: '遗传扰动', path: 'perturbation/sclambda/README.md' },
  { id: 'state', title: 'STATE', category: '遗传扰动', path: 'perturbation/state/README.md' },
  { id: 'systema', title: 'Systema', category: '遗传扰动', path: 'perturbation/systema/README.md' },
  { id: 'tahoe-100m', title: 'Tahoe-100M / Tahoe-x1', category: '遗传扰动', path: 'perturbation/tahoe-100m/README.md' },

  // --- 评估与 Benchmark ---
  { id: 'biology-driven-insights', title: 'Biology-driven Insights', category: '评估与 Benchmark', path: 'benchmarks/biology-driven-insights/README.md' },
  { id: 'deeper-evaluation-scfms', title: 'Deeper Evaluation of scFMs', category: '评估与 Benchmark', path: 'benchmarks/deeper-evaluation-scfms/README.md' },
  { id: 'metric-mirages', title: 'Metric Mirages', category: '评估与 Benchmark', path: 'benchmarks/metric-mirages/README.md' },
  { id: 'multimodal-integration-benchmark', title: 'Multimodal Integration Benchmark', category: '评估与 Benchmark', path: 'benchmarks/multimodal-integration-benchmark/README.md' },
  { id: 'perturbench', title: 'PerturBench', category: '评估与 Benchmark', path: 'benchmarks/perturbench/README.md' },
  { id: 'ssl-effective-use', title: 'SSL Effective Use', category: '评估与 Benchmark', path: 'benchmarks/ssl-effective-use/README.md' },
  { id: 'virtual-cell-challenge', title: 'Virtual Cell Challenge', category: '评估与 Benchmark', path: 'benchmarks/virtual-cell-challenge/README.md' },
  { id: 'zero-shot-limitations', title: 'Zero-shot Limitations', category: '评估与 Benchmark', path: 'benchmarks/zero-shot-limitations/README.md' },

  // --- 虚拟细胞 ---
  { id: 'cellforge', title: 'CellForge', category: '虚拟细胞', path: 'virtual-cell/cellforge/README.md' },
  { id: 'the-virtual-cell', title: 'The Virtual Cell', category: '虚拟细胞', path: 'virtual-cell/the-virtual-cell/README.md' },
  { id: 'vcworld', title: 'VCWorld', category: '虚拟细胞', path: 'virtual-cell/vcworld/README.md' },

  // --- 病理基础模型 ---
  { id: 'conch', title: 'CONCH', category: '病理基础模型', path: 'pathology/conch/README.md' },
  { id: 'uni', title: 'UNI', category: '病理基础模型', path: 'pathology/uni/README.md' },
  { id: 'whole-slide-fm', title: 'Whole-Slide FM', category: '病理基础模型', path: 'pathology/whole-slide-fm/README.md' },

  // --- 综述与展望 ---
  { id: 'build-virtual-cell-ai', title: 'Build Virtual Cell with AI', category: '综述与展望', path: 'surveys/build-virtual-cell-ai/README.md' },
  { id: 'harnessing-fm-omics', title: 'Harnessing the Deep Learning Power of Foundation Models in Single-Cell Omics', category: '综述与展望', path: 'surveys/harnessing-fm-omics/README.md' },
  { id: 'interpretation-perturbation-sc', title: 'Interpretation, Extrapolation and Perturbation of Single Cells', category: '综述与展望', path: 'surveys/interpretation-perturbation-sc/README.md' },
  { id: 'multimodal-fm-cell-biology', title: 'Towards Multimodal Foundation Models in Molecular Cell Biology', category: '综述与展望', path: 'surveys/multimodal-fm-cell-biology/README.md' },
  { id: 'transformers-sc-omics-review', title: 'Transformers in Single-Cell Omics: A Review and New Perspectives', category: '综述与展望', path: 'surveys/transformers-sc-omics-review/README.md' },
]
