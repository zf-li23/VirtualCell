/**
 * 将 Markdown 中的相对笔记链接路径映射到 note ID。
 *
 * 路径规范: notes/<分类>/<模型名>/README.md
 */
const notePathToId: Record<string, string> = {
  'README.md': 'overview',

  // fm-classic/ 经典语言模型
  'fm-classic/geneformer/README.md': 'geneformer',
  'fm-classic/scgpt/README.md': 'scgpt',
  'fm-classic/scfoundation/README.md': 'scfoundation',
  'fm-classic/scbert/README.md': 'scbert',
  'fm-classic/scmulan/README.md': 'scmulan',
  'fm-classic/cellfm/README.md': 'cellfm',
  'fm-classic/sclong/README.md': 'sclong',
  'fm-classic/xtrimogene/README.md': 'xtrimogene',
  'fm-classic/heimdall/README.md': 'heimdall',
  'fm-classic/scpeft/README.md': 'scpeft',
  'fm-classic/epiagent/README.md': 'epiagent',
  'fm-classic/scprint/README.md': 'scprint',
  'fm-classic/scprint-2/README.md': 'scprint-2',
  'fm-classic/scpoli/README.md': 'scpoli',
  'fm-classic/tabulam/README.md': 'tabulam',

  // fm-spatial/ 空间组学
  'fm-spatial/nicheformer/README.md': 'nicheformer',
  'fm-spatial/novae/README.md': 'novae',
  'fm-spatial/scgpt-spatial/README.md': 'scgpt-spatial',
  'fm-spatial/visual-omics-fm/README.md': 'visual-omics-fm',
  'fm-spatial/omnicell/README.md': 'omnicell',

  // fm-world-model/ 世界模型
  'fm-world-model/genejepa/README.md': 'genejepa',

  // fm-cross-species/ 跨物种
  'fm-cross-species/cell-atlas-fm/README.md': 'cell-atlas-fm',
  'fm-cross-species/saturn/README.md': 'saturn',
  'fm-cross-species/genecompass/README.md': 'genecompass',
  'fm-cross-species/transcriptformer/README.md': 'transcriptformer',

  // fm-graph/ 图与网络
  'fm-graph/scnet/README.md': 'scnet',
  'fm-graph/gears/README.md': 'gears',
  'fm-graph/pertadapt/README.md': 'pertadapt',
  'fm-graph/cell-plm/README.md': 'cell-plm',

  // fm-llm/ FM + LLM
  'fm-llm/cell2sentence/README.md': 'cell2sentence',
  'fm-llm/cellama/README.md': 'cellama',
  'fm-llm/cassia/README.md': 'cassia',
  'fm-llm/scchat/README.md': 'scchat',
  'fm-llm/celltok/README.md': 'celltok',
  'fm-llm/langcell/README.md': 'langcell',
  'fm-llm/scouter/README.md': 'scouter',
  'fm-llm/scelmo/README.md': 'scelmo',

  // perturbation/ 遗传扰动
  'perturbation/tahoe-100m/README.md': 'tahoe-100m',
  'perturbation/state/README.md': 'state',
  'perturbation/cpa/README.md': 'cpa',
  'perturbation/systema/README.md': 'systema',
  'perturbation/sclambda/README.md': 'sclambda',
  'perturbation/cinema-ot/README.md': 'cinema-ot',
  'perturbation/scdrugmap/README.md': 'scdrugmap',
  'perturbation/scgenept/README.md': 'scgenept',
  'perturbation/pca-still-rules/README.md': 'pca-still-rules',
  'perturbation/perturbation-linear-baselines/README.md': 'perturbation-linear-baselines',

  // benchmarks/ 评估
  'benchmarks/virtual-cell-challenge/README.md': 'virtual-cell-challenge',
  'benchmarks/perturbench/README.md': 'perturbench',
  'benchmarks/metric-mirages/README.md': 'metric-mirages',
  'benchmarks/zero-shot-limitations/README.md': 'zero-shot-limitations',
  'benchmarks/ssl-effective-use/README.md': 'ssl-effective-use',
  'benchmarks/biology-driven-insights/README.md': 'biology-driven-insights',
  'benchmarks/deeper-evaluation-scfms/README.md': 'deeper-evaluation-scfms',
  'benchmarks/multimodal-integration-benchmark/README.md': 'multimodal-integration-benchmark',

  // virtual-cell/ 虚拟细胞
  'virtual-cell/virtual-cell-challenge/README.md': 'virtual-cell-challenge',
  'virtual-cell/the-virtual-cell/README.md': 'the-virtual-cell',
  'virtual-cell/vcworld/README.md': 'vcworld',
  'virtual-cell/cellforge/README.md': 'cellforge',

  // pathology/ 病理
  'pathology/uni/README.md': 'uni',
  'pathology/conch/README.md': 'conch',
  'pathology/whole-slide-fm/README.md': 'whole-slide-fm',

  // surveys/ 综述
  'surveys/transformers-sc-omics-review/README.md': 'transformers-sc-omics-review',
  'surveys/harnessing-fm-omics/README.md': 'harnessing-fm-omics',
  'surveys/multimodal-fm-cell-biology/README.md': 'multimodal-fm-cell-biology',
  'surveys/interpretation-perturbation-sc/README.md': 'interpretation-perturbation-sc',
  'surveys/build-virtual-cell-ai/README.md': 'build-virtual-cell-ai',
}

/**
 * 解析 Markdown 中的 href，返回匹配的 noteId（如果存在）。
 */
export function resolveNoteId(href: string): string | null {
  // 去掉 ./ 前缀，并移除 hash/query 片段
  const normalized = href.replace(/^\.\//, '').split('#')[0].split('?')[0]
  return notePathToId[normalized] ?? null
}

/**
 * 判断 href 是否为外部链接（以 http 开头）。
 */
export function isExternalLink(href: string): boolean {
  return /^https?:\/\//i.test(href)
}

/**
 * 判断 href 是否为锚点链接（以 # 开头且不含路径）。
 */
export function isAnchorLink(href: string): boolean {
  return href.startsWith('#')
}
