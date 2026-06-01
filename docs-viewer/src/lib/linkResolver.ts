// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成
// 手动修改将被覆盖！请修改笔记的 frontmatter 后重新运行。

/**
 * 将 Markdown 中的相对笔记链接路径映射到 note ID。
 */
const notePathToId: Record<string, string> = {
  'README.md': 'overview',
  'benchmarks/biology-driven-insights/README.md': 'biology-driven-insights',
  'benchmarks/deeper-evaluation-scfms/README.md': 'deeper-evaluation-scfms',
  'benchmarks/metric-mirages/README.md': 'metric-mirages',
  'benchmarks/multimodal-integration-benchmark/README.md': 'multimodal-integration-benchmark',
  'benchmarks/perturbench/README.md': 'perturbench',
  'benchmarks/ssl-effective-use/README.md': 'ssl-effective-use',
  'benchmarks/virtual-cell-challenge/README.md': 'virtual-cell-challenge',
  'benchmarks/zero-shot-limitations/README.md': 'zero-shot-limitations',
  'fm-classic/cellfm/README.md': 'cellfm',
  'fm-classic/epiagent/README.md': 'epiagent',
  'fm-classic/geneformer/README.md': 'geneformer',
  'fm-classic/heimdall/README.md': 'heimdall',
  'fm-classic/scbert/README.md': 'scbert',
  'fm-classic/scfoundation/README.md': 'scfoundation',
  'fm-classic/scgpt/README.md': 'scgpt',
  'fm-classic/sclong/README.md': 'sclong',
  'fm-classic/scmulan/README.md': 'scmulan',
  'fm-classic/scpeft/README.md': 'scpeft',
  'fm-classic/scpoli/README.md': 'scpoli',
  'fm-classic/scprint/README.md': 'scprint',
  'fm-classic/scprint-2/README.md': 'scprint-2',
  'fm-classic/tabulam/README.md': 'tabulam',
  'fm-classic/xtrimogene/README.md': 'xtrimogene',
  'fm-cross-species/cell-atlas-fm/README.md': 'cell-atlas-fm',
  'fm-cross-species/genecompass/README.md': 'genecompass',
  'fm-cross-species/saturn/README.md': 'saturn',
  'fm-cross-species/transcriptformer/README.md': 'transcriptformer',
  'fm-graph/cell-plm/README.md': 'cell-plm',
  'fm-graph/gears/README.md': 'gears',
  'fm-graph/pertadapt/README.md': 'pertadapt',
  'fm-graph/scnet/README.md': 'scnet',
  'fm-llm/cassia/README.md': 'cassia',
  'fm-llm/cell2sentence/README.md': 'cell2sentence',
  'fm-llm/cellama/README.md': 'cellama',
  'fm-llm/celltok/README.md': 'celltok',
  'fm-llm/langcell/README.md': 'langcell',
  'fm-llm/scchat/README.md': 'scchat',
  'fm-llm/scelmo/README.md': 'scelmo',
  'fm-llm/scouter/README.md': 'scouter',
  'fm-spatial/nicheformer/README.md': 'nicheformer',
  'fm-spatial/novae/README.md': 'novae',
  'fm-spatial/omnicell/README.md': 'omnicell',
  'fm-spatial/scgpt-spatial/README.md': 'scgpt-spatial',
  'fm-spatial/visual-omics-fm/README.md': 'visual-omics-fm',
  'fm-world-model/genejepa/README.md': 'genejepa',
  'pathology/conch/README.md': 'conch',
  'pathology/uni/README.md': 'uni',
  'pathology/whole-slide-fm/README.md': 'whole-slide-fm',
  'perturbation/cinema-ot/README.md': 'cinema-ot',
  'perturbation/cpa/README.md': 'cpa',
  'perturbation/pca-still-rules/README.md': 'pca-still-rules',
  'perturbation/perturbation-linear-baselines/README.md': 'perturbation-linear-baselines',
  'perturbation/scdrugmap/README.md': 'scdrugmap',
  'perturbation/scgenept/README.md': 'scgenept',
  'perturbation/sclambda/README.md': 'sclambda',
  'perturbation/state/README.md': 'state',
  'perturbation/systema/README.md': 'systema',
  'perturbation/tahoe-100m/README.md': 'tahoe-100m',
  'surveys/build-virtual-cell-ai/README.md': 'build-virtual-cell-ai',
  'surveys/harnessing-fm-omics/README.md': 'harnessing-fm-omics',
  'surveys/interpretation-perturbation-sc/README.md': 'interpretation-perturbation-sc',
  'surveys/multimodal-fm-cell-biology/README.md': 'multimodal-fm-cell-biology',
  'surveys/transformers-sc-omics-review/README.md': 'transformers-sc-omics-review',
  'virtual-cell/cellforge/README.md': 'cellforge',
  'virtual-cell/the-virtual-cell/README.md': 'the-virtual-cell',
  'virtual-cell/vcworld/README.md': 'vcworld',
}

export function resolveNoteId(href: string): string | null {
  const normalized = href.replace(/^\.\//, '').split('#')[0].split('?')[0]
  return notePathToId[normalized] ?? null
}

/** 判断是否外部链接 */
export function isExternalLink(href: string): boolean {
  return href.startsWith('http://') || href.startsWith('https://')
}

/** 判断是否锚点链接 */
export function isAnchorLink(href: string): boolean {
  return href.startsWith('#')
}
