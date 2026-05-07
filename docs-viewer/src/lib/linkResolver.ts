/**
 * 将 Markdown 中的相对笔记链接路径映射到 note ID。
 *
 * 路径规范: notes/<分类>/<模型名>/README.md
 */
const notePathToId: Record<string, string> = {
  'README.md': 'overview',

  // fm/ 单细胞基础模型
  'fm/geneformer/README.md': 'geneformer',
  'fm/scgpt/README.md': 'scgpt',
  'fm/scfoundation/README.md': 'scfoundation',
  'fm/cell-atlas-fm/README.md': 'cell-atlas-fm',
  'fm/nicheformer/README.md': 'nicheformer',
  'fm/novae/README.md': 'novae',
  'fm/scbert/README.md': 'scbert',
  'fm/scpoli/README.md': 'scpoli',
  'fm/scprint/README.md': 'scprint',
  'fm/genecompass/README.md': 'genecompass',
  'fm/saturn/README.md': 'saturn',
  'fm/epiagent/README.md': 'epiagent',
  'fm/visual-omics-fm/README.md': 'visual-omics-fm',
  'fm/xtrimogene/README.md': 'xtrimogene',
  'fm/sclong/README.md': 'sclong',
  'fm/cellfm/README.md': 'cellfm',

  // fm-llm/ FM + LLM
  'fm-llm/cell2sentence/README.md': 'cell2sentence',
  'fm-llm/cellama/README.md': 'cellama',
  'fm-llm/cassia/README.md': 'cassia',
  'fm-llm/scchat/README.md': 'scchat',

  // perturbation/ 遗传扰动
  'perturbation/tahoe-100m/README.md': 'tahoe-100m',
  'perturbation/state/README.md': 'state',
  'perturbation/pertadapt/README.md': 'pertadapt',

  // virtual-cell/ 虚拟细胞
  'virtual-cell/virtual-cell-challenge/README.md': 'virtual-cell-challenge',
  'virtual-cell/the-virtual-cell/README.md': 'the-virtual-cell',
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
