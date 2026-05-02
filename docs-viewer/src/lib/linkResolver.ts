/**
 * 将 Markdown 中的相对笔记链接路径映射到 note ID。
 *
 * 例如: ./Geneformer/README.md → geneformer
 */
const notePathToId: Record<string, string> = {
  'README.md': 'overview',
  'Geneformer/README.md': 'geneformer',
  'scGPT/README.md': 'scgpt',
  'scFoundation/README.md': 'scfoundation',
  'UCE/README.md': 'uce',
  'NicheFormer/README.md': 'nicheformer',
  'Novae/README.md': 'novae',
  'GraphST/README.md': 'graphst',
  'SpaceFlow/README.md': 'spaceflow',
  'SPADE/README.md': 'spade',
  'SpaGCN/README.md': 'spagcn',
  'OpenBioMed/README.md': 'openbiomed',
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
