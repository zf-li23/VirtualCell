export type NoteLoader = () => Promise<string>

function fetchNote(path: string): NoteLoader {
  return () => fetch(`/notes/${path}`).then((r) => r.text())
}

export const noteLoaders: Record<string, NoteLoader> = {
  overview: fetchNote('README.md'),

  // 单细胞基础模型 (FM)
  geneformer: fetchNote('fm/geneformer/README.md'),
  scgpt: fetchNote('fm/scgpt/README.md'),
  scfoundation: fetchNote('fm/scfoundation/README.md'),
  'cell-atlas-fm': fetchNote('fm/cell-atlas-fm/README.md'),
  nicheformer: fetchNote('fm/nicheformer/README.md'),
  novae: fetchNote('fm/novae/README.md'),
  scbert: fetchNote('fm/scbert/README.md'),
  scpoli: fetchNote('fm/scpoli/README.md'),
  scprint: fetchNote('fm/scprint/README.md'),
  genecompass: fetchNote('fm/genecompass/README.md'),
  saturn: fetchNote('fm/saturn/README.md'),
  epiagent: fetchNote('fm/epiagent/README.md'),
  'visual-omics-fm': fetchNote('fm/visual-omics-fm/README.md'),
}
