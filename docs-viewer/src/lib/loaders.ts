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
  xtrimogene: fetchNote('fm/xtrimogene/README.md'),
  sclong: fetchNote('fm/sclong/README.md'),
  cellfm: fetchNote('fm/cellfm/README.md'),
  cell2sentence: fetchNote('fm-llm/cell2sentence/README.md'),
  cellama: fetchNote('fm-llm/cellama/README.md'),
  cassia: fetchNote('fm-llm/cassia/README.md'),
  scchat: fetchNote('fm-llm/scchat/README.md'),
  'tahoe-100m': fetchNote('perturbation/tahoe-100m/README.md'),
  state: fetchNote('perturbation/state/README.md'),
  pertadapt: fetchNote('perturbation/pertadapt/README.md'),
  'virtual-cell-challenge': fetchNote('virtual-cell/virtual-cell-challenge/README.md'),
  'the-virtual-cell': fetchNote('virtual-cell/the-virtual-cell/README.md'),
}
