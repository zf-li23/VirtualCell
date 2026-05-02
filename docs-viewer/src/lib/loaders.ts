export type NoteLoader = () => Promise<string>

function fetchNote(path: string): NoteLoader {
  return () => fetch(`/notes/${path}`).then((r) => r.text())
}

export const noteLoaders: Record<string, NoteLoader> = {
  overview: fetchNote('README.md'),
  geneformer: fetchNote('Geneformer/README.md'),
  scgpt: fetchNote('scGPT/README.md'),
  scfoundation: fetchNote('scFoundation/README.md'),
  uce: fetchNote('UCE/README.md'),
  nicheformer: fetchNote('NicheFormer/README.md'),
  novae: fetchNote('Novae/README.md'),
  graphst: fetchNote('GraphST/README.md'),
  spaceflow: fetchNote('SpaceFlow/README.md'),
  spade: fetchNote('SPADE/README.md'),
  spagcn: fetchNote('SpaGCN/README.md'),
  openbiomed: fetchNote('OpenBioMed/README.md'),
}
