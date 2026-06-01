export interface LabGroup {
  name: string
  members: string[]
  color?: string
}

export const labGroups: LabGroup[] = [
  {
    name: 'Stanford (Quake / Leskovec Lab)',
    members: ['cell-atlas-fm', 'saturn', 'gears'],
  },
  {
    name: 'Tencent AI Lab',
    members: ['scbert', 'cellfm'],
  },
  {
    name: 'BioMap (百图生科)',
    members: ['scfoundation', 'xtrimogene'],
  },
  {
    name: 'Bowang Lab (UofT / Vector Institute)',
    members: ['scgpt', 'scgpt-spatial'],
  },
  {
    name: 'Theis Lab (Helmholtz Munich)',
    members: ['scpoli', 'nicheformer', 'cpa', 'cinema-ot'],
  },
  {
    name: 'TahoeBio / Genentech',
    members: ['tahoe-100m', 'genejepa', 'scgenept'],
  },
  {
    name: 'Arc Institute',
    members: ['state', 'virtual-cell-challenge', 'transcriptformer'],
  },
  {
    name: 'Cantini Lab (Institut Curie)',
    members: ['scprint', 'scprint-2'],
  },
  {
    name: 'Mahmood Lab (Harvard / BWH)',
    members: ['uni', 'conch'],
  },
  {
    name: 'Gladstone Institutes / UCSF',
    members: ['geneformer'],
  },
  {
    name: 'CZ Biohub / Stanford',
    members: ['cell-atlas-fm', 'virtual-cell-challenge'],
  },
  {
    name: 'BGI Research (华大)',
    members: ['omnicell'],
  },
  {
    name: 'ETH Zurich / EPFL',
    members: ['systema', 'scgpt'],
  },
  {
    name: 'Microsoft Research',
    members: ['whole-slide-fm'],
  },
  {
    name: 'Snap Stanford / PNNL',
    members: ['cell-plm'],
  },
  {
    name: 'Altos Labs',
    members: ['perturbench'],
  },
]
