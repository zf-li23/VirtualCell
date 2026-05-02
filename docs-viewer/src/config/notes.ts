export interface NoteMeta {
  id: string
  title: string
  category: string
}

export const categories = [
  '概览',
  '单细胞基础模型',
  '空间转录组模型',
  '多模态生物医学平台',
] as const

export const noteMetas: NoteMeta[] = [
  { id: 'overview', title: '总览', category: '概览' },
  { id: 'geneformer', title: 'Geneformer', category: '单细胞基础模型' },
  { id: 'scgpt', title: 'scGPT', category: '单细胞基础模型' },
  { id: 'scfoundation', title: 'scFoundation', category: '单细胞基础模型' },
  { id: 'uce', title: 'UCE', category: '单细胞基础模型' },
  { id: 'nicheformer', title: 'NicheFormer', category: '空间转录组模型' },
  { id: 'novae', title: 'Novae', category: '空间转录组模型' },
  { id: 'graphst', title: 'GraphST', category: '空间转录组模型' },
  { id: 'spaceflow', title: 'SpaceFlow', category: '空间转录组模型' },
  { id: 'spade', title: 'SPADE', category: '空间转录组模型' },
  { id: 'spagcn', title: 'SpaGCN', category: '空间转录组模型' },
  { id: 'openbiomed', title: 'OpenBioMed', category: '多模态生物医学平台' },
]
