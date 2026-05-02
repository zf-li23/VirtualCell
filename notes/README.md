# Virtual Cell 学习笔记 📚

这是一个系统学习虚拟细胞（Virtual Cell）相关基础模型的笔记仓库。

## 目录

| 模型 | 类别 | 技术路线 | 说明 |
|------|------|----------|------|
| [Geneformer](./Geneformer/README.md) | 单细胞转录组基础模型 | BERT MLM + Rank Value Encoding | 基于 Transformer 的 scRNA-seq 预训练模型 |
| [scGPT](./scGPT/README.md) | 单细胞基础模型 | GPT + 基因对 Tokenization + DSBN | 生成式单细胞预训练模型，支持多组学 |
| [scFoundation](./scFoundation/README.md) | 单细胞基础模型 | 非对称 Autoencoder + Performer | ~100M 参数，线性复杂度注意力，基因表达增强 |
| [UCE](./UCE/README.md) | 通用细胞嵌入模型 | 对比学习 + Transformer 编码器 | 跨物种通用细胞嵌入，零样本迁移 |
| [NicheFormer](./NicheFormer/README.md) | 空间转录组模型 | BERT MLM + 空间上下文 Token | 空间组学基础模型，支持多模态 |
| [Novae](./Novae/README.md) | 空间转录组模型 | SwAV + GAT v2 | 图自监督空间组学基础模型 |
| [GraphST](./GraphST/README.md) | 空间转录组模型 | DGI + GNN | 图对比学习空间域识别 |
| [SpaceFlow](./SpaceFlow/README.md) | 空间转录组模型 | DGI + 空间正则化 + Alpha Complex | 时空模式分析与 pSM |
| [SPADE](./SPADE/README.md) | 空间转录组模型 | CLIP 对比学习 + H&E 对齐 | 空间域识别，对齐 ST 与病理图像 |
| [SpaGCN](./SpaGCN/README.md) | 空间转录组模型 | GCN + DEC 聚类 + 组织学融合 | 整合 H&E 图像的空间域识别 |
| [OpenBioMed](./OpenBioMed/README.md) | 多模态生物医学平台 | Agent Skills + LangCell + BioMedGPT | 45+ 自动化生物医学技能
