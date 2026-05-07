# FM — 单细胞基础模型

> 核心单细胞基础模型（Foundation Models for Single-Cell Genomics），共 70 篇。

---

## 目录结构

每个模型在 `fm/` 下有一个独立子目录：

```
fm/
├── README.md                  ← 本文件
├── stack/                     ← Stack: In-Context Learning (2026)
├── genejepa/                  ← GeneJepa (2025)
├── unified-perturbation/      ← Unified modeling of cellular responses (2025)
├── omnicell/                  ← OmniCell (2025)
├── bidir-mamba/               ← Bidirectional Mamba (2025)
├── spatial-proteomics-fm/     ← A Foundation Model for Spatial Proteomics (2025)
├── scpeft/                    ← scPEFT (2025)
├── scprint-2/                 ← scPRINT-2 (2025)
├── novae/                     ← Novae (2025) — 📗 [→ 笔记](./novae/)
├── stpath/                    ← STPath (2025)
├── pulsar/                    ← PULSAR (2025)
├── switch/                    ← SWITCH (2025)
├── spatranslator/             ← SpaTranslator (2025)
├── scconcept/                 ← scConcept (2025)
├── latent-diffusion-sc/       ← Scalable sc Generation with Latent Diffusion (2025)
├── transcriptformer/          ← TranscriptFormer (2025)
├── atacformer/                ← Atacformer (2025)
├── epifoundation/             ← EpiFoundation (2025)
├── chromfound/                ← ChromFound (2025)
├── nicheformer/               ← Nicheformer (2025) — 📗 [→ 笔记](./nicheformer/)
├── sclinguist/                ← scLinguist (2025)
├── transcription-foundation/  ← A foundation model of transcription (2025)
├── scprint/                   ← scPRINT (2025)
├── visual-omics-fm/           ← A visual–omics foundation model (2025)
├── epiagent/                  ← EpiAgent (2025)
├── captain/                   ← CAPTAIN (2025)
├── scgpt-spatial/             ← scGPT-spatial (2025)
├── stofm/                     ← SToFM (2025)
├── scarf/                     ← SCARF (2025)
├── multimodal-perturbation/   ← Multimodal FM zero-shot perturbations (2025)
├── cellfm/                    ← CellFM (2025)
├── tabula/                    ← Tabula (2025)
├── transcriptome-proteome/    ← sc transcriptome to proteome (2025)
├── privacy-federated/         ← Privacy-preserving FM (2025)
├── scnet/                     ← scNET (2025)
├── multi-cellular-repr/       ← Multi-cellular representations (2025)
├── scfoundation/              ← scFoundation (2024) — 📗 [→ 笔记](./scfoundation/)
├── scprotein/                 ← scPROTEIN (2024)
├── sclong/                    ← scLong (2024)
├── cell-graphcompass/         ← Cell-GraphCompass (2024)
├── cell-atlas-fm/             ← A cell atlas foundation model (2024) — 📗 [→ 笔记](./cell-atlas-fm/)
├── scaling-dense/             ← Scaling Dense Representations (2024)
├── gene-repr-st/              ← Gene representation on ST (2024)
├── cancerfoundation/          ← CancerFoundation (2024)
├── cell-ontology-fm/          ← Cell-ontology guided FM (2024)
├── scmulan/                   ← scMulan (2024)
├── scgpt/                     ← scGPT (2024) — 📗 [→ 笔记](./scgpt/)
├── langcell/                  ← LangCell (2024) — � [→ stub](./langcell/)
├── cell-niche-graph/          ← Large-scale characterization of cell niches (2024)
├── scmformer/                 ← scmFormer (2024)
├── metadata-as-language/      ← Single-cell metadata as language (2024)
├── genecompass/               ← GeneCompass (2024)
├── saturn/                    ← SATURN (2024)
├── musegnn/                   ← MuSe-GNN (2023)
├── scclip/                    ← scCLIP (2023)
├── sc-mae/                    ← Single-cell Masked Autoencoder (2023)
├── scelmo/                    ← scELMo (2023)
├── divide-conquer-ssl/        ← Large-Scale Cell Representation Learning (2023)
├── cellplm/                   ← CellPLM (2024)
├── dna-to-expression/         ← sc expression prediction from DNA (2023)
├── rnaseq-coverage-dna/       ← RNA-seq coverage from DNA (2023)
├── cellpolaris/               ← CellPolaris (2023)
├── schyena/                   ← scHyena (2023)
├── scpoli/                    ← scPoli (2023)
├── geneformer/                ← Geneformer (2023) — 📗 [→ 笔记](./geneformer/)
├── generative-pretraining/    ← Generative pretraining (2023)
├── xtrimogene/                ← xTrimoGene (2023)
├── sc-expression-lm/          ← A single-cell gene expression language model (2022)
├── scbert/                    ← scBERT (2022)
└── scpretrain/                ← scPretrain (2022)
```

## 状态

| 状态 | 计数 |
|------|------|
| 📗 已有笔记 | 6 (Geneformer, scGPT, scFoundation, UCE/Cell Atlas, Nicheformer, Novae) |
| 📗 新增笔记 | 7 (scBERT, scPoli, scPRINT, GeneCompass, SATURN, EpiAgent, visual-omics-fm) |
| 📄 待补充 | 57 |
| **合计** | **70** |
