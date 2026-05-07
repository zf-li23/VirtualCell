# 🧬 细胞基础模型系统学习笔记

> 覆盖 [awesome-foundation-model-single-cell-papers](https://github.com/OmicsML/awesome-foundation-model-single-cell-papers) 全部 7 大分类、~180 篇论文/模型。
>
> 📗 = 已撰写详细笔记 &emsp; 📄 = 已创建 stub（待补充）&emsp; ⬜ = 暂未收录

---

## 📂 目录结构

```
notes/
├── README.md                  ← 主目录（你在这里）
├── benchmarks/                ← Foundation-Model-Evaluation-For-Single-cell
├── virtual-cell/              ← Virtual cell
├── surveys/                   ← Single-cell-Genomics-ML-Related-Survey-and-Perspective
├── fm/                        ← Foundation-Model-For-Single-cell → FM（核心基础模型）
├── fm-llm/                    ← Foundation-Model-For-Single-cell → FM + LLM
├── perturbation/              ← Foundation-Model-Genetic-Perturbation
└── pathology/                 ← Foundation-Model-For-Pathology
```

---

## 1️⃣ Foundation-Model-Evaluation-For-Single-cell

> 单细胞基础模型的系统评估与 Benchmark。详见 [`benchmarks/`](./benchmarks/)

| # | 年份 | 论文 / 模型 | 状态 |
|---|------|------------|------|
| 1 | 2026 | A Systematic Comparison of Single-Cell Perturbation Response Prediction Models | 📄 |
| 2 | 2026 | A unified framework enables accessible deployment and comprehensive benchmarking of scFMs | 📄 |
| 3 | 2025 | Defining and benchmarking open problems in single-cell analysis (Nat Biotech) | 📄 |
| 4 | 2025 | Benchmarking algorithms for generalizable single-cell perturbation response prediction (Nat Methods) | 📄 |
| 5 | 2025 | Batch Effects Remain a Fundamental Barrier to Universal Embeddings in scFMs | 📄 |
| 6 | 2025 | scCluBench: Comprehensive Benchmarking of Clustering Algorithms for scRNA-seq | 📄 |
| 7 | 2025 | PerturBench: Benchmarking ML Models for Cellular Perturbation Analysis | 📄 |
| 8 | 2025 | Virtual Cell Challenge: Toward a Turing test for the virtual cell (Cell) | 📄 |
| 9 | 2025 | HEIMDALL: A Modular Framework for Tokenization in scFMs | 📄 |
| 10 | 2025 | Empirical Evaluation of scFMs for Predicting Cancer Outcomes | 📄 |
| 11 | 2025 | Benchmarking cell type and gene set annotation by LLMs with AnnDictionary (Nat Comms) | 📄 |
| 12 | 2025 | Sparse Autoencoders Reveal Interpretable Features in scFMs | 📄 |
| 13 | 2025 | Reusability report: Exploring transferability of SSL models from sc to ST (Nat Mach Intell) | 📄 |
| 14 | 2025 | Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines | 📄 |
| 15 | 2025 | Multitask benchmarking of single-cell multimodal omics integration methods (Nat Methods) | 📄 |
| 16 | 2025 | BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic FMs | 📄 |
| 17 | 2025 | Biology-driven insights into the power of scFMs (Genome Biology) | 📄 |
| 18 | 2025 | BMFM-RNA (duplicate) | 📄 |
| 19 | 2025 | Benchmarking gene embeddings from sequence, expression, network, and text models | 📄 |
| 20 | 2025 | Diversity by Design: Addressing Mode Collapse Improves Perturbation Modeling | 📄 |
| 21 | 2025 | Zero-shot evaluation reveals limitations of scFMs (Genome Biology) | 📄 |
| 22 | 2025 | scDrugMap: benchmarking large FMs for drug response prediction (Nat Comms) | 📄 |
| 23 | 2025 | A Systematic Evaluation of scFMs on Cell-Type Classification Task (WSDM) | 📄 |
| 24 | 2024 | Delineating the effective use of SSL in single-cell genomics (Nat Mach Intell) | 📄 |
| 25 | 2024 | BioLLM: A standardized framework for integrating and benchmarking scFMs (Patterns) | 📄 |
| 26 | 2024 | Evaluating the role of pre-training dataset size and diversity on scFM performance | 📄 |
| 27 | 2024 | Deeper evaluation of a single-cell foundation models (Nat Mach Intell) | 📄 |
| 28 | 2024 | Assessing GPT-4 for cell type annotation in scRNA-seq (Nat Methods) | 📄 |
| 29 | 2024 | Metric Mirages in Cell Embeddings | 📄 |
| 30 | 2023 | Reusability report: Learning the transcriptional grammar in scRNA-seq using transformers (Nat Mach Intell) | 📄 |
| 31 | 2023 | A Deep Dive into Single-Cell RNA Sequencing Foundation Models | 📄 |
| 32 | 2023 | scEval: Evaluating the Utilities of LLMs in Single-cell Data Analysis | 📄 |
| 33 | 2023 | Foundation Models Meet Imbalanced Single-Cell Data When Learning Cell Type Annotations | 📄 |
| 34 | 2023 | Evaluation of large language models for discovery of gene set function | 📄 |
| 35 | 2024 | BEND: Benchmarking DNA Language Models on Biologically Meaningful Tasks (ICLR) | 📄 |

---

## 2️⃣ Virtual cell

> 虚拟细胞概念、架构与挑战。详见 [`virtual-cell/`](./virtual-cell/)

| # | 年份 | 论文 / 模型 | 状态 |
|---|------|------------|------|
| 1 | 2025 | Virtual Cell Challenge: Toward a Turing test for the virtual cell (Cell) | 📄 |
| 2 | 2025 | CellForge: Agentic Design of Virtual Cell Models | 📄 |
| 3 | 2025 | Virtual Cells: Predict, Explain, Discover | 📄 |
| 4 | 2025 | VCWorld: A Biological World Model for Virtual Cell Simulation | 📄 |
| 5 | 2025 | AI-driven virtual cell models in preclinical research (npj Digital Medicine) | 📄 |
| 6 | 2025 | The virtual cell (Nature Methods) | 📄 |
| 7 | 2025 | Grow AI virtual cells: three data pillars and closed-loop learning (Cell Research) | 📄 |
| 8 | 2025 | Large Language Models Meet Virtual Cell: A Survey | 📄 |
| 9 | 2024 | How to build the virtual cell with AI: Priorities and opportunities (Cell) | 📄 |

---

## 3️⃣ Single-cell-Genomics-Machine-Learning-Related-Survey-and-Perspective

> 综述与展望。详见 [`surveys/`](./surveys/)

| # | 年份 | 论文 / 模型 | 状态 |
|---|------|------------|------|
| 1 | 2026 | Artificial Intelligence Virtual Organoids (AIVOs) (Bioactive Materials) | 📄 |
| 2 | 2026 | Spatial omics at the forefront (Cancer Cell) | 📄 |
| 3 | 2026 | Interpretation, extrapolation and perturbation of single cells (Nat Rev Genetics) | 📄 |
| 4 | 2025 | Computational strategies for cross-species knowledge transfer (Nat Methods) | 📄 |
| 5 | 2025 | Multimodal foundation transformer models for multiscale genomics (Nat Methods) | 📄 |
| 6 | 2025 | Towards multimodal foundation models in molecular cell biology (Nature) | 📄 |
| 7 | 2025 | Tokenization and deep learning architectures in genomics: A review | 📄 |
| 8 | 2025 | Decoding Cell Fate: Integrated Experimental and Computational Analysis (Bioinformatics) | 📄 |
| 9 | 2025 | A perspective on developing FMs for analyzing spatial transcriptomic data (Quant Bio) | 📄 |
| 10 | 2025 | Insights, opportunities, and challenges provided by large cell atlases (Genome Bio) | 📄 |
| 11 | 2025 | Overcoming barriers to the wide adoption of scLLMs in biomedical research (Nat Biotech) | 📄 |
| 12 | 2025 | Foundation models in bioinformatics (National Science Review) | 📄 |
| 13 | 2025 | Large language models for drug discovery and development (Patterns) | 📄 |
| 14 | 2025 | Single-cell foundation models: bringing AI into cell biology (Exp & Mol Med) | 📄 |
| 15 | 2025 | A survey on foundation language models for single-cell biology (ACL) | 📄 |
| 16 | 2025 | Transformers and genome language models (Nat Mach Intell) | 📄 |
| 17 | 2024 | Harnessing the deep learning power of FMs in single-cell omics (Nat Rev Mol Cell Bio) | 📄 |
| 18 | 2024 | Toward a foundation model of causal cell and tissue biology with a Perturbation Cell and Tissue Atlas (Cell) | 📄 |
| 19 | 2024 | General-purpose pre-trained large cellular models for single-cell transcriptomic (Nat Sci Rev) | 📄 |
| 20 | 2024 | Transformers in single-cell omics: a review and new perspectives (Nat Methods) | 📄 |
| 21 | 2024 | The future of rapid and automated single-cell data analysis using reference mapping (Cell) | 📄 |
| 22 | 2024 | The Human Cell Atlas from a cell census to a unified foundation model (Nature) | 📄 |
| 23 | 2024 | A mini-review on perturbation modelling across single-cell omic modalities | 📄 |
| 24 | 2024 | Progress and opportunities of foundation models in bioinformatics (Brief Bioinfo) | 📄 |

---

## 4️⃣ Foundation-Model-For-Single-cell → FM

> 核心单细胞基础模型。详见 [`fm/`](./fm/)

| # | 年份 | 模型 | 出版 | 状态 |
|---|------|------|------|------|
| 1 | 2026 | Stack: In-Context Learning of Single-Cell Biology | bioRxiv | 📄 [→ stub](./fm/stack/) |
| 2 | 2025 | GeneJepa: A Predictive World Model of the Transcriptome | bioRxiv | 📄 [→ stub](./fm/genejepa/) |
| 3 | 2025 | Unified modeling of cellular responses to diverse perturbation types | bioRxiv | 📄 [→ stub](./fm/unified-perturbation/) |
| 4 | 2025 | OmniCell: Unified FM of sc and ST | bioRxiv | 📄 [→ stub](./fm/omnicell/) |
| 5 | 2025 | Bidirectional Mamba for Single-Cell Data | bioRxiv | 📄 [→ stub](./fm/bidir-mamba/) |
| 6 | 2025 | A Foundation Model for Spatial Proteomics | bioRxiv | 📄 [→ stub](./fm/spatial-proteomics-fm/) |
| 7 | 2025 | scPEFT: Parameter-efficient fine-tuning of scLLMs | Nat Mach Intell | 📄 [→ stub](./fm/scpeft/) |
| 8 | 2025 | scPRINT-2: Next-gen cell FMs and benchmarks | bioRxiv | 📄 [→ stub](./fm/scprint-2/) |
| 9 | 2025 | **Novae**: graph-based FM for ST | **Nat Methods** | 📗 [→ 笔记](./fm/novae/) |
| 10 | 2025 | STPath: generative FM for ST + WSI | npj Digital Med | 📄 [→ stub](./fm/stpath/) |
| 11 | 2025 | PULSAR: Multi-scale and Multicellular Biology FM | bioRxiv | 📄 [→ stub](./fm/pulsar/) |
| 12 | 2025 | SWITCH: Integrative DL of spatial multi-omics | Nat Comput Sci | 📄 [→ stub](./fm/switch/) |
| 13 | 2025 | SpaTranslator: universal spatial multi-omics translation | bioRxiv | 📄 [→ stub](./fm/spatranslator/) |
| 14 | 2025 | scConcept: Contrastive pretraining for technology-agnostic sc representations | bioRxiv | 📄 [→ stub](./fm/scconcept/) |
| 15 | 2025 | Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models | arXiv | 📄 [→ stub](./fm/latent-diffusion-sc/) |
| 16 | 2025 | TranscriptFormer: Cross-species generative cell atlas | bioRxiv | 📄 [→ stub](./fm/transcriptformer/) |
| 17 | 2025 | Atacformer: transformer-based FM for ATAC-seq | bioRxiv | 📄 [→ stub](./fm/atacformer/) |
| 18 | 2025 | EpiFoundation: FM for scATAC-seq via Peak-to-Gene Alignment | bioRxiv | 📄 [→ stub](./fm/epifoundation/) |
| 19 | 2025 | ChromFound: Universal FM for Chromatin Accessibility | NeurIPS | 📄 [→ stub](./fm/chromfound/) |
| 20 | 2025 | **Nicheformer**: FM for sc and spatial omics | **Nat Methods** | 📗 [→ 笔记](./fm/nicheformer/) |
| 21 | 2025 | scLinguist: Hyena-based cross-modality translation | bioRxiv | 📄 [→ stub](./fm/sclinguist/) |
| 22 | 2025 | A foundation model of transcription across human cell types | **Nature** | 📄 [→ stub](./fm/transcription-foundation/) |
| 23 | 2025 | **scPRINT**: pre-training on 50M cells for gene network predictions | Nat Comms | 📗 [→ 笔记](./fm/scprint/) |
| 24 | 2025 | **A visual–omics foundation model** to bridge histopathology with ST | **Nat Methods** | 📗 [→ 笔记](./fm/visual-omics-fm/) |
| 25 | 2025 | **EpiAgent**: FM for single-cell epigenomics | **Nat Methods** | 📗 [→ 笔记](./fm/epiagent/) |
| 26 | 2025 | CAPTAIN: multimodal FM for co-assayed RNA and protein | bioRxiv | 📄 [→ stub](./fm/captain/) |
| 27 | 2025 | scGPT-spatial: Continual Pretraining of scGPT for ST | bioRxiv | 📄 [→ stub](./fm/scgpt-spatial/) |
| 28 | 2025 | SToFM: Multi-scale FM for ST | BioRxiv | 📄 [→ stub](./fm/stofm/) |
| 29 | 2025 | SCARF: Single Cell ATAC-seq and RNA-seq FM | BioRxiv | 📄 [→ stub](./fm/scarf/) |
| 30 | 2025 | Multimodal FM predicts zero-shot functional perturbations and cell fate | BioRxiv | 📄 [→ stub](./fm/multimodal-perturbation/) |
| 31 | 2025 | CellFM: FM pre-trained on 100M human cells | Nat Comms | 📄 [→ stub](./fm/cellfm/) |
| 32 | 2025 | Tabula: Tabular Self-Supervised FM for sc Transcriptomics | NeurIPS | 📄 [→ stub](./fm/tabula/) |
| 33 | 2025 | A pre-trained large generative model for translating sc transcriptome to proteome | Nat Biomed Eng | 📄 [→ stub](./fm/transcriptome-proteome/) |
| 34 | 2025 | Toward a privacy-preserving predictive FM with federated learning and tabular modeling | BioRxiv | 📄 [→ stub](./fm/privacy-federated/) |
| 35 | 2025 | **scNET**: learning context-specific gene/cell embeddings with PPI | **Nat Methods** | 📄 [→ stub](./fm/scnet/) |
| 36 | 2025 | Learning multi-cellular representations for patient-level disease states | RECOMB | 📄 [→ stub](./fm/multi-cellular-repr/) |
| 37 | 2024 | **scFoundation**: Large Scale FM on sc Transcriptomics | **Nat Methods** | 📗 [→ 笔记](./fm/scfoundation/) |
| 38 | 2024 | scPROTEIN: versatile deep graph contrastive learning for sc proteomics | **Nat Methods** | 📄 [→ stub](./fm/scprotein/) |
| 39 | 2024 | scLong: Billion-Parameter FM for Long-Range Gene Context | bioRxiv | 📄 [→ stub](./fm/sclong/) |
| 40 | 2024 | Cell-GraphCompass: Modeling Single Cells with Graph Structure FM | Nat Sci Rev | 📄 [→ stub](./fm/cell-graphcompass/) |
| 41 | 2024 | **A cell atlas foundation model** for scalable search of similar human cells | **Nature** | 📗 [→ 笔记](./fm/cell-atlas-fm/) |
| 42 | 2024 | Scaling Dense Representations for Single Cell with Transcriptome-Scale Context | BioRxiv | 📄 [→ stub](./fm/scaling-dense/) |
| 43 | 2024 | A framework for gene representation on spatial transcriptomics | BioRxiv | 📄 [→ stub](./fm/gene-repr-st/) |
| 44 | 2024 | CancerFoundation: scRNA-seq FM to decipher drug resistance | BioRxiv | 📄 [→ stub](./fm/cancerfoundation/) |
| 45 | 2024 | Cell-ontology guided transcriptome foundation model | NeurIPS | 📄 [→ stub](./fm/cell-ontology-fm/) |
| 46 | 2024 | scMulan: a multitask generative pre-trained language model for sc analysis | RECOMB | 📄 [→ stub](./fm/scmulan/) |
| 47 | 2024 | **scGPT**: toward building a FM for sc multi-omics using generative AI | **Nat Methods** | 📗 [→ 笔记](./fm/scgpt/) |
| 48 | 2024 | **LangCell**: Language-Cell Pre-training for Cell Identity Understanding | ICML | � [→ stub](./fm/langcell/) |
| 49 | 2024 | Large-scale characterization of cell niches in spatial atlases using bio-inspired graph learning | bioRxiv | 📄 [→ stub](./fm/cell-niche-graph/) |
| 50 | 2024 | scmFormer: Integrates sc Proteomics and Transcriptomics by Multi-Task Transformer | bioRxiv | 📄 [→ stub](./fm/scmformer/) |
| 51 | 2024 | Single-cell metadata as language | Blog | 📄 [→ stub](./fm/metadata-as-language/) |
| 52 | 2024 | **GeneCompass**: Knowledge-Informed Cross-Species FM | Cell Research | 📗 [→ 笔记](./fm/genecompass/) |
| 53 | 2024 | **SATURN**: universal cell embeddings across species | **Nat Methods** | 📗 [→ 笔记](./fm/saturn/) |
| 54 | 2023 | MuSe-GNN: Unified Gene Representation from Multimodal Biological Graph Data | NeurIPS | 📄 [→ stub](./fm/musegnn/) |
| 55 | 2023 | scCLIP: Multi-modal Single-cell Contrastive Learning Integration Pre-training | NeurIPS Workshop | 📄 [→ stub](./fm/scclip/) |
| 56 | 2023 | Single-cell Masked Autoencoder: Accurate and Interpretable Immunophenotyper | NeurIPS Workshop | 📄 [→ stub](./fm/sc-mae/) |
| 57 | 2023 | scELMo: Embeddings from Language Models for Single-cell Data Analysis | bioRxiv | 📄 [→ stub](./fm/scelmo/) |
| 58 | 2023 | Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning | bioRxiv | 📄 [→ stub](./fm/divide-conquer-ssl/) |
| 59 | 2024 | CellPLM: Pre-training of Cell Language Model Beyond Single Cells | ICLR | 📄 [→ stub](./fm/cell-plm/) |
| 60 | 2023 | Single-cell gene expression prediction from DNA sequence at large contexts | bioRxiv | 📄 [→ stub](./fm/dna-to-expression/) |
| 61 | 2023 | Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation | bioRxiv | 📄 [→ stub](./fm/rnaseq-coverage-dna/) |
| 62 | 2023 | CellPolaris: Decoding Cell Fate through Transfer Learning of GRNs | bioRxiv | 📄 [→ stub](./fm/cellpolaris/) |
| 63 | 2023 | scHyena: FM for Full-Length scRNA-Seq Analysis in Brain | bioRxiv | 📄 [→ stub](./fm/schiena/) |
| 64 | 2023 | **scPoli**: Population-level integration of sc datasets enables multi-scale analysis | **Nat Methods** | 📗 [→ 笔记](./fm/scpoli/) |
| 65 | 2023 | **Transfer learning enables predictions in network biology** (Geneformer) | **Nature** | 📗 [→ 笔记](./fm/geneformer/) |
| 66 | 2023 | Generative pretraining from large-scale transcriptomes for sc deciphering | iScience | 📄 [→ stub](./fm/generative-pretraining/) |
| 67 | 2023 | xTrimoGene: Efficient and Scalable Representation Learner for scRNA-Seq | NeurIPS | 📄 [→ stub](./fm/xtrimogene/) |
| 68 | 2022 | A single-cell gene expression language model | arXiv | 📄 [→ stub](./fm/sc-expression-lm/) |
| 69 | 2022 | **scBERT**: large-scale pretrained deep language model for cell type annotation | Nat Mach Intell | 📗 [→ 笔记](./fm/scbert/) |
| 70 | 2022 | scPretrain: multi-task self-supervised learning for cell-type classification | Bioinformatics | 📄 [→ stub](./fm/scpretrain/) |

---

## 5️⃣ Foundation-Model-For-Single-cell → FM + LLM

> 基础模型与大语言模型的结合。详见 [`fm-llm/`](./fm-llm/)

| # | 年份 | 模型 | 出版 | 状态 |
|---|------|------|------|------|
| 1 | 2026 | OKR-Cell: Open World Knowledge Aided scFM with Cross-Modal Cell-Language Pre-training | bioRxiv | 📄 |
| 2 | 2026 | spEMO: Multi-Modal FMs for Spatial Multi-Omic and Histopathology Data | Nat Biomed Eng | 📄 |
| 3 | 2025 | Scouter: predicts transcriptional responses with LLM embeddings | Nat Comput Sci | 📄 |
| 4 | 2025 | CASSIA: multi-agent LLM for automated cell annotation | Nat Comms | 📄 |
| 5 | 2025 | CellForge: Agentic Design of Virtual Cell Models | bioRxiv | 📄 |
| 6 | 2025 | TISSUENARRATOR: Generative Modeling of ST with LLMs | bioRxiv | 📄 |
| 7 | 2025 | Multimodal learning enables chat-based exploration of single-cell data | **Nat Biotech** | 📄 |
| 8 | 2025 | CellHermes: Harmonizing multimodal data for omics understanding | bioRxiv | 📄 |
| 9 | 2025 | CellTok: Early-Fusion Multimodal LLM for sc Transcriptomics via Tokenization | bioRxiv | 📄 |
| 10 | 2025 | LLM Consensus Improves Cell Type Annotation Accuracy for scRNA-seq | bioRxiv | 📄 |
| 11 | 2025 | Towards Applying LLMs to Complement scFMs | bioRxiv | 📄 |
| 12 | 2025 | Language-Enhanced Representation Learning for sc Transcriptomics | bioRxiv | 📄 |
| 13 | 2025 | sciLaMA: sc Representation Learning with Prior Knowledge from LLMs | ICML | 📄 |
| 14 | 2025 | Scaling LLMs for Next-Generation Single-Cell Analysis | bioRxiv | 📄 |
| 15 | 2025 | Simple and effective embedding model for sc biology built from ChatGPT | Nat Biomed Eng | 📄 |
| 16 | 2025 | TEDDY: A Family of FMs for Understanding Single Cell Biology | ICML Workshop | 📄 |
| 17 | 2024 | Joint embedding of transcriptomes and text for interactive scRNA-seq exploration | ICLR Workshop | 📄 |
| 18 | 2024 | CELLama: FM for sc and ST by Cell Embedding Leveraging LLM Abilities | bioRxiv | 📄 |
| 19 | 2024 | scChat: LLM-Powered Co-Pilot for Contextualized scRNA-seq Analysis | bioRxiv | 📄 |
| 20 | 2024 | Cell2Sentence: Teaching LLMs the Language of Biology | ICML | 📄 |
| 21 | 2024 | scReader: Prompting LLMs to Interpret scRNA-seq Data | arXiv | 📄 |
| 22 | 2024 | Assessing GPT-4 for cell type annotation in scRNA-seq (dup) | Nat Methods | 📄 |

---

## 6️⃣ Foundation-Model-Genetic-Perturbation

> 遗传扰动响应预测基础模型。详见 [`perturbation/`](./perturbation/)

| # | 年份 | 模型 | 出版 | 状态 |
|---|------|------|------|------|
| 1 | 2025 | Diffusion-based debiasing for transcriptional response prediction | PNAS | 📄 |
| 2 | 2025 | Tahoe-100M: Giga-Scale sc Perturbation Atlas | bioRxiv | 📄 |
| 3 | 2025 | Tahoe-x1: Scaling Perturbation-Trained scFMs to 3B Parameters | bioRxiv | 📄 |
| 4 | 2025 | SpatialProp: tissue perturbation modeling with spatial ST | bioRxiv | 📄 |
| 5 | 2025 | PertAdapt: Unlocking scFMs for Genetic Perturbation Prediction | bioRxiv | 📄 |
| 6 | 2025 | In silico biological discovery with large perturbation models | Nat Comput Sci | 📄 |
| 7 | 2025 | Systema: evaluating genetic perturbation response prediction | Nat Biotech | 📄 |
| 8 | 2025 | scFME: scFM Evaluation for In-Silico Perturbation | bioRxiv | 📄 |
| 9 | 2025 | Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines | **Nat Methods** | 📄 |
| 10 | 2025 | STATE: Predicting cellular responses to perturbation across diverse contexts | bioRxiv | 📄 |
| 11 | 2024 | scLAMBDA: Modeling and predicting multi-gene perturbation responses | bioRxiv | 📄 |
| 12 | 2024 | Benchmarking a foundational cell model for post-perturbation RNAseq prediction | bioRxiv | 📄 |
| 13 | 2024 | Benchmarking Transcriptomics FMs for Perturbation Analysis: one PCA still rules them all | bioRxiv | 📄 |
| 14 | 2024 | PertEval-scFM: Benchmarking scFMs for Perturbation Effect Prediction | ICML | 📄 |
| 15 | 2024 | scGenePT: Is language all you need for modeling sc perturbations? | BioRxiv | 📄 |
| 16 | 2023 | CINEMA-OT: Causal identification of sc perturbation effects | Nat Methods | 📄 |

---

## 7️⃣ Foundation-Model-For-Pathology

> 病理学基础模型。详见 [`pathology/`](./pathology/)

| # | 年份 | 模型 | 出版 | 状态 |
|---|------|------|------|------|
| 1 | 2024 | A FM for joint segmentation, detection and recognition of biomedical objects across 9 modalities | Nat Methods | 📄 |
| 2 | 2024 | A whole-slide FM for digital pathology from real-world data | **Nature** | 📄 |
| 3 | 2024 | Towards a general-purpose FM for computational pathology | Nat Medicine | 📄 |
| 4 | 2024 | A visual-language FM for computational pathology | Nat Medicine | 📄 |
| 5 | 2023 | A visual–language FM for pathology image analysis using medical Twitter | Nat Medicine | 📄 |

---

## 状态图例

| 图标 | 含义 |
|------|------|
| 📗 | 已撰写完整学习笔记，链接到对应目录 |
| 📄 | 已创建 stub README，等待补充 |
| ⬜ | 尚未收录，计划中 |

---

## 学习进度

```
📗 已完成:  14 / ~180  (8%)
📄 Stub:    ~166 / ~180 (92%)
⬜ 待办:      0 / ~180  (0%)
```

---

## 导航

| 你想做什么 | 去哪里 |
|-----------|--------|
| 查看全部论文目录 | [`papers.md`](../papers.md) |
| 查看笔记总目录 | 本文件 |
| 查看学习模板 | [`_template.md`](./_template.md) |
| 查看某个模型笔记 | `fm/<模型名>/README.md` |

*最后更新：2026-05-07*
