#!/usr/bin/env python3
import re, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(BASE, "notes")

# dirname -> (model_name, paper_title, url, venue)
M = {
"atacformer":("Atacformer","Atacformer: A Transformer-based Foundation Model for Analysis and Interpretation of ATAC-seq Data","https://www.biorxiv.org/content/10.1101/2025.11.03.685753v1","bioRxiv 2025"),
"bidir-mamba":("Bidirectional Mamba","Bidirectional Mamba for Single-Cell Data: Efficient Context Learning with Biological Fidelity","https://arxiv.org/abs/2504.16956","bioRxiv 2025"),
"cancerfoundation":("CancerFoundation","CancerFoundation: A Single-cell RNA Sequencing Foundation Model to Decipher Drug Resistance in Cancer","https://www.biorxiv.org/content/10.1101/2024.11.01.621087v1.full.pdf","bioRxiv 2024"),
"captain":("CAPTAIN","CAPTAIN: A Multimodal Foundation Model Pretrained on Co-assayed Single-cell RNA and Protein","https://www.biorxiv.org/content/10.1101/2025.07.07.663366v1","bioRxiv 2025"),
"cell-graphcompass":("Cell-GraphCompass","Cell-GraphCompass: Modeling Single Cells with Graph Structure Foundation Model","https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwaf255/8172492","National Science Review 2024"),
"cell-niche-graph":("Cell Niche Graph","Large-scale Characterization of Cell Niches in Spatial Atlases Using Bio-inspired Graph Learning","https://www.biorxiv.org/content/10.1101/2024.02.21.581428v1","bioRxiv 2024"),
"cell-ontology-fm":("Cell Ontology FM","Cell-ontology Guided Transcriptome Foundation Model","https://openreview.net/forum?id=aeYNVtTo7o","NeurIPS 2024"),
"cell-plm":("CellPLM","CellPLM: Pre-training of Cell Language Model Beyond Single Cells","https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1","ICLR 2024"),
"cellfm":("CellFM","CellFM: A Large-scale Foundation Model Pre-trained on Transcriptomics of 100 Million Human Cells","https://www.nature.com/articles/s41467-025-59926-5","Nature Communications 2025"),
"cellpolaris":("CellPolaris","CellPolaris: Decoding Cell Fate through Generalization Transfer Learning of Gene Regulatory Networks","https://www.biorxiv.org/content/10.1101/2023.09.25.559244v1","bioRxiv 2023"),
"chromfound":("ChromFound","ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data","https://arxiv.org/abs/2505.12638","NeurIPS 2025"),
"divide-conquer-ssl":("Divide-and-Conquer SSL","Large-Scale Cell Representation Learning via Divide-and-Conquer Contrastive Learning","https://arxiv.org/pdf/2306.04371.pdf","bioRxiv 2023"),
"dna-to-expression":("DNA to Expression","Single-cell Gene Expression Prediction from DNA Sequence at Large Contexts","https://www.biorxiv.org/content/10.1101/2023.07.26.550634v1.full","bioRxiv 2023"),
"epifoundation":("EpiFoundation","EpiFoundation: A Foundation Model for Single-Cell ATAC-seq via Peak-to-Gene Alignment","https://www.biorxiv.org/content/10.1101/2025.02.05.636688v2","bioRxiv 2025"),
"gene-repr-st":("Gene Representation ST","A Framework for Gene Representation on Spatial Transcriptomics","https://www.biorxiv.org/content/10.1101/2024.09.27.615337v5","bioRxiv 2024"),
"genejepa":("GeneJepa","GeneJepa: A Predictive World Model of the Transcriptome","https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1","bioRxiv 2025"),
"generative-pretraining":("Generative Pretraining","Generative Pretraining from Large-scale Transcriptomes for Single-cell Deciphering","https://www.sciencedirect.com/science/article/pii/S2589004223006132","iScience 2023"),
"langcell":("LangCell","LangCell: Language-Cell Pre-training for Cell Identity Understanding","https://dl.acm.org/doi/10.5555/3692070.3694600","ICML 2024"),
"latent-diffusion-sc":("Latent Diffusion SC","Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models","https://arxiv.org/abs/2511.02986","arXiv 2025"),
"metadata-as-language":("Metadata as Language","Single-cell Metadata as Language","https://www.nxn.se/valent/2024/2/4/single-cell-metadata-as-language","2024"),
"multi-cellular-repr":("Multi-Cellular Repr","Learning Multi-cellular Representations of Single-cell Transcriptomics Data","https://link.springer.com/chapter/10.1007/978-3-031-90252-9_27","RECOMB 2025"),
"multimodal-perturbation":("Multimodal Perturbation","Multimodal Foundation Model Predicts Zero-shot Functional Perturbations","https://www.biorxiv.org/content/10.1101/2024.12.19.629561v2","bioRxiv 2025"),
"musegnn":("MuSe-GNN","MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data","https://openreview.net/forum?id=4UCktT9XZx","NeurIPS 2023"),
"omnicell":("OmniCell","OmniCell: Unified Foundation Modeling of Single-Cell and Spatial Transcriptomics","https://www.biorxiv.org/content/10.64898/2025.12.29.696804v1","bioRxiv 2025"),
"privacy-federated":("Privacy-Preserving FM","Toward a Privacy-preserving Predictive Foundation Model with Federated Learning","https://www.biorxiv.org/content/10.1101/2025.01.06.631427v1","bioRxiv 2025"),
"pulsar":("PULSAR","PULSAR: A Foundation Model for Multi-scale and Multicellular Biology","https://www.biorxiv.org/content/10.1101/2025.11.24.685470v1","bioRxiv 2025"),
"rnaseq-coverage-dna":("RNA-seq Coverage from DNA","Predicting RNA-seq Coverage from DNA Sequence as a Unifying Model of Gene Regulation","https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1","bioRxiv 2023"),
"sc-expression-lm":("sc Expression LM","A Single-cell Gene Expression Language Model","https://arxiv.org/abs/2210.14330","arXiv 2022"),
"sc-mae":("scMAE","Single-cell Masked Autoencoder: An Accurate and Interpretable Automated Immunophenotyper","https://openreview.net/pdf?id=2mq6uezuGj","NeurIPS AI4Sci 2023"),
"scaling-dense":("Scaling Dense","Scaling Dense Representations for Single Cell with Transcriptome-Scale Context","https://www.biorxiv.org/content/10.1101/2024.11.28.625303v1.full","bioRxiv 2024"),
"scaraf":("SCARF","SCARF: Single Cell ATAC-seq and RNA-seq Foundation Model","https://www.biorxiv.org/content/10.1101/2025.04.07.647689v1","bioRxiv 2025"),
"scarf":("SCARF","SCARF: Single Cell ATAC-seq and RNA-seq Foundation Model","https://www.biorxiv.org/content/10.1101/2025.04.07.647689v1","bioRxiv 2025"),
"scclip":("scCLIP","scCLIP: Multi-modal Single-cell Contrastive Learning Integration Pre-training","https://openreview.net/pdf?id=KMtM5ZHxct","NeurIPS AI4Sci 2023"),
"scconcept":("scConcept","scConcept: Contrastive Pretraining for Technology-agnostic Single-cell Representations","https://www.biorxiv.org/content/10.1101/2025.10.14.682419v1","bioRxiv 2025"),
"scelmo":("scELMo","scELMo: Embeddings from Language Models are Good Learners for Single-cell Data Analysis","https://www.biorxiv.org/content/10.1101/2023.12.07.569910v1.full.pdf","bioRxiv 2023"),
"scgpt-spatial":("scGPT-spatial","scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics","https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1","bioRxiv 2025"),
"schiena":("scHyena","scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain","https://arxiv.org/abs/2310.02713","bioRxiv 2023"),
"sclinguist":("scLinguist","scLinguist: A Pre-trained Hyena-based Foundation Model for Cross-modality Translation","https://www.biorxiv.org/content/10.1101/2025.09.30.679123v1","bioRxiv 2025"),
"sclong":("scLong","scLong: A Billion-Parameter Foundation Model for Capturing Long-Range Gene Context","https://www.biorxiv.org/content/10.1101/2024.11.09.622759v2","bioRxiv 2024"),
"scmformer":("scmFormer","scmFormer Integrates Large-Scale Single-Cell Proteomics and Transcriptomics Data","https://pubmed.ncbi.nlm.nih.gov/38483032/","bioRxiv 2024"),
"scmulan":("scMulan","scMulan: A Multitask Generative Pre-trained Language Model for Single-cell Analysis","https://www.biorxiv.org/content/10.1101/2024.01.25.577152v1","RECOMB 2024"),
"scnet":("scNET","scNET: Learning Context-specific Gene and Cell Embeddings by Integrating Single-cell Gene Expression Data with Protein-Protein Interactions","https://www.nature.com/articles/s41592-025-02627-0","Nature Methods 2025"),
"scpeft":("scPEFT","Harnessing the Power of Single-cell Large Language Models with Parameter-efficient Fine-tuning Using scPEFT","https://www.nature.com/articles/s42256-025-01170-z","Nature Machine Intelligence 2025"),
"scpretrain":("scPretrain","scPretrain: Multi-task Self-supervised Learning for Cell-type Classification","https://academic.oup.com/bioinformatics/article/38/6/1607/6499287","Bioinformatics 2022"),
"scprint-2":("scPRINT-2","scPRINT-2: Towards the Next-generation of Cell Foundation Models and Benchmarks","https://www.biorxiv.org/content/10.64898/2025.12.11.693702v2","bioRxiv 2025"),
"scprotein":("scPROTEIN","scPROTEIN: A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding","https://www.nature.com/articles/s41592-024-02214-9","Nature Methods 2024"),
"spatial-proteomics-fm":("Spatial Proteomics FM","A Foundation Model for Spatial Proteomics","https://arxiv.org/abs/2506.03373","bioRxiv 2025"),
"spatranslator":("SpaTranslator","SpaTranslator: A Deep Generative Framework for Universal Spatial Multi-omics Cross-modality Translation","https://www.biorxiv.org/content/10.1101/2025.11.15.688644v1","bioRxiv 2025"),
"stack":("Stack","Stack: In-Context Learning of Single-Cell Biology","https://www.biorxiv.org/content/10.64898/2026.01.09.698608v1","bioRxiv 2026"),
"stofm":("SToFM","SToFM: A Multi-scale Foundation Model for Spatial Transcriptomics","https://arxiv.org/abs/2507.11588","bioRxiv 2025"),
"stpath":("STPath","STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole-slide Images","https://www.nature.com/articles/s41746-025-02020-3","npj Digital Medicine 2025"),
"switch":("SWITCH","Integrative Deep Learning of Spatial Multi-omics with SWITCH","https://www.nature.com/articles/s43588-025-00891-w","Nature Computational Science 2025"),
"tabulam":("Tabula","Tabula: A Tabular Self-Supervised Foundation Model for Single-Cell Transcriptomics","https://neurips.cc/virtual/2025/poster/117659","NeurIPS 2025"),
"transcriptformer":("TranscriptFormer","A Cross-Species Generative Cell Atlas: The TranscriptFormer Single-cell Model","http://biorxiv.org/content/10.1101/2025.04.25.650731v2","bioRxiv 2025"),
"transcription-foundation":("Transcription Foundation","A Foundation Model of Transcription Across Human Cell Types","https://www.nature.com/articles/s41586-024-08391-z","Nature 2025"),
"transcriptome-proteome":("Transcriptome-Proteome","A Pre-trained Large Generative Model for Translating Single-cell Transcriptome to Proteome","https://www.biorxiv.org/content/10.1101/2023.07.04.547619v2","Nature Biomedical Engineering 2025"),
"unified-perturbation":("Unified Perturbation","Unified Modeling of Cellular Responses to Diverse Perturbation Types","https://www.biorxiv.org/content/10.1101/2025.11.13.688367v1","bioRxiv 2025"),
"xtrimogene":("xTrimoGene","xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data","https://openreview.net/forum?id=gdwcoBCMVi","NeurIPS 2023"),
"cassia":("CASSIA","CASSIA: A Multi-agent Large Language Model for Automated and Interpretable Cell Annotation","https://www.nature.com/articles/s41467-025-67084-x","Nature Communications 2025"),
"cell2sentence":("Cell2Sentence","Cell2Sentence: Teaching Large Language Models the Language of Biology","https://www.biorxiv.org/content/10.1101/2023.09.11.557287v1","ICML 2024"),
"cellama":("CELLama","CELLama: Foundation Model for Single Cell and Spatial Transcriptomics by Cell Embedding Leveraging Language Model Abilities","https://www.biorxiv.org/content/10.1101/2024.05.08.593094v1","bioRxiv 2024"),
"cellforge":("CellForge","CellForge: Agentic Design of Virtual Cell Models","https://arxiv.org/abs/2508.02276","bioRxiv 2025"),
"cellhermes":("CellHermes","Language May Be All Omics Needs: Harmonizing Multimodal Data for Omics Understanding with CellHermes","https://www.biorxiv.org/content/10.1101/2025.11.07.687322v1","bioRxiv 2025"),
"celltok":("CellTok","CellTok: Early-Fusion Multimodal Large Language Model for Single-Cell Transcriptomics","https://www.biorxiv.org/content/10.1101/2025.10.22.684047v1","bioRxiv 2025"),
"chat-based-sc-exploration":("Chat-based sc Exploration","Multimodal Learning Enables Chat-based Exploration of Single-cell Data","https://www.nature.com/articles/s41587-025-02857-9","Nature Biotechnology 2025"),
"chatgpt-embedding-sc":("ChatGPT Embedding SC","Simple and Effective Embedding Model for Single-Cell Biology Built from ChatGPT","https://www.nature.com/articles/s41551-024-01284-6","Nature Biomedical Engineering 2025"),
"gpt4-cell-annotation":("GPT-4 Cell Annotation","Assessing GPT-4 for Cell Type Annotation in Single-cell RNA-seq Analysis","https://www.nature.com/articles/s41592-024-02235-4","Nature Methods 2024"),
"joint-embed-transcript-text":("Joint Embed Transcript Text","Joint Embedding of Transcriptomes and Text Enables Interactive scRNA-seq Data Exploration","https://openreview.net/forum?id=yWiZaE4k3K","ICLR Workshop MLGenX 2024"),
"language-enhanced-repr":("Language Enhanced Repr","Language-Enhanced Representation Learning for Single-Cell Transcriptomics","https://arxiv.org/abs/2503.09427","bioRxiv 2025"),
"llm-complement-scfm":("LLM Complement SCFM","Towards Applying Large Language Models to Complement Single-Cell Foundation Models","https://arxiv.org/abs/2507.10039","bioRxiv 2025"),
"llm-consensus-annotation":("LLM Consensus Annotation","Large Language Model Consensus Improves Cell Type Annotation Accuracy","https://www.biorxiv.org/content/10.1101/2025.04.10.647852v1","bioRxiv 2025"),
"okr-cell":("OKR-Cell","OKR-Cell: Open World Knowledge Aided Single-Cell Foundation Model","https://www.biorxiv.org/content/10.64898/2026.01.09.698573v1","bioRxiv 2026"),
"scaling-llm-sc":("Scaling LLM SC","Scaling Large Language Models for Next-Generation Single-Cell Analysis","https://www.biorxiv.org/content/10.1101/2025.04.14.648850v1","bioRxiv 2025"),
"scchat":("scChat","scChat: A Large Language Model-Powered Co-Pilot for Contextualized scRNA-seq Analysis","https://www.biorxiv.org/content/10.1101/2024.10.01.616063v1.full.pdf+html","bioRxiv 2024"),
"scilama":("sciLaMA","sciLaMA: A Single-Cell Representation Learning Framework to Leverage Prior Knowledge from LLMs","https://openreview.net/forum?id=0m4VsLwj5s","ICML 2025"),
"scouter":("Scouter","Scouter Predicts Transcriptional Responses to Genetic Perturbations with LLM Embeddings","https://www.nature.com/articles/s43588-025-00912-8","Nature Computational Science 2025"),
"screader":("scReader","scReader: Prompting Large Language Models to Interpret scRNA-seq Data","https://arxiv.org/abs/2412.18156","arXiv 2024"),
"spemo":("spEMO","spEMO: Leveraging Multi-Modal Foundation Models for Analyzing Spatial Multi-Omic Data","https://www.biorxiv.org/content/10.1101/2025.01.13.632818v3","Nature Biomedical Engineering 2026"),
"teddy":("TEDDY","TEDDY: A Family of Foundation Models for Understanding Single Cell Biology","https://arxiv.org/pdf/2503.03485","ICML Workshop GenBio 2025"),
"tissuenarrator":("TissueNarrator","TISSUENARRATOR: Generative Modeling of Spatial Transcriptomics with LLMs","https://www.biorxiv.org/content/10.1101/2025.11.24.690325v1.full.pdf","bioRxiv 2025"),
"anndictionary":("AnnDictionary","Benchmarking Cell Type and Gene Set Annotation by LLMs with AnnDictionary","https://www.nature.com/articles/s41467-025-64511-x","Nature Communications 2025"),
"batch-effects-barrier":("Batch Effects Barrier","Batch Effects Remain a Fundamental Barrier to Universal Embeddings in scFMs","https://www.biorxiv.org/content/10.64898/2025.12.19.695371v1","bioRxiv 2025"),
"bend":("BEND","BEND: Benchmarking DNA Language Models on Biologically Meaningful Tasks","https://openreview.net/pdf?id=uKB4cFNQFg","ICLR 2024"),
"biollm":("BioLLM","BioLLM: A Standardized Framework for Integrating and Benchmarking Single-cell Foundation Models","https://www.sciencedirect.com/science/article/pii/S2666389925001746","Patterns 2024"),
"biology-driven-insights":("Biology-driven Insights","Biology-driven Insights into the Power of Single-cell Foundation Models","https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03781-6","Genome Biology 2025"),
"bmfm-rna":("BMFM-RNA","BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models","https://arxiv.org/abs/2506.14861","arXiv 2025"),
"cancer-outcomes-evaluation":("Cancer Outcomes Evaluation","Empirical Evaluation of Single-Cell Foundation Models for Predicting Cancer Outcomes","https://www.biorxiv.org/content/10.1101/2025.10.31.685892v1","bioRxiv 2025"),
"cell-type-classification-eval":("Cell Type Classification Eval","A Systematic Evaluation of Single-Cell Foundation Models on Cell-Type Classification","https://dl.acm.org/doi/10.1145/3701551.3708811","WSDM 2025"),
"deep-dive-scfms":("Deep Dive SCFMs","A Deep Dive into Single-Cell RNA Sequencing Foundation Models","https://www.biorxiv.org/content/10.1101/2023.10.19.563100v1.abstract","bioRxiv 2023"),
"deeper-evaluation-scfms":("Deeper Evaluation SCFMs","Deeper Evaluation of Single-Cell Foundation Models","https://www.nature.com/articles/s42256-024-00949-w","Nature Machine Intelligence 2024"),
"gene-embeddings-benchmark":("Gene Embeddings Benchmark","Benchmarking Gene Embeddings for Functional Prediction Tasks","https://www.biorxiv.org/content/10.1101/2025.01.29.635607v1","bioRxiv 2025"),
"heimdall":("HEIMDALL","HEIMDALL: A Modular Framework for Tokenization in Single-Cell Foundation Models","https://www.biorxiv.org/content/10.1101/2025.11.09.687403v2","bioRxiv 2025"),
"imbalanced-cell-annotation":("Imbalanced Cell Annotation","Foundation Models Meet Imbalanced Single-Cell Data When Learning Cell Type Annotations","https://www.biorxiv.org/content/10.1101/2023.10.24.563625v1","bioRxiv 2023"),
"llm-gene-set-function":("LLM Gene Set Function","Evaluation of Large Language Models for Discovery of Gene Set Function","https://arxiv.org/abs/2309.04019","bioRxiv 2023"),
"metric-mirages":("Metric Mirages","Metric Mirages in Cell Embeddings","https://www.biorxiv.org/content/10.1101/2024.04.02.587824v1","bioRxiv 2024"),
"mode-collapse-perturbation":("Mode Collapse Perturbation","Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling","https://arxiv.org/abs/2506.22641","bioRxiv 2025"),
"multimodal-integration-benchmark":("Multimodal Integration Benchmark","Multitask Benchmarking of Single-cell Multimodal Omics Integration Methods","https://www.nature.com/articles/s41592-025-02856-3","Nature Methods 2025"),
"open-problems-sc-analysis":("Open Problems SC Analysis","Defining and Benchmarking Open Problems in Single-cell Analysis","https://www.nature.com/articles/s41587-025-02694-w","Nature Biotechnology 2025"),
"perturbation-baselines":("Perturbation Baselines","Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines","https://www.biorxiv.org/content/10.1101/2025.10.20.683304v1","bioRxiv 2025"),
"perturbation-benchmarking":("Perturbation Benchmarking","Benchmarking Algorithms for Generalizable Single-cell Perturbation Response Prediction","https://www.nature.com/articles/s41592-025-02980-0","Nature Methods 2025"),
"perturbench":("PerturBench","PerturBench: Benchmarking Machine Learning Models for Cellular Perturbation Analysis","https://arxiv.org/abs/2408.10609","arXiv 2025"),
"pretraining-size-diversity":("Pretraining Size Diversity","Evaluating the Role of Pre-training Dataset Size and Diversity on scFM Performance","https://www.biorxiv.org/content/10.1101/2024.12.13.628448v1","bioRxiv 2024"),
"sccluben":("scCluBench","scCluBench: Comprehensive Benchmarking of Clustering Algorithms for scRNA-seq","https://arxiv.org/abs/2512.02471","arXiv 2025"),
"scdrugmap":("scDrugMap","scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction","https://www.nature.com/articles/s41467-025-67481-2","Nature Communications 2025"),
"sceval":("scEval","Evaluating the Utilities of Large Language Models in Single-cell Data Analysis","https://www.biorxiv.org/content/10.1101/2023.09.08.555192v2","bioRxiv 2023"),
"sparse-autoencoders-scfm":("Sparse Autoencoders SCFM","Sparse Autoencoders Reveal Interpretable Features in Single-Cell Foundation Models","https://www.biorxiv.org/content/10.1101/2025.10.22.681631v1","bioRxiv 2025"),
"ssl-effective-use":("SSL Effective Use","Delineating the Effective Use of Self-supervised Learning in Single-cell Genomics","https://www.nature.com/articles/s42256-024-00934-3","Nature Machine Intelligence 2024"),
"systematic-perturbation-compare":("Systematic Perturbation Compare","A Systematic Comparison of Single-Cell Perturbation Response Prediction Models","https://www.biorxiv.org/content/10.1101/2024.12.23.630036v2.abstract","bioRxiv 2026"),
"transcriptional-grammar":("Transcriptional Grammar","Learning the Transcriptional Grammar in scRNA-seq Data Using Transformers","https://www.nature.com/articles/s42256-023-00757-8","Nature Machine Intelligence 2023"),
"transferability-sc-to-st":("Transferability SC to ST","Exploring the Transferability of SSL Models from Single-cell to Spatial Transcriptomics","https://www.nature.com/articles/s42256-025-01097-5","Nature Machine Intelligence 2025"),
"unified-benchmarking-framework":("Unified Benchmarking Framework","A Unified Framework for Deployment and Benchmarking of Single-cell Foundation Models","https://www.biorxiv.org/content/10.64898/2026.01.06.698060v1","bioRxiv 2026"),
"virtual-cell-challenge":("Virtual Cell Challenge","Virtual Cell Challenge: Toward a Turing Test for the Virtual Cell","https://www.cell.com/cell/fulltext/S0092-8674(25)00675-0","Cell 2025"),
"zero-shot-limitations":("Zero-shot Limitations","Zero-shot Evaluation Reveals Limitations of Single-cell Foundation Models","https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x","Genome Biology 2025"),
"biomedical-seg-det-rec":("Biomedical Seg-Det-Rec","A Foundation Model for Joint Segmentation, Detection and Recognition of Biomedical Objects","https://www.nature.com/articles/s41592-024-02499-w","Nature Methods 2024"),
"general-purpose-pathology":("General Purpose Pathology","Towards a General-purpose Foundation Model for Computational Pathology","https://www.nature.com/articles/s41591-024-02857-3","Nature Medicine 2024"),
"visual-language-pathology":("Visual Language Pathology","A Visual-Language Foundation Model for Computational Pathology","https://www.nature.com/articles/s41591-024-02856-4","Nature Medicine 2024"),
"visual-language-pathology-twitter":("Visual Language Pathology Twitter","A Visual-Language Foundation Model for Pathology Image Analysis Using Medical Twitter","https://www.nature.com/articles/s41591-023-02504-3","Nature Medicine 2023"),
"whole-slide-fm":("Whole-Slide FM","A Whole-Slide Foundation Model for Digital Pathology from Real-World Data","https://www.nature.com/articles/s41586-024-07441-w","Nature 2024"),
"benchmark-cell-model-perturbation":("Benchmark Cell Model Perturbation","Benchmarking a Foundational Cell Model for Post-perturbation RNAseq Prediction","https://www.biorxiv.org/content/biorxiv/early/2024/10/01/2024.09.30.615843.full.pdf","bioRxiv 2024"),
"cinema-ot":("CINEMA-OT","Causal Identification of Single-cell Experimental Perturbation Effects with CINEMA-OT","https://www.nature.com/articles/s41592-023-02040-5","Nature Methods 2023"),
"diffusion-debias-perturbation":("Diffusion Debiasing Perturbation","Predicting the Unseen: A Diffusion-based Debiasing Framework","https://www.pnas.org/doi/10.1073/pnas.2525268122","PNAS 2025"),
"in-silico-discovery":("In Silico Discovery","In Silico Biological Discovery with Large Perturbation Models","https://www.nature.com/articles/s43588-025-00870-1","Nature Computational Science 2025"),
"pca-still-rules":("PCA Still Rules","Benchmarking Transcriptomics Foundation Models for Perturbation Analysis: One PCA Still Rules Them All","https://arxiv.org/abs/2410.13956","bioRxiv 2024"),
"pertadapt":("PertAdapt","PertAdapt: Unlocking scFMs for Genetic Perturbation Prediction","https://www.biorxiv.org/content/10.1101/2025.11.21.689655v1","bioRxiv 2025"),
"perteval-scfm":("PertEval-scFM","PertEval-scFM: Benchmarking Single-Cell Foundation Models for Perturbation Effect Prediction","https://openreview.net/forum?id=t04D9bkKUq","ICML 2024"),
"perturbation-linear-baselines":("Perturbation Linear Baselines","Deep-learning-based Gene Perturbation Effect Prediction Does Not Yet Outperform Simple Linear Baselines","https://www.nature.com/articles/s41592-025-02772-6","Nature Methods 2025"),
"scfme":("scFME","Single Cell Foundation Models Evaluation (scFME) for In-Silico Perturbation","https://www.biorxiv.org/content/10.1101/2025.09.22.677811v1","bioRxiv 2025"),
"scgenept":("scGenePT","scGenePT: Is Language All You Need for Modeling Single-cell Perturbations?","https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1","bioRxiv 2024"),
"sclambda":("scLAMBDA","Modeling and Predicting Single-cell Multi-gene Perturbation Responses with scLAMBDA","https://www.biorxiv.org/content/10.1101/2024.12.04.626878v1","bioRxiv 2024"),
"spatialprop":("SpatialProp","SpatialProp: Tissue Perturbation Modeling with Spatially Resolved scRNA-seq","https://www.biorxiv.org/content/10.64898/2025.11.30.691355v1","bioRxiv 2025"),
"state":("STATE","Predicting Cellular Responses to Perturbation Across Diverse Contexts with STATE","https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2","bioRxiv 2025"),
"systema":("Systema","Systema: A Framework for Evaluating Genetic Perturbation Response Prediction","https://www.nature.com/articles/s41587-025-02777-8","Nature Biotechnology 2025"),
"tahoe-100m":("Tahoe-100M","Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas","https://www.biorxiv.org/content/10.1101/2025.02.20.639398v3","bioRxiv 2025"),
"tahoe-x1":("Tahoe-x1","Tahoe-x1: Scaling Perturbation-Trained scFMs to 3 Billion Parameters","https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1","bioRxiv 2025"),
"aivo":("AIVO","Artificial Intelligence Virtual Organoids (AIVOs)","","Bioactive Materials 2026"),
"barriers-sc-llm":("Barriers SC LLM","Overcoming Barriers to the Wide Adoption of Single-cell LLMs in Biomedical Research","https://www.nature.com/articles/s41587-025-02846-y","Nature Methods 2025"),
"cell-atlases-opportunities":("Cell Atlases Opportunities","Insights, Opportunities, and Challenges Provided by Large Cell Atlases","https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03771-8","Genome Biology 2025"),
"cross-species-transfer":("Cross-Species Transfer","Computational Strategies for Cross-species Knowledge Transfer","https://www.nature.com/articles/s41592-025-02931-9","Nature Methods 2025"),
"decoding-cell-fate":("Decoding Cell Fate","Decoding Cell Fate: Integrated Experimental and Computational Analysis at the Single-Cell Level","https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaf603/8315140","Bioinformatics 2025"),
"fm-bioinformatics":("FM Bioinformatics","Foundation Models in Bioinformatics","https://academic.oup.com/nsr/article/12/4/nwaf028/7979309","National Science Review 2025"),
"fm-spatial-transcriptomic":("FM Spatial Transcriptomic","A Perspective on Developing Foundation Models for Analyzing Spatial Transcriptomic Data","https://onlinelibrary.wiley.com/doi/full/10.1002/qub2.70010","Quantitative Biology 2025"),
"general-purpose-cellular-models":("General Purpose Cellular Models","General-purpose Pre-trained Large Cellular Models for Single-cell Transcriptomic","https://academic.oup.com/nsr/article/11/11/nwae340/7775526","National Science Review 2024"),
"harnessing-fm-omics":("Harnessing FM Omics","Harnessing the Deep Learning Power of Foundation Models in Single-cell Omics","https://www.nature.com/articles/s41580-024-00756-6","Nature Reviews MCB 2024"),
"hca-to-unified-fm":("HCA to Unified FM","The Human Cell Atlas from a Cell Census to a Unified Foundation Model","https://www.nature.com/articles/s41586-024-08338-4","Nature 2024"),
"interpretation-perturbation-sc":("Interpretation Perturbation SC","Interpretation, Extrapolation and Perturbation of Single Cells","https://www.nature.com/articles/s41576-025-00920-4","Nature Reviews Genetics 2026"),
"llm-drug-discovery":("LLM Drug Discovery","Large Language Models for Drug Discovery and Development","https://www.sciencedirect.com/science/article/pii/S2666389925001941","Patterns 2025"),
"mini-review-perturbation-modalities":("Mini-review Perturbation Modalities","A Mini-review on Perturbation Modelling Across Single-cell Omic Modalities","https://www.cell.com/cell/fulltext/S0092-8674(24)01332-1","Comp Struc Biotech J 2024"),
"multimodal-fm-cell-biology":("Multimodal FM Cell Biology","Towards Multimodal Foundation Models in Molecular Cell Biology","https://www.nature.com/articles/s41586-025-08710-y","Nature 2025"),
"multimodal-transformer-genomics":("Multimodal Transformer Genomics","Multimodal Foundation Transformer Models for Multiscale Genomics","https://www.nature.com/articles/s41592-025-02918-6","Nature Methods 2025"),
"perturbation-cell-tissue-atlas":("Perturbation Cell Tissue Atlas","Toward a Foundation Model of Causal Cell and Tissue Biology with a Perturbation Cell and Tissue Atlas","https://www.cell.com/cell/abstract/S0092-8674(24)00829-8","Cell 2024"),
"progress-opportunities-fm-bioinfo":("Progress Opportunities FM Bioinfo","Progress and Opportunities of Foundation Models in Bioinformatics","https://academic.oup.com/bib/article/25/6/bbae548/7842778","Brief in Bioinformatics 2024"),
"reference-mapping-future":("Reference Mapping Future","The Future of Rapid and Automated Single-cell Data Analysis Using Reference Mapping","https://www.cell.com/cell/fulltext/S0092-8674(24)00301-5","Cell 2024"),
"scfm-into-cell-biology":("SCFM into Cell Biology","Single-cell Foundation Models: Bringing Artificial Intelligence into Cell Biology","https://www.nature.com/articles/s12276-025-01547-5","Exp Mol Med 2025"),
"spatial-omics-forefront":("Spatial Omics Forefront","Spatial Omics at the Forefront: Emerging Technologies, Analytical Innovations, and Clinical Applications","https://www.cell.com/cancer-cell/fulltext/S1535-6108(25)00543-4","Cancer Cell 2026"),
"survey-fm-sc-biology":("Survey FM SC Biology","A Survey on Foundation Language Models for Single-cell Biology","https://aclanthology.org/2025.acl-long.26/","ACL 2025"),
"tokenization-review":("Tokenization Review","Tokenization and Deep Learning Architectures in Genomics: A Comprehensive Review","https://pmc.ncbi.nlm.nih.gov/articles/PMC12356405/","Comp Struc Biotech J 2025"),
"transformers-genome-lm":("Transformers Genome LM","Transformers and Genome Language Models","https://www.nature.com/articles/s42256-025-01007-9","Nature Machine Intelligence 2025"),
"transformers-sc-omics-review":("Transformers SC Omics Review","Transformers in Single-cell Omics: A Review and New Perspectives","https://www.nature.com/articles/s41592-024-02353-z","Nature Methods 2024"),
"ai-virtual-cell-preclinical":("AI Virtual Cell Preclinical","AI-driven Virtual Cell Models in Preclinical Research","https://www.nature.com/articles/s41746-025-02198-6","npj Digital Medicine 2025"),
"build-virtual-cell-ai":("Build Virtual Cell AI","How to Build the Virtual Cell with Artificial Intelligence: Priorities and Opportunities","https://www.cell.com/cell/fulltext/S0092-8674(24)01332-1","Cell 2024"),
"grow-ai-virtual-cells":("Grow AI Virtual Cells","Grow AI Virtual Cells: Three Data Pillars and Closed-loop Learning","https://www.nature.com/articles/s41422-025-01101-y","Cell Research 2025"),
"llm-virtual-cell-survey":("LLM Virtual Cell Survey","Large Language Models Meet Virtual Cell: A Survey","https://arxiv.org/abs/2510.07706","bioRxiv 2025"),
"the-virtual-cell":("The Virtual Cell","The Virtual Cell","https://www.nature.com/articles/s41592-025-02951-5","Nature Methods 2025"),
"vcworld":("VCWorld","VCWorld: A Biological World Model for Virtual Cell Simulation","https://arxiv.org/abs/2512.00306","bioRxiv 2025"),
"virtual-cells-predict":("Virtual Cells Predict","Virtual Cells: Predict, Explain, Discover","https://arxiv.org/abs/2505.14613","bioRxiv 2025"),
}

# Filled notes that need specific paper link fixes
FIXES = {
"cell-atlas-fm":"| **论文** | [Universal Cell Embeddings: A Foundation Model for Cell Biology](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1), bioRxiv 2024<br>[A Cell Atlas Foundation Model for Scalable Search of Similar Human Cells](https://www.nature.com/articles/s41586-024-08411-y), Nature 2024 |",
"visual-omics-fm":"| **论文** | [A Visual–Omics Foundation Model to Bridge Histopathology with Spatial Transcriptomics](https://www.nature.com/articles/s41592-025-02707-1), Nature Methods 2025 |",
"scfoundation":"| **论文** | [scFoundation: Large Scale Foundation Model on Single-cell Transcriptomics](https://www.nature.com/articles/s41592-024-02305-7), Nature Methods 2024<br>[xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data](https://arxiv.org/abs/2311.15156), arXiv 2023 (Earlier Version) |",
}

def process(filepath, d):
    with open(filepath, 'r', encoding='utf-8') as f:
        c = f.read()
    
    info = M.get(d)
    if info is None:
        print(f"  ? No mapping for '{d}'")
        return
    
    name, title, url, venue = info
    
    # Check if it's filled (has real paper link) or unfilled (has placeholder)
    is_filled = not bool(re.search(r'\|\s*\*\*论文\*\*\s*\|\s*\[标题\]\(链接\)\s*\|', c))
    
    if is_filled:
        if d in FIXES:
            c = re.sub(r'\| \*\*论文\*\* .* \|', FIXES[d], c, count=1)
            print(f"  FIXED: {d}")
        else:
            print(f"  OK (filled): {d}")
    else:
        c = c.replace("[模型名称] 学习笔记", f"{name} 学习笔记", 1)
        c = c.replace("[模型名称]", name)
        link = f"[{title}]({url})" if url else title
        cell = f"{link}, {venue}" if venue else link
        newline = f"| **论文** | {cell} |"
        c = re.sub(r'\|\s*\*\*论文\*\*\s*\|\s*\[标题\]\(链接\)\s*\|', newline, c, count=1)
        print(f"  FILLED: {d}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(c)

def main():
    for root, dirs, files in os.walk(NOTES):
        if "README.md" not in files: continue
        rel = os.path.relpath(root, NOTES)
        if rel == "." or rel.startswith("_"): continue
        d = os.path.basename(root)
        p = os.path.join(root, "README.md")
        print(f"Processing: {rel}")
        process(p, d)

if __name__ == "__main__":
    main()
