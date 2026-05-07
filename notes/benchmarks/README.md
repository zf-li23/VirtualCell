# Benchmarks — 单细胞基础模型评估

> Foundation-Model-Evaluation-For-Single-cell，共 35 篇评估与 Benchmark 论文。

---

## 目录结构

```
benchmarks/
├── README.md
├── systematic-perturbation-compare/   ← #1 Systematic Comparison of Perturbation Response
├── unified-benchmarking-framework/    ← #2 Unified benchmarking framework
├── open-problems-sc-analysis/         ← #3 Defining open problems (Nat Biotech 2025)
├── perturbation-benchmarking/         ← #4 Benchmarking perturbation response (Nat Methods)
├── batch-effects-barrier/             ← #5 Batch Effects Barrier
├── sccluben/                          ← #6 scCluBench
├── perturbench/                       ← #7 PerturBench
├── virtual-cell-challenge/            ← #8 Virtual Cell Challenge
├── heimdall/                          ← #9 HEIMDALL tokenization
├── cancer-outcomes-evaluation/        ← #10 scFMs for Cancer Outcomes
├── anndictionary/                     ← #11 AnnDictionary
├── sparse-autoencoders-scfm/          ← #12 Sparse Autoencoders in scFMs
├── transferability-sc-to-st/          ← #13 SSL from sc to ST
├── perturbation-baselines/            ← #14 Perturbation models vs baselines
├── multimodal-integration-benchmark/  ← #15 Multitask benchmarking multimodal
├── bmfm-rna/                          ← #16 BMFM-RNA
├── biology-driven-insights/           ← #17 Biology-driven insights
├── gene-embeddings-benchmark/         ← #19 Benchmarking gene embeddings
├── mode-collapse-perturbation/        ← #20 Mode Collapse in Perturbation
├── zero-shot-limitations/             ← #21 Zero-shot evaluation limitations
├── scdrugmap/                         ← #22 scDrugMap
├── cell-type-classification-eval/     ← #23 scFMs on Cell-Type Classification
├── ssl-effective-use/                 ← #24 SSL in single-cell genomics
├── biollm/                            ← #25 BioLLM framework
├── pretraining-size-diversity/        ← #26 Pre-training dataset size & diversity
├── deeper-evaluation-scfms/           ← #27 Deeper evaluation
├── gpt4-cell-annotation/              ← #28 GPT-4 cell type annotation
├── metric-mirages/                    ← #29 Metric Mirages
├── transcriptional-grammar/           ← #30 Transcriptional grammar (reusability)
├── deep-dive-scfms/                   ← #31 Deep Dive into scFMs
├── sceval/                            ← #32 scEval
├── imbalanced-cell-annotation/        ← #33 Imbalanced data for cell annotation
├── llm-gene-set-function/             ← #34 LLMs for gene set function
└── bend/                              ← #35 BEND: Benchmarking DNA LMs
```

## 设计思路

Benchmark 论文不写详细"模型架构笔记"，而是以 **评测报告摘要** 形式记录：

1. **评测了哪些模型？**
2. **用了哪些数据集和任务？**
3. **关键结论是什么？**
4. **对你自己的项目有什么启示？**

## 状态

| 状态 | 计数 |
|------|------|
| ⬜ 待创建 | 35 |
