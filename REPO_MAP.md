# 论文 ↔ GitHub 仓库映射表

> 📌 本文记录 `notes/` 下每篇论文对应的官方代码仓库地址。
> 用途：批量 `git clone` 下载代码、快速定位源码。
>
> 最后更新：2026-05-20
>
> 图例：
> - ✅ 仓库已验证可用
> - ⚠️ 仓库可能需进一步确认
> - ❌ 未找到公开仓库
> - 📝 综述/观点文章，通常无专属仓库

---

## 目录

- [单细胞基础模型 (FM) — `notes/fm/`](#单细胞基础模型-fm--notesfm)
- [FM + LLM — `notes/fm-llm/`](#fm--llm--notesfm-llm)
- [遗传扰动 — `notes/perturbation/`](#遗传扰动--notesperturbation)
- [评估与 Benchmark — `notes/benchmarks/`](#评估与-benchmark--notesbenchmarks)
- [虚拟细胞 — `notes/virtual-cell/`](#虚拟细胞--notesvirtual-cell)
- [病理基础模型 — `notes/pathology/`](#病理基础模型--notestpathology)
- [综述与展望 — `notes/surveys/`](#综述与展望--notestsurveys)

---

## 单细胞基础模型 (FM) — `notes/fm/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `atacformer` | ❌ 未找到 | ❌ |
| `bidir-mamba` | ❌ 未找到 | ❌ |
| `cancerfoundation` | ❌ 未找到 | ❌ |
| `captain` | ❌ 未找到 | ❌ |
| `cell-atlas-fm` | [https://github.com/snap-stanford/UCE](https://github.com/snap-stanford/UCE) | ✅ |
| `cell-graphcompass` | ❌ 未找到 | ❌ |
| `cell-niche-graph` | ❌ 未找到 | ❌ |
| `cell-ontology-fm` | ❌ 未找到 | ❌ |
| `cell-plm` | ❌ 未找到 | ❌ |
| `cellfm` | ❌ 未找到 | ❌ |
| `cellpolaris` | ❌ 未找到 | ❌ |
| `chromfound` | ❌ 未找到 | ❌ |
| `divide-conquer-ssl` | ❌ | ❌ |
| `dna-to-expression` | ❌ 未找到 | ❌ |
| `epiagent` | ❌ 未找到 | ❌ |
| `epifoundation` | ❌ 未找到 | ❌ |
| `gene-repr-st` | ❌ | ❌ |
| `genecompass` | ❌ 未找到 | ❌ |
| `geneformer` | [https://hf-mirror.com/ctheodoris/Geneformer](https://hf-mirror.com/ctheodoris/Geneformer)（HuggingFace） | ✅ |
| `genejepa` | ❌ 未找到 | ❌ |
| `generative-pretraining` | ❌ | ❌ |
| `langcell` | [https://github.com/PharMolix/LangCell](https://github.com/PharMolix/LangCell) | ✅ |
| `latent-diffusion-sc` | ❌ 未找到 | ❌ |
| `metadata-as-language` | ❌ 博客文章 | ❌ |
| `multi-cellular-repr` | ❌ | ❌ |
| `multimodal-perturbation` | ❌ 未找到 | ❌ |
| `musegnn` | ❌ 未找到 | ❌ |
| `nicheformer` | [https://github.com/theislab/nicheformer](https://github.com/theislab/nicheformer) | ✅ |
| `novae` | [https://github.com/prism-oncology/novae](https://github.com/prism-oncology/novae) | ✅ |
| `omnicell` | ❌ | ❌ |
| `privacy-federated` | ❌ | ❌ |
| `pulsar` | ❌ | ❌ |
| `rnaseq-coverage-dna` | ❌ | ❌ |
| `saturn` | [https://github.com/snap-stanford/saturn](https://github.com/snap-stanford/saturn) | ✅ |
| `sc-expression-lm` | ❌ | ❌ |
| `sc-mae` | ❌ 未找到 | ❌ |
| `scaling-dense` | ❌ | ❌ |
| `scaraf` | ❌ 无法确定对应论文 | ❌ |
| `scarf` | ❌ 未找到 | ❌ |
| `scbert` | [https://github.com/TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT) | ✅ |
| `scclip` | ❌ 未找到 | ❌ |
| `scconcept` | ❌ 未找到 | ❌ |
| `scelmo` | ❌ 未找到 | ❌ |
| `scfoundation` | [https://github.com/biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | ✅ |
| `scgpt` | [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | ✅ |
| `scgpt-spatial` | [https://github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)（同仓库） | ✅ |
| `schiena` (scHyena) | ❌ 未找到 | ❌ |
| `sclinguist` | ❌ 未找到 | ❌ |
| `sclong` | ❌ 未找到 | ❌ |
| `scmformer` | ❌ | ❌ |
| `scmulan` | ❌ 未找到 | ❌ |
| `scnet` | ❌ 未找到 | ❌ |
| `scpeft` | ❌ 未找到 | ❌ |
| `scpoli` | [https://github.com/theislab/scPoli](https://github.com/theislab/scPoli) | ✅ |
| `scpretrain` | ❌ 未找到 | ❌ |
| `scprint` | [https://github.com/jkobject/scPRINT](https://github.com/jkobject/scPRINT) | ✅ |
| `scprint-2` | [https://github.com/jkobject/scPRINT](https://github.com/jkobject/scPRINT)（同仓库） | ✅ |
| `scprotein` | ❌ | ❌ |
| `spatial-proteomics-fm` | ❌ | ❌ |
| `spatranslator` | ❌ | ❌ |
| `stack` | ❌ | ❌ |
| `stofm` | ❌ | ❌ |
| `stpath` | ❌ | ❌ |
| `switch` | ❌ 未找到 | ❌ |
| `tabulam` (Tabula) | ❌ | ❌ |
| `transcriptformer` | ❌ | ❌ |
| `transcription-foundation` | ❌ 未找到 | ❌ |
| `transcriptome-proteome` | ❌ | ❌ |
| `unified-perturbation` | ❌ | ❌ |
| `visual-omics-fm` | ❌ 未找到 | ❌ |
| `xtrimogene` | ❌ 未找到 | ❌ |

---

## FM + LLM — `notes/fm-llm/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `cassia` | ❌ 未找到 | ❌ |
| `cell2sentence` | ❌ 未找到 | ❌ |
| `cellama` | ❌ 未找到 | ❌ |
| `cellforge` | ❌ 未找到 | ❌ |
| `cellhermes` | ❌ | ❌ |
| `celltok` | ❌ | ❌ |
| `chat-based-sc-exploration` | ❌ | ❌ |
| `chatgpt-embedding-sc` | ❌ (基于 OpenAI API) | ❌ |
| `gpt4-cell-annotation` | ❌ (基于 GPT-4 API) | ❌ |
| `joint-embed-transcript-text` | ❌ | ❌ |
| `language-enhanced-repr` | ❌ | ❌ |
| `llm-complement-scfm` | ❌ | ❌ |
| `llm-consensus-annotation` | ❌ | ❌ |
| `okr-cell` | ❌ | ❌ |
| `scaling-llm-sc` | ❌ | ❌ |
| `scchat` | ❌ 未找到 | ❌ |
| `scilama` | ❌ 未找到 | ❌ |
| `scouter` | ❌ 未找到 | ❌ |
| `screader` | ❌ 未找到 | ❌ |
| `spemo` | ❌ | ❌ |
| `teddy` | ❌ | ❌ |
| `tissuenarrator` | ❌ | ❌ |

---

## 遗传扰动 — `notes/perturbation/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `benchmark-cell-model-perturbation` | ❌ | ❌ |
| `cinema-ot` | ❌ 未找到 | ❌ |
| `diffusion-debias-perturbation` | ❌ | ❌ |
| `in-silico-discovery` | ❌ 未找到 | ❌ |
| `pca-still-rules` | ❌ | ❌ |
| `pertadapt` | ❌ 未找到 | ❌ |
| `perteval-scfm` | ❌ 未找到 | ❌ |
| `perturbation-linear-baselines` | ❌ | ❌ |
| `scfme` | ❌ | ❌ |
| `scgenept` | ❌ 未找到 | ❌ |
| `sclambda` | ❌ 未找到 | ❌ |
| `spatialprop` | ❌ | ❌ |
| `state` | ❌ 未找到 | ❌ |
| `systema` | ❌ 未找到 | ❌ |
| `tahoe-100m` | ❌ 未找到 | ❌ |
| `tahoe-x1` | ❌ 未找到（与 tahoe-100m 同仓库） | ❌ |

---

## 评估与 Benchmark — `notes/benchmarks/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `anndictionary` | ❌ | ❌ |
| `batch-effects-barrier` | ❌ | ❌ |
| `bend` | ❌ BEND 仓库与单细胞无关 | ❌ |
| `biollm` | ❌ 未找到 | ❌ |
| `biology-driven-insights` | ❌ | ❌ |
| `bmfm-rna` | ❌ 未找到 | ❌ |
| `cancer-outcomes-evaluation` | ❌ | ❌ |
| `cell-type-classification-eval` | ❌ | ❌ |
| `deep-dive-scfms` | ❌ | ❌ |
| `deeper-evaluation-scfms` | ❌ | ❌ |
| `gene-embeddings-benchmark` | ❌ | ❌ |
| `gpt4-cell-annotation` | ❌ (使用 GPT-4 API) | ❌ |
| `heimdall` | ❌ 未找到 | ❌ |
| `imbalanced-cell-annotation` | ❌ | ❌ |
| `llm-gene-set-function` | ❌ | ❌ |
| `metric-mirages` | ❌ | ❌ |
| `mode-collapse-perturbation` | ❌ | ❌ |
| `multimodal-integration-benchmark` | ❌ | ❌ |
| `open-problems-sc-analysis` | ❌ 未找到 | ❌ |
| `perturbation-baselines` | ❌ | ❌ |
| `perturbation-benchmarking` | ❌ | ❌ |
| `perturbench` | ❌ 未找到 | ❌ |
| `pretraining-size-diversity` | ❌ | ❌ |
| `sccluben` | ❌ 未找到 | ❌ |
| `scdrugmap` | ❌ 未找到 | ❌ |
| `sceval` | ❌ 未找到 | ❌ |
| `sparse-autoencoders-scfm` | ❌ | ❌ |
| `ssl-effective-use` | ❌ | ❌ |
| `systematic-perturbation-compare` | ❌ | ❌ |
| `transcriptional-grammar` | ❌ | ❌ |
| `transferability-sc-to-st` | ❌ | ❌ |
| `unified-benchmarking-framework` | ❌ | ❌ |
| `virtual-cell-challenge` | ❌ 未找到 | ❌ |
| `zero-shot-limitations` | ❌ | ❌ |

---

## 虚拟细胞 — `notes/virtual-cell/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `ai-virtual-cell-preclinical` | 📝 综述 | ❌ |
| `build-virtual-cell-ai` | 📝 观点文章 | ❌ |
| `cellforge` | ❌ 未找到 | ❌ |
| `grow-ai-virtual-cells` | 📝 观点文章 | ❌ |
| `llm-virtual-cell-survey` | 📝 综述 | ❌ |
| `the-virtual-cell` | 📝 评论文章 | ❌ |
| `vcworld` | ❌ 可能是闭源 | ❌ |
| `virtual-cell-challenge` | ❌ 未找到 | ❌ |
| `virtual-cells-predict` | ❌ | ❌ |

---

## 病理基础模型 — `notes/pathology/`

| 目录名 | 代码仓库 | 状态 |
|--------|---------|------|
| `biomedical-seg-det-rec` | ❌ 未找到 | ❌ |
| `general-purpose-pathology` | ❌ 未找到 | ❌ |
| `visual-language-pathology` | [https://github.com/mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH) | ⚠️ |
| `visual-language-pathology-twitter` | ❌ | ❌ |
| `whole-slide-fm` | ❌ 未找到 | ❌ |

---

## 综述与展望 — `notes/surveys/`

> 📝 综述/观点/展望类文章通常没有代码仓库，标为 `N/A`

| 目录名 | 代码仓库 | 说明 |
|--------|---------|------|
| `aivo` | N/A | 综述 |
| `barriers-sc-llm` | N/A | 评论 |
| `cell-atlases-opportunities` | N/A | 综述 |
| `cross-species-transfer` | N/A | Comment in Nat Methods |
| `decoding-cell-fate` | N/A | 综述 |
| `fm-bioinformatics` | N/A | 综述 |
| `fm-spatial-transcriptomic` | N/A | 观点 |
| `general-purpose-cellular-models` | N/A | 综述 |
| `harnessing-fm-omics` | N/A | 综述 |
| `hca-to-unified-fm` | N/A | 观点 |
| `interpretation-perturbation-sc` | N/A | 综述 |
| `llm-drug-discovery` | N/A | 综述 |
| `mini-review-perturbation-modalities` | N/A | 综述 |
| `multimodal-fm-cell-biology` | N/A | 综述 (Nature) |
| `multimodal-transformer-genomics` | N/A | 综述 (Nat Methods) |
| `perturbation-cell-tissue-atlas` | N/A | 观点 (Cell) |
| `progress-opportunities-fm-bioinfo` | N/A | 综述 |
| `reference-mapping-future` | N/A | 观点 (Cell) |
| `scfm-into-cell-biology` | N/A | 综述 |
| `spatial-omics-forefront` | N/A | 综述 |
| `survey-fm-sc-biology` | N/A | 综述 (ACL) |
| `tokenization-review` | N/A | 综述 |
| `transformers-genome-lm` | N/A | 综述 (Nat Mach Intell) |
| `transformers-sc-omics-review` | N/A | 综述 (Nat Methods) |

---

## 批量下载脚本

### 已下载（`repos/` 目录中，已验证）

| 仓库 | 对应笔记 | 实际 URL |
|------|---------|---------|
| `Geneformer/` | `notes/fm/geneformer/` | HuggingFace (hf-mirror.com) |
| `scGPT/` | `notes/fm/scgpt/`、`notes/fm/scgpt-spatial/` | `bowang-lab/scGPT` |
| `scFoundation/` | `notes/fm/scfoundation/` | `biomap-research/scFoundation` |
| `UCE/` | `notes/fm/cell-atlas-fm/` | `snap-stanford/UCE` |
| `nicheformer/` | `notes/fm/nicheformer/` | `theislab/nicheformer` |
| `novae/` | `notes/fm/novae/` | `prism-oncology/novae` |
| `saturn/` | `notes/fm/saturn/` | `snap-stanford/saturn` |
| `scBERT/` | `notes/fm/scbert/` | `TencentAILabHealthcare/scBERT` |
| `scPoli/` | `notes/fm/scpoli/` | `theislab/scPoli` |
| `scPRINT/` | `notes/fm/scprint/`、`notes/fm/scprint-2/` | `jkobject/scPRINT` |
| `LangCell/` | `notes/fm/langcell/` | `PharMolix/LangCell` |
| `GraphST/` | 空间组学工具 | `JinmiaoChenLab/GraphST` |
| `SpaceFlow/` | 空间组学工具 | `hongleir/SpaceFlow` |
| `SpaGCN/` | 空间组学工具 | `jianhuupenn/SpaGCN` |
| `SPADE/` | 空间组学工具 | `uclabair/SPADE` |
| `OpenBioMed/` | 生物医药通用工具 | `BioFM/OpenBioMed` |

### 批量下载未下载的仓库

```bash
# 从 REPO_MAP.md 提取所有 GitHub URL 并批量 clone
grep -oP 'https://github\.com/[^)]+' REPO_MAP.md | sort -u | while read url; do
  repo_name=$(echo "$url" | sed 's|https://github.com/||' | tr '/' '-')
  if [ ! -d "repos/$repo_name" ]; then
    git clone --depth 1 "$url" "repos/$repo_name"
  fi
done
```

> ⚠️ 注意：以上仓库链接大部分来自文献中的官方声明或 GitHub 搜索，部分标记为 ⚠️ 的链接需要手动验证。
> 如果你在使用中发现链接失效或找到正确链接，欢迎更新本文。
