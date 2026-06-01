#!/usr/bin/env python3
"""
sync_notes_config.py

自动发现 notes/ 目录中已撰写的笔记，生成前端配置文件。

工作流程：
  1. 扫描 notes/*/*/README.md，读取 frontmatter
  2. 对没有 frontmatter 的笔记自动添加（推断 status）
  3. 筛选 status=done 的笔记 → 生成 notes.ts 配置

用法:
  python scripts/sync_notes_config.py              # 添加 frontmatter + 生成配置
  python scripts/sync_notes_config.py --dry-run    # 仅预览，不写文件
  python scripts/sync_notes_config.py --frontmatter-only  # 只添加 frontmatter
  python scripts/sync_notes_config.py --validate   # 检查所有笔记的模板残留
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "notes"
CONFIG_DIR = PROJECT_ROOT / "docs-viewer" / "src" / "config"
LOADER_PATH = PROJECT_ROOT / "docs-viewer" / "src" / "lib" / "loaders.ts"
CONFIG_PATH = CONFIG_DIR / "notes.ts"

# frontmatter 正则
FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# 模板残留检测标记
TEMPLATE_MARKERS = [
    "用 ASCII 艺术图展示整体流程",
    "│Encoder  │  ← 组件名",
    "[组件1 名称]",
    "[组件2 名称]",
    "### 3.1 [创新点1]",
    "模型与 X 模型的核心区别是什么",
    "笔记最后更新：YYYY-MM-DD",
    "[相关论文1]",
]

# 分类名称映射（目录名 → 前端显示名）
CATEGORY_NAMES = {
    "fm-classic": "FM + 经典语言模型",
    "fm-spatial": "FM + 空间组学",
    "fm-world-model": "FM + 世界模型",
    "fm-cross-species": "FM + 跨物种/通用嵌入",
    "fm-graph": "FM + 图与网络",
    "fm-llm": "FM + LLM",
    "perturbation": "遗传扰动",
    "benchmarks": "评估与 Benchmark",
    "virtual-cell": "虚拟细胞",
    "pathology": "病理基础模型",
    "surveys": "综述与展望",
}

CATEGORY_ORDER = ["fm-classic", "fm-spatial", "fm-world-model",
                  "fm-cross-species", "fm-graph", "fm-llm",
                  "perturbation", "benchmarks",
                  "virtual-cell", "pathology", "surveys"]


def has_deep_content(text: str) -> bool:
    """判断笔记是否有实质内容"""
    if "一句话总结：这个模型的核心思想和技术路线" in text:
        return False
    if "### 核心思想" in text and "说明 + 代码/公式" not in text:
        return True
    if "## 1. 模型概述" in text and text.count("|") > 20:
        # 有表格且不是模板占位符
        if "如 BERT / GPT" not in text:
            return True
    return False


def has_metadata(text: str) -> bool:
    """判断笔记是否已有论文元数据（标题、年份等被填充过）"""
    # 检查表格中是否有非占位符的内容
    if "**发布日期** | YYYY-MM" not in text and "**发布日期** | 如" not in text:
        if re.search(r"\*\*发布日期\*\* \| \d{4}", text):
            return True
    return False


def infer_status(text: str) -> str:
    """根据内容推断笔记状态"""
    if has_deep_content(text):
        return "done"
    if has_metadata(text):
        return "metadata"
    return "template"


def parse_frontmatter(text: str) -> dict:
    m = FM_PATTERN.match(text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm


def write_frontmatter(text: str, status: str) -> str:
    """添加或更新 frontmatter"""
    fm = parse_frontmatter(text)
    if fm:
        # 更新现有 frontmatter
        old_fm = FM_PATTERN.match(text).group(0)
        fm["status"] = status
        if status == "done" and "filled" not in fm:
            from datetime import date
            fm["filled"] = date.today().isoformat()
        new_fm = "---\n" + "\n".join(f"{k}: {v}" for k, v in fm.items()) + "\n---\n"
        return text.replace(old_fm, new_fm)
    else:
        # 添加新 frontmatter
        from datetime import date
        fm_text = f"---\nstatus: {status}\n"
        if status == "done":
            fm_text += f"filled: {date.today().isoformat()}\n"
        fm_text += "---\n\n"
        return fm_text + text


def validate_all_notes(notes: list[dict]):
    """检查所有已完成笔记的模板残留"""
    print("检查模板残留...")
    has_error = False
    checked = 0
    for n in notes:
        if n["status"] != "done":
            continue
        checked += 1
        text = n["text"]
        found = [m for m in TEMPLATE_MARKERS if m in text]
        if found:
            print(f"  ❌ {n['category']}/{n['id']}: {found}")
            has_error = True
    if not has_error:
        print(f"  ✅ {checked} 篇已完成笔记全部通过")
    else:
        print(f"  ⚠️  请修复后重新运行")
    return has_error


def scan_notes() -> list[dict]:
    """扫描所有笔记，返回信息列表"""
    notes = []
    for section in sorted(NOTES_DIR.iterdir()):
        if not section.is_dir() or section.name.startswith("."):
            continue
        for note_dir in sorted(section.iterdir()):
            readme = note_dir / "README.md"
            if not readme.exists():
                continue
            text = readme.read_text(encoding="utf-8")
            fm = parse_frontmatter(text)
            status = fm.get("status") or infer_status(text)
            title = _guess_title(note_dir.name, text)
            notes.append({
                "id": note_dir.name,
                "title": title,
                "category": section.name,
                "path": f"{section.name}/{note_dir.name}/README.md",
                "status": status,
                "file": readme,
                "text": text,
            })
    return notes


def _guess_title(dir_name: str, text: str) -> str:
    """从内容或目录名推断标题"""
    # 优先使用硬编码映射表（确保标题规范性）
    name_map = {
        "cell-atlas-fm": "UCE / Cell Atlas FM",
        "scgpt-spatial": "scGPT-spatial",
        "scprint-2": "scPRINT-2",
        "gpt4-cell-annotation": "GPT-4 Cell Annotation",
        "the-virtual-cell": "The Virtual Cell",
        "virtual-cell-challenge": "Virtual Cell Challenge",
        "cell-plm": "CellPLM",
        "sc-mae": "scMAE",
        "gene-repr-st": "Gene Representation for ST",
        "multi-cellular-repr": "Multi-cellular Representations",
        "metadata-as-language": "Metadata as Language",
        "sc-expression-lm": "scRNA-seq Expression LM",
        "divide-conquer-ssl": "Divide-and-Conquer SSL",
        "dna-to-expression": "DNA to Expression",
        "rnaseq-coverage-dna": "RNA-seq Coverage from DNA",
        "privacy-federated": "Privacy-preserving Federated",
        "scaling-dense": "Scaling Dense Representations",
        "spatial-proteomics-fm": "Spatial Proteomics FM",
        "visual-omics-fm": "Visual-Omics FM",
        "transcription-foundation": "Transcription Foundation Model",
        "transcriptome-proteome": "Transcriptome to Proteome",
        "unified-perturbation": "Unified Perturbation Model",
        "multimodal-perturbation": "Multimodal Perturbation FM",
        "cell-niche-graph": "Cell Niche Graph",
        "cell-ontology-fm": "Cell Ontology FM",
        "chatgpt-embedding-sc": "ChatGPT Embedding for SC",
        "chat-based-sc-exploration": "Chat-based SC Exploration",
        "joint-embed-transcript-text": "Joint Embed Transcript-Text",
        "language-enhanced-repr": "Language-enhanced Representation",
        "llm-complement-scfm": "LLM Complement scFM",
        "llm-consensus-annotation": "LLM Consensus Annotation",
        "scaling-llm-sc": "Scaling LLM for SC",
        "llm-gene-set-function": "LLM Gene Set Function",
        "llm-virtual-cell-survey": "LLM + Virtual Cell Survey",
        "open-problems-sc-analysis": "Open Problems SC Analysis",
        "perturbation-benchmarking": "Perturbation Benchmarking",
        "perturbation-baselines": "Perturbation Baselines",
        "ssl-effective-use": "SSL Effective Use",
        "systematic-perturbation-compare": "Systematic Perturbation Compare",
        "batch-effects-barrier": "Batch Effects Barrier",
        "biology-driven-insights": "Biology-driven Insights",
        "cancer-outcomes-evaluation": "Cancer Outcomes Evaluation",
        "cell-type-classification-eval": "Cell Type Classification Eval",
        "deep-dive-scfms": "Deep Dive into scFMs",
        "deeper-evaluation-scfms": "Deeper Evaluation of scFMs",
        "gene-embeddings-benchmark": "Gene Embeddings Benchmark",
        "imbalanced-cell-annotation": "Imbalanced Cell Annotation",
        "mode-collapse-perturbation": "Mode Collapse Perturbation",
        "multimodal-integration-benchmark": "Multimodal Integration Benchmark",
        "pretraining-size-diversity": "Pretraining Size & Diversity",
        "transferability-sc-to-st": "Transferability SC to ST",
        "unified-benchmarking-framework": "Unified Benchmarking Framework",
        "zero-shot-limitations": "Zero-shot Limitations",
        "metric-mirages": "Metric Mirages",
        "sparse-autoencoders-scfm": "Sparse Autoencoders for scFMs",
        "transcriptional-grammar": "Transcriptional Grammar",
        "in-silico-discovery": "In Silico Discovery",
        "perturbation-linear-baselines": "Perturbation Linear Baselines",
        "pca-still-rules": "PCA Still Rules",
        "benchmark-cell-model-perturbation": "Benchmark Cell Model Perturbation",
        "ai-virtual-cell-preclinical": "AI Virtual Cell Preclinical",
        "build-virtual-cell-ai": "Build Virtual Cell with AI",
        "grow-ai-virtual-cells": "Grow AI Virtual Cells",
        "virtual-cells-predict": "Virtual Cells: Predict, Explain, Discover",
        "general-purpose-pathology": "General Purpose Pathology FM",
        "visual-language-pathology": "Visual-Language Pathology FM",
        "visual-language-pathology-twitter": "Visual-Language Pathology (Twitter)",
        "whole-slide-fm": "Whole-Slide FM",
        "biomedical-seg-det-rec": "Biomedical Seg/Det/Rec",
        "geneformer": "Geneformer",
        "scgpt": "scGPT",
        "scfoundation": "scFoundation",
        "scbert": "scBERT",
        "scpoli": "scPoli",
        "scprint": "scPRINT",
        "nicheformer": "Nicheformer",
        "novae": "Novae",
        "saturn": "SATURN",
        "genecompass": "GeneCompass",
        "epiagent": "EpiAgent",
        "xtrimogene": "xTrimoGene",
        "sclong": "scLong",
        "cellfm": "CellFM",
        "scpeft": "scPEFT",
        "scnet": "scNET",
        "scelmo": "scELMo",
        "genejepa": "GeneJEPA",
        "sclinguist": "scLinguist",
        "cellpolaris": "CellPolaris",
        "scconcept": "scConcept",
        "musegnn": "MuSe-GNN",
        "tabulam": "Tabula",
        "scprotein": "scPROTEIN",
        "cell2sentence": "Cell2Sentence",
        "cellama": "CELLama",
        "scchat": "scChat",
        "cassia": "CASSIA",
        "cinema-ot": "CINEMA-OT",
        "gears": "GEARS",
        "cpa": "CPA",
        "pertadapt": "PertAdapt",
        "tahoe-100m": "Tahoe-100M / Tahoe-x1",
        "scmulan": "scMulan",
    }
    if dir_name in name_map:
        return name_map[dir_name]
    # 从 h1 获取
    m = re.search(r"^# (.+?)学习笔记", text, re.MULTILINE)
    if m:
        t = m.group(1).strip()
        if t != "[模型名称]":
            return t
    # 从目录名转换
    return dir_name.replace("-", " ").title()


def generate_notes_ts(notes: list[dict]) -> str:
    """生成 notes.ts 内容"""
    lines = []
    lines.append('// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成')
    lines.append('// 手动修改将被覆盖！请修改笔记的 frontmatter 后重新运行。')
    lines.append('')
    lines.append('export interface NoteMeta {')
    lines.append('  id: string')
    lines.append('  title: string')
    lines.append('  category: string')
    lines.append('  path: string')
    lines.append('}')
    lines.append('')
    lines.append('export const categories = [')
    for cat in CATEGORY_ORDER:
        lines.append(f"  '{CATEGORY_NAMES[cat]}',")
    lines.append('] as const')
    lines.append('')
    lines.append('export const noteMetas: NoteMeta[] = [')

    # 🔹 首先输出 overview
    lines.append("  { id: 'overview', title: '总览', category: '概览', path: 'README.md' },")

    for cat in CATEGORY_ORDER:
        cat_notes = [n for n in notes if n["category"] == cat and n["status"] == "done"]
        if not cat_notes:
            continue
        cat_name = CATEGORY_NAMES[cat]
        lines.append(f"")
        lines.append(f"  // --- {cat_name} ---")
        for n in cat_notes:
            lines.append(f"  {{ id: '{n['id']}', title: '{n['title']}', category: '{cat_name}', path: '{n['path']}' }},")

    lines.append(']')
    lines.append('')
    return "\n".join(lines)


def generate_loaders_ts(notes: list[dict]) -> str:
    """生成 loaders.ts 内容"""
    lines = []
    lines.append('// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成')
    lines.append('// 手动修改将被覆盖！')
    lines.append('')
    lines.append("import { noteMetas } from '../config/notes'")
    lines.append('')
    lines.append('export type NoteLoader = () => Promise<string>')
    lines.append('')
    lines.append('function fetchNote(path: string): NoteLoader {')
    lines.append("  return () => fetch(`${import.meta.env.BASE_URL}notes/${path}`).then((r) => r.text())")
    lines.append('}')
    lines.append('')
    lines.append('export const noteLoaders: Record<string, NoteLoader> = {}')
    lines.append('for (const meta of noteMetas) {')
    lines.append('  noteLoaders[meta.id] = fetchNote(meta.path)')
    lines.append('}')
    lines.append('')
    return "\n".join(lines)


def add_frontmatter_to_all(notes: list[dict], dry_run: bool = False):
    """为所有没有 frontmatter 的笔记添加"""
    count = 0
    for n in notes:
        if not parse_frontmatter(n["text"]):
            status = infer_status(n["text"])
            new_text = write_frontmatter(n["text"], status)
            if dry_run:
                print(f"  [DRY] {n['category']}/{n['id']} → status={status}")
            else:
                n["file"].write_text(new_text, encoding="utf-8")
                print(f"  ✅ {n['category']}/{n['id']} → status={status}")
            n["text"] = new_text
            n["status"] = status
            count += 1
    return count


def main():
    dry_run = "--dry-run" in sys.argv
    fm_only = "--frontmatter-only" in sys.argv

    print("扫描 notes/...")
    notes = scan_notes()

    # 统计
    done = [n for n in notes if n["status"] == "done"]
    meta = [n for n in notes if n["status"] == "metadata"]
    tmpl = [n for n in notes if n["status"] == "template"]
    print(f"  总计: {len(notes)} | done: {len(done)} | metadata: {len(meta)} | template: {len(tmpl)}\n")

    # 添加 frontmatter
    print("添加/更新 frontmatter:")
    added = add_frontmatter_to_all(notes, dry_run)
    print(f"  {added} 个笔记已更新\n")

    if fm_only:
        return

    if "--validate" in sys.argv:
        validate_all_notes(notes)
        return

    # 重新扫描（frontmatter 已更新）
    if added > 0 and not dry_run:
        notes = scan_notes()
        done = [n for n in notes if n["status"] == "done"]

    # 生成配置
    if dry_run:
        print(f"[DRY] 将生成 {len(done)} 条笔记配置到 notes.ts")
        for n in done:
            print(f"  {n['category']}/{n['id']} ({n['title']})")
        return

    print(f"生成前端配置 ({len(done)} 条笔记)...")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config_ts = generate_notes_ts(notes)
    CONFIG_PATH.write_text(config_ts, encoding="utf-8")
    print(f"  ✅ {CONFIG_PATH}")

    loaders_ts = generate_loaders_ts(notes)
    LOADER_PATH.write_text(loaders_ts, encoding="utf-8")
    print(f"  ✅ {LOADER_PATH}")

    print("\n完成！请运行 npm run build 验证构建。")


if __name__ == "__main__":
    main()
