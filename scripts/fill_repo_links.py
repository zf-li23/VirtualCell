#!/usr/bin/env python3
"""
fill_repo_links.py

功能：
  1. 将 repos/ 中已克隆仓库的 URL 填入笔记的"代码"行
  2. 将论文元数据（标题、年份、期刊）填入笔记的概述表格
  3. 尝试克隆尚未下载的仓库

用法:
  python scripts/fill_repo_links.py                  # 填充 + 下载
  python scripts/fill_repo_links.py --fill-only      # 只填充
  python scripts/fill_repo_links.py --clone-only     # 只下载
  python scripts/fill_repo_links.py --metadata-only  # 只填元数据
"""

import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOS_DIR = PROJECT_ROOT / "repos"
NOTES_DIR = PROJECT_ROOT / "notes"

# ========== 笔记目录名 → (owner/repo, status, title, year, venue) ==========
NOTE_META = {
    # === notes/fm/ ===
    "geneformer": ("ctheodoris/Geneformer", "⚠️", "Geneformer", "2023", "Nature"),
    "scgpt": ("bowang-lab/scGPT", "✅", "scGPT", "2024", "Nature Methods"),
    "scfoundation": ("biomap-research/scFoundation", "✅", "scFoundation", "2024", "Nature Methods"),
    "scbert": ("TencentAILabHealthcare/scBERT", "✅", "scBERT", "2022", "Nature Machine Intelligence"),
    "scpoli": ("theislab/scPoli", "✅", "scPoli", "2023", "Nature Methods"),
    "scprint": ("jkobject/scPRINT", "✅", "scPRINT", "2025", "Nature Communications"),
    "nicheformer": ("theislab/nicheformer", "✅", "Nicheformer", "2025", "Nature Methods"),
    "novae": ("prism-oncology/novae", "✅", "Novae", "2025", "Nature Methods"),
    "cell-atlas-fm": ("snap-stanford/UCE", "✅", "UCE / Cell Atlas FM", "2024", "Nature"),
    "saturn": ("snap-stanford/saturn", "✅", "SATURN", "2024", "Nature Methods"),
    "langcell": ("PharMolix/LangCell", "✅", "LangCell", "2024", "ICML"),
    "genecompass": (None, "⚠️", "GeneCompass", "2024", "Cell Research"),
    "epiagent": (None, "⚠️", "EpiAgent", "2025", "Nature Methods"),
    "xtrimogene": (None, "⚠️", "xTrimoGene", "2023", "NeurIPS"),
    "sclong": (None, "⚠️", "scLong", "2024", "bioRxiv"),
    "cellfm": (None, "⚠️", "CellFM", "2025", "Nature Communications"),
    "scpeft": (None, "⚠️", "scPEFT", "2025", "Nature Machine Intelligence"),
    "scmulan": (None, "⚠️", "scMulan", "2024", "RECOMB"),
    "scnet": (None, "⚠️", "scNET", "2025", "Nature Methods"),
    "scelmo": (None, "⚠️", "scELMo", "2023", "bioRxiv"),
    "scarf": (None, "⚠️", "SCARF", "2025", "bioRxiv"),
    "genejepa": (None, "⚠️", "GeneJepa", "2025", "bioRxiv"),
    "sclinguist": (None, "⚠️", "scLinguist", "2025", "bioRxiv"),
    "cell-plm": (None, "⚠️", "CellPLM", "2024", "ICLR"),
    "cellpolaris": (None, "⚠️", "CellPolaris", "2023", "bioRxiv"),
    "sc-mae": (None, "⚠️", "scMAE", "2023", "NeurIPS Workshop"),
    "scclip": (None, "⚠️", "scCLIP", "2023", "NeurIPS Workshop"),
    "scconcept": (None, "⚠️", "scConcept", "2025", "bioRxiv"),
    "scpretrain": (None, "⚠️", "scPretrain", "2022", "Bioinformatics"),
    "musegnn": (None, "⚠️", "MuSe-GNN", "2023", "NeurIPS"),
    "switch": (None, "⚠️", "SWITCH", "2025", "Nature Computational Science"),
    "cell-graphcompass": (None, "⚠️", "Cell-GraphCompass", "2024", "National Science Review"),
    "cell-ontology-fm": (None, "⚠️", "Cell Ontology FM", "2024", "NeurIPS"),
    "visual-omics-fm": (None, "⚠️", "Visual-Omics FM", "2025", "Nature Methods"),
    "scgpt-spatial": ("bowang-lab/scGPT", "✅", "scGPT-spatial", "2025", "bioRxiv"),
    "scprint-2": ("jkobject/scPRINT", "✅", "scPRINT-2", "2025", "bioRxiv"),

    # === notes/fm-llm/ ===
    "cassia": (None, "⚠️", "CASSIA", "2025", "Nature Communications"),
    "cell2sentence": (None, "⚠️", "Cell2Sentence", "2024", "ICML"),
    "cellama": (None, "⚠️", "CELLama", "2024", "Advanced Science"),
    "scchat": (None, "⚠️", "scChat", "2024", "AIChE Journal"),
    "gpt4-cell-annotation": (None, "⚠️", "GPT-4 Cell Annotation", "2024", "Nature Methods"),

    # === notes/perturbation/ ===
    "cinema-ot": (None, "⚠️", "CINEMA-OT", "2023", "Nature Methods"),
    "pertadapt": (None, "⚠️", "PertAdapt", "2025", "bioRxiv"),
    "state": (None, "⚠️", "STATE", "2025", "bioRxiv"),
    "tahoe-100m": (None, "⚠️", "Tahoe-100M", "2025", "bioRxiv"),
    "tahoe-x1": (None, "⚠️", "Tahoe-x1", "2025", "bioRxiv"),

    # === notes/benchmarks/ ===
    "virtual-cell-challenge": (None, "⚠️", "Virtual Cell Challenge", "2025", "Cell"),
    "zero-shot-limitations": (None, "⚠️", "Zero-shot Limitations of scFMs", "2025", "Genome Biology"),
    "metric-mirages": (None, "⚠️", "Metric Mirages", "2024", "bioRxiv"),

    # === notes/virtual-cell/ ===
    "the-virtual-cell": (None, "⚠️", "The Virtual Cell", "2025", "Nature Methods"),
}


def get_cloned_repos() -> dict[str, str]:
    repos = {}
    if not REPOS_DIR.exists():
        return repos
    for d in REPOS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            r = subprocess.run(
                ["git", "-C", str(d), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                continue
            url = r.stdout.strip()
            if url.startswith("git@"):
                url = url.replace(":", "/").replace("git@", "https://").removesuffix(".git")
            repos[d.name] = url
        except Exception:
            continue
    return repos


def find_note_path(note_dir: str) -> Path | None:
    for p in ["fm", "fm-llm", "perturbation", "benchmarks", "virtual-cell", "pathology", "surveys"]:
        path = NOTES_DIR / p / note_dir / "README.md"
        if path.exists():
            return path
    return None


def fill_table_row(content: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(\| \*\*{re.escape(key)}\*\* \|).*(\|)$", re.MULTILINE)
    if pattern.search(content):
        return pattern.sub(rf"\1 {value} \2", content)
    lines = content.split("\n")
    table_last = -1
    in_table = False
    for i, line in enumerate(lines):
        if re.match(r"^\| .+ \| .+ \|$", line):
            table_last = i
            in_table = True
        else:
            if in_table and not line.strip():
                break
            in_table = False
    if table_last >= 0:
        lines.insert(table_last + 1, f"| **{key}** | {value} |")
        return "\n".join(lines)
    return content


def step_fill_notes():
    print("=" * 60)
    print("填充笔记: 仓库链接 + 论文元数据")
    print("=" * 60)

    cloned = get_cloned_repos()
    print(f"已克隆 {len(cloned)} 个仓库\n")

    filled_code = 0
    filled_meta = 0
    not_found = 0

    for note_dir, (owner_repo, status, title, year, venue) in sorted(NOTE_META.items()):
        note_path = find_note_path(note_dir)
        if not note_path:
            not_found += 1
            continue

        content = note_path.read_text(encoding="utf-8")
        changed = False

        # 元数据：论文、发布日期、出版
        for key, val in [("论文", f"[{title}]"), ("发布日期", year), ("出版", venue)]:
            new_c = fill_table_row(content, key, val)
            if new_c != content:
                content = new_c
                changed = True

        # 代码行
        if owner_repo:
            repo_url = f"https://github.com/{owner_repo}"
            for rd, url in cloned.items():
                if owner_repo in url:
                    repo_url = url
                    break
            new_c = fill_table_row(content, "代码", f"[GitHub]({repo_url})")
            if new_c != content:
                content = new_c
                changed = True

        if changed:
            note_path.write_text(content, encoding="utf-8")
            if owner_repo:
                print(f"  ✅ {note_dir:25s} {title:30s} {year} {venue:25s} +代码")
            else:
                print(f"  ✅ {note_dir:25s} {title:30s} {year} {venue:25s}")
            filled_meta += 1
            if owner_repo:
                filled_code += 1

    print(f"\n📊 结果: {filled_code} 代码行, {filled_meta} 元数据, {not_found} 无笔记")


def step_clone_new():
    print("\n" + "=" * 60)
    print("下载新仓库")
    print("=" * 60)

    cloned = get_cloned_repos()
    to_download = {}
    for note_dir, (owner_repo, status, *_ ) in NOTE_META.items():
        if not owner_repo or status != "✅":
            continue
        if not find_note_path(note_dir):
            continue
        if any(owner_repo in url for url in cloned.values()):
            continue
        to_download[owner_repo] = note_dir

    print(f"需要下载: {len(to_download)} 个\n")
    success = 0
    for owner_repo, note_dir in sorted(to_download.items()):
        dest = REPOS_DIR / owner_repo.replace("/", "-")
        if dest.exists():
            success += 1
            continue
        for url in [f"git@github.com:{owner_repo}.git", f"https://github.com/{owner_repo}.git"]:
            print(f"  📦 {owner_repo:40s}", end=" ")
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                              capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                print("✅")
                success += 1
                break
            print(f"❌ {r.stderr.strip()[:60]}")
            if dest.exists():
                subprocess.run(["rm", "-rf", str(dest)])
        time.sleep(0.5)
    print(f"\n📊 新下载: {success} 个")


def main():
    fill = "--clone-only" not in sys.argv
    clone = "--fill-only" not in sys.argv

    if fill:
        step_fill_notes()
    if clone:
        step_clone_new()


if __name__ == "__main__":
    main()
