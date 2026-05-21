#!/usr/bin/env python3
"""
fill_repo_links.py

1. 读取 repos/ 中已克隆仓库的实际 remote URL
2. 将这些 URL 填入笔记 README.md 的"代码"行
3. 尝试用 SSH 克隆尚未下载的仓库

用法:
  python scripts/fill_repo_links.py              # 填充 + 下载
  python scripts/fill_repo_links.py --fill-only  # 只填充
  python scripts/fill_repo_links.py --clone-only # 只下载
"""

import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOS_DIR = PROJECT_ROOT / "repos"
NOTES_DIR = PROJECT_ROOT / "notes"

# ========== 笔记目录名 → GitHub owner/repo 映射 ==========
# 格式: "笔记目录名": ("owner/repo", "状态")
# ✅ = 已验证可用  ⚠️ = 待验证
NOTE_REPO_MAP = {
    # === notes/fm/ ===
    "geneformer": ("ctheodoris/Geneformer", "⚠️"),
    "scgpt": ("bowang-lab/scGPT", "✅"),
    "scfoundation": ("biomap-research/scFoundation", "✅"),
    "scbert": ("TencentAILabHealthcare/scBERT", "✅"),
    "scpoli": ("theislab/scPoli", "✅"),
    "scprint": ("jkobject/scPRINT", "✅"),
    "nicheformer": ("theislab/nicheformer", "✅"),
    "novae": ("prism-oncology/novae", "✅"),
    "cell-atlas-fm": ("snap-stanford/UCE", "✅"),
    "saturn": ("snap-stanford/saturn", "✅"),
    "langcell": ("PharMolix/LangCell", "✅"),
    "genecompass": ("biomap-research/GeneCompass", "⚠️"),
    "epiagent": ("Genentech/EpiAgent", "⚠️"),
    "xtrimogene": ("biomap-research/xTrimoGene", "⚠️"),
    "scpeft": ("bowang-lab/scPEFT", "⚠️"),
    "sclong": ("SJTU-GZ-SingleCell/scLong", "⚠️"),
    "cellfm": ("TencentAILabHealthcare/CellFM", "⚠️"),
    "scmulan": ("THU-ML-Lab/scMulan", "⚠️"),
    "scnet": ("TheShed/scNET", "⚠️"),
    "scelmo": ("rsinghlab/scELMo", "⚠️"),
    "scarf": ("clue-team/SCARF", "⚠️"),
    "genejepa": ("jkobject/GeneJepa", "⚠️"),
    "sclinguist": ("Hao-Sheng/scLinguist", "⚠️"),
    "cell-plm": ("ZhiyuanYuan/CellPLM", "⚠️"),
    "cellpolaris": ("ZhiyuanYuan/CellPolaris", "⚠️"),
    "sc-mae": ("Hao-Sheng/scMAE", "⚠️"),
    "scclip": ("ZhiyuanYuan/scCLIP", "⚠️"),
    "scconcept": ("ZhiyuanYuan/scConcept", "⚠️"),
    "scpretrain": ("ZhiyuanYuan/scPretrain", "⚠️"),
    "musegnn": ("PharMolix/MuSe-GNN", "⚠️"),
    "switch": ("PharMolix/SWITCH", "⚠️"),
    "cell-graphcompass": ("PharMolix/Cell-GraphCompass", "⚠️"),
    "scilama": ("PharMolix/sciLaMA", "⚠️"),
    "schiena": ("v000000/scHyena", "⚠️"),
    "epifoundation": ("Genentech/EpiFoundation", "⚠️"),
    "multimodal-perturbation": ("Genentech/multimodal-perturbation", "⚠️"),
    "captain": ("Genentech/CAPTAIN", "⚠️"),
    "cell-niche-graph": ("theislab/cell-niche-graph", "⚠️"),
    "dna-to-expression": ("functionalbio/dna-to-expression", "⚠️"),
    "visual-omics-fm": ("PathologyFoundation/visual-omics-fm", "⚠️"),
    "cell-ontology-fm": ("OmicsML/CellOntologyFM", "⚠️"),
    "atacformer": ("omegafm/atacformer", "⚠️"),
    "chromfound": ("omegafm/chromfound", "⚠️"),
    "cancerfoundation": ("biomap-research/CancerFoundation", "⚠️"),
    "latent-diffusion-sc": ("GenerativeSC/latent-diffusion-sc", "⚠️"),
    "transcription-foundation": ("EnhancerNet/transcription-foundation", "⚠️"),
    "bidir-mamba": ("ciatac/BiDir-Mamba", "⚠️"),
    "scgpt-spatial": ("bowang-lab/scGPT", "✅"),
    "scprint-2": ("jkobject/scPRINT", "✅"),

    # === notes/fm-llm/ ===
    "cassia": ("theislab/CASSIA", "⚠️"),
    "cell2sentence": ("scBio-Lab/cell2sentence", "⚠️"),
    "cellama": ("biomap-research/CELLama", "⚠️"),
    "scchat": ("ZixiangLuo/scChat", "⚠️"),
    "scgenept": ("bowang-lab/scGenePT", "⚠️"),
    "scouter": ("bowang-lab/Scouter", "⚠️"),
    "screader": ("ZhiyuanYuan/scReader", "⚠️"),
    "cellforge": ("cellophane-software/cellforge", "⚠️"),

    # === notes/perturbation/ ===
    "cinema-ot": ("theislab/cinema-ot", "⚠️"),
    "pertadapt": ("theislab/PertAdapt", "⚠️"),
    "state": ("theislab/STATE", "⚠️"),
    "systema": ("theislab/systema", "⚠️"),
    "sclambda": ("theislab/scLAMBDA", "⚠️"),
    "perteval-scfm": ("jkobject/PertEval-scFM", "⚠️"),
    "tahoe-100m": ("MarioniLab/tahoe-100m", "⚠️"),
    "tahoe-x1": ("MarioniLab/tahoe-100m", "⚠️"),
    "in-silico-discovery": ("MarioniLab/insilicodiscovery", "⚠️"),

    # === notes/benchmarks/ ===
    "biollm": ("BioFM/BioLLM", "⚠️"),
    "heimdall": ("BioMap-Research/HEIMDALL", "⚠️"),
    "perturbench": ("BioMap-Research/PerturBench", "⚠️"),
    "scdrugmap": ("BioMap-Research/scDrugMap", "⚠️"),
    "bmfm-rna": ("jkobject/BMFM-RNA", "⚠️"),
    "sccluben": ("ZhiyuanYuan/scCluBench", "⚠️"),
    "sceval": ("WuyangZhu/scEval", "⚠️"),
    "open-problems-sc-analysis": ("theislab/open_problems_sc_analysis", "⚠️"),

    # === notes/virtual-cell/ ===
    "virtual-cell-challenge": ("czbiohub/virtual-cell-challenge", "⚠️"),

    # === notes/pathology/ ===
    "general-purpose-pathology": ("hustvl/UNI", "⚠️"),
    "visual-language-pathology": ("mahmoodlab/CONCH", "✅"),
    "biomedical-seg-det-rec": ("PathologyFoundation/biomedparse", "⚠️"),
}


def get_cloned_repos() -> dict[str, str]:
    """返回 { repo_dir_name: https_url } 的映射"""
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
    """找到笔记文件路径"""
    prefixes = ["fm", "fm-llm", "perturbation", "benchmarks", "virtual-cell", "pathology", "surveys"]
    for p in prefixes:
        path = NOTES_DIR / p / note_dir / "README.md"
        if path.exists():
            return path
    return None


def fill_note_repo_link(note_path: Path, repo_url: str) -> bool:
    """在笔记 README.md 中填入/添加 代码 行"""
    if not note_path.exists():
        return False

    content = note_path.read_text(encoding="utf-8")
    repo_link = f"[GitHub]({repo_url})"

    # 情况 1: 已有 | **代码** | ... | 行 → 替换
    code_pattern = re.compile(r"(\| \*\*代码\*\* \|).*(\|)", re.MULTILINE)
    if code_pattern.search(content):
        new_content = code_pattern.sub(rf"\1 {repo_link} \2", content)
        if new_content != content:
            note_path.write_text(new_content, encoding="utf-8")
            return True
        return False

    # 情况 2: 没有代码行，但有 | **许可** | 行 → 在许可前插入
    license_pattern = re.compile(r"^(\| \*\*许可\*\* \|.*\|)$", re.MULTILINE)
    if license_pattern.search(content):
        new_content = license_pattern.sub(
            f"| **代码** | {repo_link} |\n\\1", content
        )
        note_path.write_text(new_content, encoding="utf-8")
        return True

    # 情况 3: 在表格末尾追加
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
        lines.insert(table_last + 1, f"| **代码** | {repo_link} |")
        note_path.write_text("\n".join(lines), encoding="utf-8")
        return True

    return False


def step_fill_notes():
    """将已克隆仓库的 URL 填入笔记"""
    print("=" * 60)
    print("步骤 1: 将已克隆仓库的 URL 填入笔记")
    print("=" * 60)

    cloned = get_cloned_repos()
    print(f"已克隆 {len(cloned)} 个仓库\n")

    filled = 0
    skipped_no_note = 0
    skipped_no_match = 0
    not_found = 0

    for note_dir, (owner_repo, status) in sorted(NOTE_REPO_MAP.items()):
        # 查找是否已克隆
        repo_url = None
        for rd, url in cloned.items():
            if owner_repo in url:
                repo_url = url
                break

        if not repo_url:
            not_found += 1
            continue

        note_path = find_note_path(note_dir)
        if not note_path:
            skipped_no_note += 1
            continue

        if fill_note_repo_link(note_path, repo_url):
            print(f"  ✅ {note_dir:25s} → {repo_url}")
            filled += 1
        else:
            skipped_no_match += 1

    print(f"\n📊 结果: {filled} 已填入, {skipped_no_note} 无笔记, "
          f"{skipped_no_match} 匹配失败, {not_found} 未克隆")


def step_clone_new():
    """克隆尚未下载的仓库"""
    print("\n" + "=" * 60)
    print("步骤 2: 下载新仓库（使用 SSH）")
    print("=" * 60)

    cloned = get_cloned_repos()

    # 过滤出需要下载的确认仓库
    to_download = {}
    for note_dir, (owner_repo, status) in NOTE_REPO_MAP.items():
        if status != "✅":
            continue
        if not find_note_path(note_dir):
            continue
        already = any(owner_repo in url for url in cloned.values())
        if not already:
            to_download[owner_repo] = note_dir

    print(f"需要下载: {len(to_download)} 个\n")

    success = 0
    for owner_repo, note_dir in sorted(to_download.items()):
        dest = REPOS_DIR / owner_repo.replace("/", "-")
        if dest.exists():
            print(f"  ⏭️  {owner_repo}")
            success += 1
            continue

        ssh_url = f"git@github.com:{owner_repo}.git"
        https_url = f"https://github.com/{owner_repo}.git"
        print(f"  📦 {owner_repo:40s}", end=" ")

        r = subprocess.run(
            ["git", "clone", "--depth", "1", ssh_url, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            print("✅ (SSH)")
            success += 1
            time.sleep(0.5)
            continue

        r2 = subprocess.run(
            ["git", "clone", "--depth", "1", https_url, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if r2.returncode == 0:
            print("✅ (HTTPS)")
            success += 1
        else:
            print(f"❌ {r2.stderr.strip()[:80]}")
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
