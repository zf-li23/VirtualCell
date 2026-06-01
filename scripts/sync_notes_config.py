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
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "docs-viewer" / "src" / "config"
LOADER_PATH = PROJECT_ROOT / "docs-viewer" / "src" / "lib" / "loaders.ts"
CONFIG_PATH = CONFIG_DIR / "notes.ts"
LIB_DIR = PROJECT_ROOT / "docs-viewer" / "src" / "lib"

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
    "fm-multi-omics": "FM + 多组学整合",
    "fm-llm": "FM + LLM",
    "perturbation": "遗传扰动",
    "benchmarks": "评估与 Benchmark",
    "virtual-cell": "虚拟细胞",
    "pathology": "病理基础模型",
    "surveys": "综述与展望",
}

CATEGORY_ORDER = ["fm-classic", "fm-spatial", "fm-world-model",
                  "fm-cross-species", "fm-graph", "fm-multi-omics", "fm-llm",
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
    """扫描所有笔记，从 frontmatter 读取元数据"""
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
            # 优先从 frontmatter 读取，fallback 到推导
            note_id = fm.get("id") or note_dir.name
            title = fm.get("title") or _guess_title_fallback(note_dir.name, text)
            category = fm.get("category") or section.name
            notes.append({
                "id": note_id,
                "title": title,
                "category": category,
                "path": f"{section.name}/{note_dir.name}/README.md",
                "status": status,
                "file": readme,
                "text": text,
            })
    return notes


def _guess_title_fallback(dir_name: str, text: str) -> str:
    """从 H1 标题或目录名推断标题（仅当 frontmatter 没有 title 时使用）"""
    m = re.search(r"^# (.+?)学习笔记", text, re.MULTILINE)
    if m:
        t = m.group(1).strip()
        if t and t != "[模型名称]":
            return t
    m = re.search(r"^# (.+)", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
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


def generate_link_resolver_ts(notes: list[dict]) -> str:
    """生成 linkResolver.ts，从 frontmatter 自动映射路径→id"""
    lines = []
    lines.append('// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成')
    lines.append('// 手动修改将被覆盖！请修改笔记的 frontmatter 后重新运行。')
    lines.append('')
    lines.append('/**')
    lines.append(' * 将 Markdown 中的相对笔记链接路径映射到 note ID。')
    lines.append(' */')
    lines.append('const notePathToId: Record<string, string> = {')
    lines.append("  'README.md': 'overview',")
    for n in notes:
        if n["status"] != "done":
            continue
        lines.append(f"  '{n['path']}': '{n['id']}',")
    lines.append('}')
    lines.append('')
    lines.append('export function resolveNoteId(href: string): string | null {')
    lines.append("  const normalized = href.replace(/^\\.\\//, '').split('#')[0].split('?')[0]")
    lines.append('  return notePathToId[normalized] ?? null')
    lines.append('}')
    lines.append('')
    lines.append('/** 判断是否外部链接 */')
    lines.append('export function isExternalLink(href: string): boolean {')
    lines.append("  return href.startsWith('http://') || href.startsWith('https://')")
    lines.append('}')
    lines.append('')
    lines.append('/** 判断是否锚点链接 */')
    lines.append('export function isAnchorLink(href: string): boolean {')
    lines.append("  return href.startsWith('#')")
    lines.append('}')
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


# ====== 校验与生成工具 ======

CONFIG_FILES_TO_VALIDATE = [
    Path(__file__).resolve().parents[1] / "docs-viewer/src/config/researchGroups.ts",
    Path(__file__).resolve().parents[1] / "docs-viewer/src/config/techLineage.ts",
    Path(__file__).resolve().parents[1] / "docs-viewer/src/config/roadmap.ts",
]


def validate_config_refs(notes: list[dict]):
    """检查所有手动配置文件中引用的 noteId 是否存在于已完成笔记中"""
    done_ids = {n["id"] for n in notes if n["status"] == "done"}
    has_error = False
    for cfg_path in CONFIG_FILES_TO_VALIDATE:
        if not cfg_path.exists():
            continue
        text = cfg_path.read_text(encoding="utf-8")
        refs = re.findall(r"noteId:\s*'([^']+)'", text)
        for m in re.finditer(r"members:\s*\[([^\]]+)\]", text):
            refs.extend(re.findall(r"'([^']+)'", m.group(1)))
        refs = sorted(set(refs))
        bad = [r for r in refs if r not in done_ids]
        if bad:
            print(f"  ❌ {cfg_path.name}: 缺失的 noteId: {bad}")
            has_error = True
        else:
            print(f"  ✅ {cfg_path.name}: 所有 {len(refs)} 个引用均有效")
    if has_error:
        sys.exit(1)
    return not has_error


def find_code_url(note_id: str) -> str:
    """从 data/*.json 查找代码仓库 URL"""
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("id") == note_id and d.get("code_url"):
                return d["code_url"]
        except:
            pass
    return ""


def generate_repo_map_md(notes: list[dict]) -> str:
    """从 notes + data/*.json + repos/ 自动生成 REPO_MAP.md"""
    lines = []
    lines.append("# 论文 ↔ GitHub 仓库映射表")
    lines.append("")
    lines.append("> 📌 此文件由 `scripts/sync_notes_config.py` 自动生成。")
    lines.append("> 手动修改将被覆盖！请修改笔记 frontmatter 或 data/*.json。")
    lines.append(">")
    lines.append("> 图例：")
    lines.append("> - ✅ 仓库已验证可用")
    lines.append("> - ❌ 未找到公开仓库")
    lines.append("")
    from collections import OrderedDict
    cat_notes = OrderedDict()
    for n in notes:
        if n["status"] != "done":
            continue
        cat = n["category"]
        cat_notes.setdefault(cat, []).append(n)
    CAT_HEADERS = {
        "fm-classic": "FM + 经典语言模型", "fm-spatial": "FM + 空间组学",
        "fm-world-model": "FM + 世界模型", "fm-cross-species": "FM + 跨物种/通用嵌入",
        "fm-graph": "FM + 图与网络", "fm-llm": "FM + LLM",
        "perturbation": "遗传扰动", "benchmarks": "评估与 Benchmark",
        "fm-multi-omics": "FM + 多组学整合", "pathology": "病理基础模型", "surveys": "综述与展望",
    }
    for cat, cat_list in cat_notes.items():
        lines.append("")
        lines.append(f"## {CAT_HEADERS.get(cat, cat)}")
        lines.append("")
        lines.append("| 目录名 | 代码仓库 | 状态 |")
        lines.append("|--------|---------|------|")
        for n in cat_list:
            note_id = n["id"]
            # 尝试多个来源获取 code_url
            code_url = ""
            fm = parse_frontmatter(n["text"])
            if "code_url" in fm:
                code_url = fm["code_url"]
            if not code_url:
                code_url = find_code_url(note_id)
            if code_url:
                lines.append(f"| `{note_id}` | [{code_url}]({code_url}) | ✅ |")
            else:
                lines.append(f"| `{note_id}` | ❌ 未找到 | ❌ |")
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
        validate_config_refs(notes)
        return

    if "--generate-repo-map" in sys.argv:
        print("生成 REPO_MAP.md...")
        repo_map = generate_repo_map_md(notes)
        REPO_MAP_PATH = PROJECT_ROOT / "REPO_MAP.md"
        REPO_MAP_PATH.write_text(repo_map, encoding="utf-8")
        print(f"  ✅ {REPO_MAP_PATH}")
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
    LIB_DIR.mkdir(parents=True, exist_ok=True)

    config_ts = generate_notes_ts(notes)
    CONFIG_PATH.write_text(config_ts, encoding="utf-8")
    print(f"  ✅ {CONFIG_PATH}")

    loaders_ts = generate_loaders_ts(notes)
    LOADER_PATH.write_text(loaders_ts, encoding="utf-8")
    print(f"  ✅ {LOADER_PATH}")

    link_resolver_ts = generate_link_resolver_ts(notes)
    LINK_RESOLVER_PATH = LIB_DIR / "linkResolver.ts"
    LINK_RESOLVER_PATH.write_text(link_resolver_ts, encoding="utf-8")
    print(f"  ✅ {LINK_RESOLVER_PATH}")

    # 校验引用
    print("\n校验配置文件引用...")
    validate_config_refs(notes)

    print("\n完成！请运行 npm run build 验证构建。")


if __name__ == "__main__":
    main()
