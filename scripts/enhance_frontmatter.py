#!/usr/bin/env python3
"""
enhance_frontmatter.py

给所有 status=done 的笔记添加/补全 frontmatter 字段：
  id, title, category, code_url

用法: python3 scripts/enhance_frontmatter.py
"""

import json, re, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "notes"
DATA_DIR = PROJECT_ROOT / "data"

# frontmatter 正则
FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# 分类由父目录名决定（当前已经是最新分类结构）
CATEGORY_MAP = {
    "fm-classic": "fm-classic",
    "fm-spatial": "fm-spatial",
    "fm-world-model": "fm-world-model",
    "fm-cross-species": "fm-cross-species",
    "fm-graph": "fm-graph",
    "fm-llm": "fm-llm",
    "perturbation": "perturbation",
    "benchmarks": "benchmarks",
    "virtual-cell": "virtual-cell",
    "pathology": "pathology",
    "surveys": "surveys",
}

def load_code_urls():
    """从 data/*.json 加载 code_url 映射"""
    mapping = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            id_ = d.get("id") or f.stem
            if d.get("code_url"):
                mapping[id_] = d["code_url"]
        except:
            pass
    return mapping

def parse_frontmatter(text):
    m = FM_PATTERN.match(text)
    if not m:
        return {}, text
    fm = {}
    for line in m.group(1).strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, m.group(0)

def write_frontmatter(text, new_fm):
    """用新的 frontmatter 字典替换旧的"""
    old_fm_str = FM_PATTERN.match(text).group(0) if FM_PATTERN.match(text) else ""
    lines = ["---"]
    for k in ["status", "filled", "id", "title", "category"]:
        if k in new_fm:
            lines.append(f"{k}: {new_fm[k]}")
    if new_fm.get("code_url"):
        lines.append(f"code_url: {new_fm['code_url']}")
    lines.append("---")
    lines.append("")
    new_str = "\n".join(lines)
    if old_fm_str:
        # 确保 old_fm_str 后紧跟着正文
        rest = text[len(old_fm_str):]
        # 跳过 old_fm_str 后可能已有的空行
        rest = rest.lstrip("\n")
        return new_str + rest
    else:
        return new_str + text.lstrip()

def get_h1_title(text):
    """从 markdown 正文提取 H1 标题"""
    m = re.search(r"^# (.+?)学习笔记", text, re.MULTILINE)
    if m:
        t = m.group(1).strip()
        if t and t != "[模型名称]":
            return t
    m = re.search(r"^# (.+)", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""

def main():
    code_urls = load_code_urls()
    updated = 0
    
    for section in sorted(NOTES_DIR.iterdir()):
        if not section.is_dir() or section.name.startswith("."):
            continue
        category = CATEGORY_MAP.get(section.name)
        if category is None:
            continue  # 跳过未知目录（如 notes/fm/ 模板残留）
        
        for note_dir in sorted(section.iterdir()):
            readme = note_dir / "README.md"
            if not readme.exists():
                continue
            text = readme.read_text(encoding="utf-8")
            
            # 只处理 done 笔记
            fm, fm_str = parse_frontmatter(text)
            if fm.get("status") != "done":
                continue
            
            note_id = fm.get("id") or note_dir.name
            
            # 提取标题
            title = fm.get("title") or get_h1_title(text) or note_dir.name.replace("-", " ").title()
            
            # 查找 code_url
            code_url = fm.get("code_url") or code_urls.get(note_id) or code_urls.get(note_dir.name) or ""
            
            new_fm = {
                "status": "done",
                "filled": fm.get("filled", "2026-06-01"),
                "id": note_id,
                "title": title,
                "category": category,
            }
            if code_url:
                new_fm["code_url"] = code_url
            
            old_fm_str = FM_PATTERN.match(text).group(0) if FM_PATTERN.match(text) else ""
            new_text = write_frontmatter(text, new_fm)
            
            if new_text != text:
                readme.write_text(new_text, encoding="utf-8")
                print(f"  ✅ {section.name}/{note_dir.name}: id={note_id}, category={category}")
                updated += 1
            else:
                # 检查是否真的没变
                print(f"  ➖ {section.name}/{note_dir.name}: 无需更新")
    
    print(f"\n总计更新: {updated} 个笔记")

if __name__ == "__main__":
    main()
