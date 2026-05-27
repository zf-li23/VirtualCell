#!/usr/bin/env python3
"""
generate_note.py

从结构化数据生成完整的笔记文件。
特点：
  - 完整文件写入（不是部分替换），杜绝模板残留
  - 写入后自动运行 sync_notes_config.py 校验
  - 支持 JSON 输入和命令行参数两种方式

用法:
  # 方式1: JSON 文件
  python scripts/generate_note.py data/my_note.json

  # 方式2: 命令行参数
  python scripts/generate_note.py \\
    --id scgpt --title "scGPT" --year 2024 --venue "Nature Methods" \\
    --category fm --arch "GPT (Causal)" --task "生成式预训练" \\
    --params "50M" --data "33M+ 细胞" \\
    --body notes/fm/scgpt/body.md

JSON 格式示例: scripts/generate_note.py --example
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = PROJECT_ROOT / "notes"

# 分类映射
CATEGORY_MAP = {
    "fm": "fm",
    "fm-llm": "fm-llm",
    "perturbation": "perturbation",
    "benchmarks": "benchmarks",
    "virtual-cell": "virtual-cell",
    "pathology": "pathology",
    "surveys": "surveys",
}

TEMPLATE = """\
---
status: done
filled: {date}
---

# {title} 学习笔记

> {summary}

---

## 📋 目录

1. [模型概述](#1-模型概述)
2. [模型架构](#2-模型架构)
3. [核心创新](#3-核心创新)
4. [数据预处理](#4-数据预处理)
5. [Tokenization 与输入编码](#5-tokenization-与输入编码)
6. [预训练](#6-预训练)
7. [下游任务](#7-下游任务)
8. [代码结构速览](#8-代码结构速览)
9. [关键概念 Q&A](#9-关键概念-qa)
10. [延伸阅读](#10-延伸阅读)

---

## 1. 模型概述

| 属性 | 描述 |
|------|------|
| **论文** | [{title}]({paper_url}) |
| **发布日期** | {year} |
| **出版** | {venue} |
| **架构** | {arch} |
| **预训练任务** | {task} |
| **输入** | {input} |
| **输出** | {output} |
| **词表** | {vocab} |
| **参数规模** | {params} |
| **预训练数据** | {data} |
| **代码** | [GitHub]({code_url}) |
| **许可** | {license} |

### 核心思想

> {summary}

---

## 2. 模型架构

{architecture}

## 3. 核心创新

{innovation}

## 4. 数据预处理

{preprocessing}

## 5. Tokenization 与输入编码

{tokenization}

## 6. 预训练

{pretrain}

## 7. 下游任务

{downstream}

## 8. 代码结构速览

{code_structure}

## 9. 关键概念 Q&A

{qa}

## 10. 延伸阅读

{references}
"""


def show_example():
    print("""\
{
  "id": "scgpt",
  "title": "scGPT",
  "year": "2024",
  "venue": "Nature Methods",
  "category": "fm",
  "paper_url": "https://www.nature.com/articles/s41592-024-02201-0",
  "code_url": "https://github.com/bowang-lab/scGPT",
  "summary": "一句话描述这个模型的核心思想。",
  "arch": "GPT (Causal Transformer)",
  "task": "生成式预训练 / MLM",
  "input": "基因 pair token + 表达值",
  "output": "基因表达预测 / 细胞嵌入",
  "vocab": "~30K 基因",
  "params": "~50M",
  "data": "33M+ 细胞",
  "license": "MIT",
  "architecture": "### 2.1 整体架构\\n\\n架构描述...",
  "innovation": "### 3.1 核心创新\\n\\n创新描述...",
  "preprocessing": "### 4.1 Pipeline\\n\\n预处理描述...",
  "tokenization": "### 5.1 基因编码\\n\\n编码描述...",
  "pretrain": "### 6.1 预训练数据\\n\\n预训练描述...",
  "downstream": "| 任务 | 方法 | 性能 |\\n| --- | --- | --- |\\n| 细胞注释 | Fine-tune | SOTA |",
  "code_structure": "```\\nproject/\\n├── model.py\\n└── ...\\n```",
  "qa": "**Q: 核心区别？**\\n\\n**A**: ...",
  "references": "- [论文](url)\\n- [代码](url)"
}""")


def parse_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> dict:
    args = sys.argv[1:]
    if not args:
        show_example()
        sys.exit(0)

    if args[0] == "--example":
        show_example()
        sys.exit(0)

    data = {}

    # JSON file input
    json_path = Path(args[0])
    if json_path.suffix == ".json" and json_path.exists():
        data = parse_json(json_path)
        # 如果 JSON 中有 body 字段，解析路径（相对 CWD）
        if "body" in data:
            body_path = Path(data["body"])
            if not body_path.is_absolute():
                body_path = Path.cwd() / body_path
            data["body"] = str(body_path.resolve())

    # CLI arg input (覆盖/补充 JSON)
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:].replace("-", "_")
            i += 1
            if i < len(args) and not args[i].startswith("--"):
                data[key] = args[i]
                i += 1
        else:
            i += 1
    return data


def validate_data(data: dict) -> list[str]:
    required = ["id", "title", "year", "venue", "category"]
    missing = [k for k in required if k not in data]
    if missing:
        return [f"缺少必填字段: {', '.join(missing)}"]
    if data["category"] not in CATEGORY_MAP:
        return [f"无效分类: {data['category']}，可选: {list(CATEGORY_MAP.keys())}"]
    return []


def generate_note(data: dict) -> str:
    from datetime import date

    defaults = {
        "paper_url": "",
        "code_url": "",
        "summary": "",
        "arch": "",
        "task": "",
        "input": "",
        "output": "",
        "vocab": "",
        "params": "",
        "data": "",
        "license": "",
        "architecture": "",
        "innovation": "",
        "preprocessing": "",
        "tokenization": "",
        "pretrain": "",
        "downstream": "",
        "code_structure": "",
        "qa": "",
        "references": "",
    }

    # body content from file
    body_path = data.get("body")
    if body_path:
        bp = Path(body_path)
        if bp.exists():
            body = bp.read_text(encoding="utf-8")
            # Parse body by ## section headers
            current_section = None
            for line in body.split("\n"):
                m = re.match(r"^## (\d+)\.\s+(.+)", line)
                if m:
                    section_map = {
                        "2": "architecture", "3": "innovation",
                        "4": "preprocessing", "5": "tokenization",
                        "6": "pretrain", "7": "downstream",
                        "8": "code_structure", "9": "qa", "10": "references",
                    }
                    current_section = section_map.get(m.group(1))
                elif current_section and current_section in defaults:
                    defaults[current_section] += line + "\n"

    # Merge
    for k, v in defaults.items():
        if k not in data or not data[k]:
            data[k] = v

    data["date"] = date.today().isoformat()

    return TEMPLATE.format(**data)


def write_note(content: str, category: str, note_id: str) -> Path:
    dest = NOTES_DIR / category / note_id / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest


def validate_note(path: Path) -> list[str]:
    """检查生成的笔记是否有模板残留"""
    text = path.read_text(encoding="utf-8")
    markers = [
        "用 ASCII 艺术图展示整体流程",
        "│Encoder  │  ← 组件名",
        "[组件1 名称]", "[组件2 名称]",
        "### 3.1 [创新点1]",
        "模型与 X 模型的核心区别是什么",
        "笔记最后更新：YYYY-MM-DD",
        "[相关论文1]",
    ]
    found = [m for m in markers if m in text]
    return found


def main():
    data = parse_args()

    errors = validate_data(data)
    if errors:
        for e in errors:
            print(f"❌ {e}")
        sys.exit(1)

    print(f"生成笔记: {data['id']} ({data['title']}, {data['year']} {data['venue']})")

    content = generate_note(data)
    path = write_note(content, CATEGORY_MAP[data["category"]], data["id"])
    print(f"  写入: {path} ({len(content.split(chr(10)))} 行)")

    # 校验
    remnants = validate_note(path)
    if remnants:
        print(f"  ❌ 发现 {len(remnants)} 处模板残留!")
        for r in remnants:
            print(f"     - {r}")
        sys.exit(1)

    print(f"  ✅ 模板残留检查通过")

    # 重新生成前端配置
    print(f"  运行 sync_notes_config.py...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "sync_notes_config.py")],
        capture_output=True, text=True,
    )
    print(f"  sync_notes_config: {'✅' if result.returncode == 0 else '❌'}")

    print(f"\n✅ {data['id']} 完成!")


if __name__ == "__main__":
    main()
