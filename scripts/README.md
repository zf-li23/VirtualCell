# scripts — 工具脚本

## 文件说明

| 脚本 | 用途 |
|------|------|
| `generate_note.py` | **从结构化数据生成完整笔记**（JSON 或 CLI 参数）。完整文件写入，自动校验残留，自动更新前端配置。 |
| `fill_repo_links.py` | 从 `repos/` 获取仓库 URL 填入笔记代码行 + 论文元数据 |
| `sync_notes_config.py` | **自动发现** `notes/` 中已撰写的笔记（status=done），生成前端 `notes.ts` + `loaders.ts` |
| `copy_template.sh` | 将 `_template.md` 复制到未填充的笔记目录 |

## 撰写新笔记的标准流程

```bash
# 1. 准备结构化数据 (JSON)
cat > data/scgpt.json << 'EOF'
{
  "id": "scgpt",
  "title": "scGPT",
  "year": "2024",
  "venue": "Nature Methods",
  "category": "fm",
  ...
}
EOF

# 2. 生成笔记（完整文件写入，自动校验）
python scripts/generate_note.py data/scgpt.json

# 3. 构建验证
cd docs-viewer && npm run build
```

## 笔记状态标签（Frontmatter）

每篇笔记文件头包含 YAML frontmatter：

```yaml
---
status: done    # template | metadata | done
filled: 2026-05-27
---
```

- `template` — 空白模板，等待填写
- `metadata` — 仅有论文元数据（标题/年份/期刊等）
- `done` — 已撰写深度内容

运行 `sync_notes_config.py` 会自动扫描 `status=done` 的笔记并注册到前端。

