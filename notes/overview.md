# 🧬 VirtualCell 笔记浏览器

> 单细胞基础模型 | 虚拟细胞 | 空间组学 | 深度学习

---

## 📊 统计概览

当前已涵盖 **26 篇深度笔记**，覆盖以下分类：

| 分类 | 篇数 | 代表性笔记 |
|------|:----:|-----------|
| 单细胞基础模型 (FM) | 16 | Geneformer, scGPT, scFoundation, scBERT, scPoli, scPRINT, Nicheformer, Novae, UCE, SATURN, LangCell, GeneCompass, EpiAgent, Visual-Omics FM, xTrimoGene, scLong, CellFM |
| FM + LLM | 4 | Cell2Sentence, CELLama, CASSIA, scChat |
| 遗传扰动 | 3 | Tahoe-100M/x1, STATE, PertAdapt |
| 虚拟细胞 | 2 | Virtual Cell Challenge, The Virtual Cell |
| 评估与 Benchmark | 1 | Virtual Cell Challenge |

## 🗺️ 学习路线

```
基础模型 →  了解主流架构 (BERT/GPT/AE/GNN)
    ↓
    FM+LLM →  细胞与语言模型的交叉
    ↓
  扰动预测 →  从描述到预测的跨越
    ↓
 虚拟细胞 →  终极目标
```

## 📝 笔记状态

每篇笔记文件头包含 frontmatter 标签：

- `status: template` — 空白模板
- `status: metadata` — 仅有论文元数据
- `status: done` — 已撰写深度内容

只有 `status: done` 的笔记会自动出现在侧边栏中。

## 🔧 工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/generate_note.py` | 从结构化数据生成完整笔记 |
| `scripts/fill_repo_links.py` | 填充仓库链接和元数据 |
| `scripts/sync_notes_config.py` | 自动发现笔记 + 模板残留检测 |
| `scripts/copy_template.sh` | 初始化笔记模板 |
