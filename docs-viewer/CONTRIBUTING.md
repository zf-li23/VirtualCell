# docs-viewer 开发指南

> VirtualCell 笔记浏览器 — React 19 + TypeScript + Vite

---

## 快速开始

```bash
cd docs-viewer
npm install       # 安装依赖（已有 node_modules 可跳过）
npm run dev       # 启动开发服务器 (http://localhost:5173/VirtualCell/)
```

## 构建

```bash
npm run build     # 类型检查 + Vite 构建 → dist/
npm run preview   # 预览构建产物
npm run lint      # ESLint 检查
```

## 目录结构

```
docs-viewer/
├── public/
│   ├── favicon.svg
│   ├── icons.svg
│   └── notes -> ../../notes     # symlink 到笔记目录
├── src/
│   ├── main.tsx                 # 入口
│   ├── App.tsx                  # 主布局 (Sidebar + NoteViewer)
│   ├── App.css                  # 布局样式
│   ├── index.css                # 全局样式 (CSS 变量 + 暗色主题)
│   ├── components/
│   │   ├── Sidebar.tsx          # 左侧导航栏
│   │   ├── NoteViewer.tsx       # 笔记阅读器
│   │   ├── Markdown.tsx         # Markdown 渲染 (react-markdown)
│   │   └── Roadmap.tsx          # 学习路线图组件
│   ├── hooks/
│   │   └── useNotes.ts          # 笔记加载/搜索/状态管理
│   ├── config/
│   │   ├── notes.ts             # 笔记元数据 (ID / 标题 / 分类)
│   │   └── roadmap.ts           # 学习路线图配置
│   └── lib/
│       ├── loaders.ts           # 笔记内容加载器 (fetch)
│       └── linkResolver.ts      # 笔记间链接解析
```

## 添加新笔记

1. 在 `../notes/<分类>/<模型名>/README.md` 创建笔记（参考 `../notes/_template.md`）
2. 在 `src/config/notes.ts` 中添加 `NoteMeta` 条目
3. 在 `src/lib/loaders.ts` 中添加对应的 loader
4. 在 `src/lib/linkResolver.ts` 中添加路径映射（确保笔记间互链可用）
5. （可选）在 `src/config/roadmap.ts` 中加入学习路线

## 技术栈

| 用途 | 选型 |
|------|------|
| 框架 | React 19 |
| 语言 | TypeScript ~6.0 |
| 构建 | Vite 8 |
| Markdown | react-markdown + remark-gfm |
| 代码高亮 | react-syntax-highlighter (Prism) |
| 样式 | 纯 CSS (CSS 变量 + 暗色模式) |

## 注意事项

- 笔记内容通过 `fetch('/notes/...')` 加载，依赖 `public/notes` symlink
- 新分类需同时更新 `src/config/notes.ts` 中的 `categories` 数组
- 前端没有路由库，笔记切换通过状态管理实现；如需深度链接，考虑引入 `react-router`
