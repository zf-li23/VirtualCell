# 🚀 部署到 GitHub Pages

本文档说明如何将 VirtualCell 笔记浏览器部署到 GitHub Pages。

## 前置条件

- GitHub 仓库已推送到 `https://github.com/<你的用户名>/VirtualCell`
- 仓库的 **Settings > Pages > Build and deployment** 中，「Source」选择 **GitHub Actions**

## 自动部署（推荐）

项目已配置 GitHub Actions workflow（`.github/workflows/deploy-pages.yml`），每次 push 到 `main` 分支时自动构建并部署。

### 启用 GitHub Pages

1. 打开 GitHub 仓库 → **Settings** → **Pages**
2. 在 **Build and deployment** 下：
   - **Source**: 选择 `GitHub Actions`
3. 保存

### 触发部署

```bash
git add .
git commit -m "docs-viewer: 适配 GitHub Pages 部署"
git push origin main
```

推送后，GitHub Actions 会自动运行 `deploy-pages.yml`，约 1-2 分钟完成部署。

部署成功后，网站地址为：

```
https://<你的用户名>.github.io/VirtualCell/
```

### 手动触发

也可以手动触发部署：

1. GitHub 仓库 → **Actions** → **Deploy to GitHub Pages**
2. 点击 **Run workflow** → **Run workflow**

## 部署流程说明

```
git push → GitHub Actions 触发
  ├─ Checkout 代码
  ├─ npm ci (安装依赖)
  ├─ npm run build (Vite 构建，BASE=/VirtualCell/)
  ├─ upload-pages-artifact (上传 dist/)
  └─ deploy-pages (部署到 GitHub Pages)
```

### 构建时自动设置 Base 路径

`vite.config.ts` 中通过环境变量 `BASE` 控制路径前缀：

- **本地开发**：`BASE` 未设置 → 默认 `/`
- **GitHub Pages 部署**：CI 中设置 `BASE=/VirtualCell/`

如果你 fork 了仓库或改了仓库名，修改 `.github/workflows/deploy-pages.yml` 中的 `BASE` 变量即可：

```yaml
env:
  BASE: /你的仓库名/
```

## 本地开发与预览

```bash
cd docs-viewer

# 安装依赖
npm install

# 启动开发服务器（hot reload）
npm run dev
# → http://localhost:5173

# 本地预览生产构建（模拟 GitHub Pages）
BASE=/VirtualCell/ npm run build
npx vite preview
# → http://localhost:4173/VirtualCell/
```

## 结构说明

```
docs-viewer/
├── public/notes → ../../notes    # 符号链接，构建时笔记会被复制到 dist/
├── src/
│   ├── components/               # React 组件
│   ├── config/                   # 笔记注册表 & 学习路线图
│   ├── hooks/                    # 数据加载逻辑
│   └── lib/                      # 链接解析 & 笔记加载器
├── dist/                         # 构建产物（已被 .gitignore 忽略）
├── vite.config.ts                # Vite 配置（含 base 路径）
└── DEPLOY.md                     # 你在这里
```

## 故障排查

### 部署后页面空白 / 404

检查 `base` 路径是否与仓库名一致：

1. 仓库 URL：`https://github.com/用户名/VirtualCell`
2. Pages URL：`https://用户名.github.io/VirtualCell/`
3. `BASE` 应为：`/VirtualCell/`

### CSS / JS 加载失败

打开浏览器 DevTools → Network，确认资源请求路径以 `/VirtualCell/` 开头。

### 笔记加载失败

确认 `notes/` 目录下有对应的 `.md` 文件。`public/notes` 是符号链接，构建时会被复制到 `dist/notes/`。
