# scripts — 可复现实验

> 本目录用于存放可运行的 Python 脚本和 Jupyter Notebook，覆盖模型推理、基准测试和数据分析。

## 目录结构

```
scripts/
├── README.md              ← 你在这里
├── tutorials/             # 模型入门教程
│   ├── geneformer_basics.ipynb
│   └── scgpt_pbmc3k.ipynb
├── benchmarks/            # 基准测试脚本
│   └── eval_embeddings.ipynb
└── utils/                 # 工具函数
    └── data_loader.py
```

## 环境

所有脚本使用 `zf-li23` conda 环境：

```bash
conda activate zf-li23
```

## 数据

公开数据集放在 `../data/` 目录下。当前已有：

| 文件 | 来源 | 大小 |
|------|------|------|
| `pbmc3k.h5ad` | scanpy.datasets.pbmc3k() | ~21 MB |
| `pbmc3k_raw.h5ad` | scanpy.datasets.pbmc3k() | ~5.6 MB |

## 添加新脚本

1. 选择合适的子目录（tutorials / benchmarks / utils）
2. 以 `.py` 或 `.ipynb` 格式添加
3. 在文件头部注明依赖和用法
