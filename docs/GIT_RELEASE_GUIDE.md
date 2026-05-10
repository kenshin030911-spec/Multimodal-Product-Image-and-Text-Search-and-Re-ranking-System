# Git Release Guide

本文档用于发布前检查，目标是让仓库只提交可复现项目所需的源码、脚本、测试、配置样例和文档。

## 会提交的内容

- `backend/`
- `frontend/`，不包含 `node_modules/` 和 `dist/`
- `README.md`
- `requirements.txt`
- `frontend/package.json`
- `frontend/package-lock.json`
- `.gitignore`
- `.env.example`
- `backend/tests/`
- `backend/scripts/`
- `data/**/README.md`
- `data/**/.gitkeep`
- `models/README.md`
- `models/.gitkeep`
- `outputs/README.md`
- `outputs/.gitkeep`
- `uploads/tmp/.gitkeep`
- `docs/GIT_RELEASE_GUIDE.md`

## 不会提交的内容

- 原始数据：`data/raw/images/`, `data/raw/metadata/styles.csv`
- 处理后数据：`data/processed/products.jsonl`, `data/processed/dataset_stats.json`, `data/processed/reranker_dataset_*/`
- 向量：`data/embeddings/*.npy`, `data/embeddings/embedding_meta.json`
- 索引：`data/index/*.npy`, `data/index/index_meta.json`
- 本地模型：`models/fashion-clip/`
- 训练模型：`models/reranker/`, `models/reranker_augmented/`, `models/reranker_pairwise/`
- 输出报告：`outputs/**`
- 临时上传：`uploads/tmp/**`
- 环境和缓存：`.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `frontend/node_modules/`, `frontend/dist/`, `frontend/.vite/`

## clone 后如何复现

1. 创建 Python 虚拟环境并安装 `requirements.txt`。
2. 进入 `frontend/` 执行 `npm install`。
3. 准备 `data/raw/metadata/styles.csv` 和 `data/raw/images/`。
4. 执行 `python backend/scripts/prepare_dataset.py`。
5. 执行 `python backend/scripts/build_embeddings.py --encoder-name fashion-clip --model-name models/fashion-clip --device cpu --batch-size 32 --overwrite`。
6. 执行 `python backend/scripts/build_index.py --overwrite`。
7. 执行 `python backend/scripts/build_reranker_dataset.py --query-templates augmented --max-queries 500 --output-dir data/processed/reranker_dataset_aug_q500_c150_pos20_neg40 --overwrite`。
8. 执行 `python backend/scripts/train_reranker.py --model-output-dir models/reranker --overwrite`。
9. 执行 `python backend/scripts/train_pairwise_reranker.py --overwrite`。
10. 执行 `python backend/scripts/run_evaluation.py --query-templates augmented --max-queries 300 --include-trained-reranker --include-pairwise-reranker --output-dir outputs/eval_reports_four_way_q300`。
11. 后端：`uvicorn backend.app.main:app --reload`。
12. 前端：`cd frontend && npm run dev`。

## 发布前检查清单

- `.env` 没有进入 Git。
- `.venv/`、`frontend/node_modules/`、`frontend/dist/` 没有进入 Git。
- `data/` 下只提交 README 和 `.gitkeep`。
- `models/` 下只提交 README 和 `.gitkeep`。
- `outputs/` 下只提交 README 和 `.gitkeep`。
- 已运行 `python -m pytest backend/tests -q`。
- 已运行 `cd frontend && npm run build`。
- 已检查 README 中的命令和脚本参数一致。

## git status 期望状态

初始化 Git 后，`git status --short` 里应该只看到源码、文档、配置样例、测试和脚本。

不应该出现：

```text
data/raw/images/
data/raw/metadata/styles.csv
data/processed/products.jsonl
data/embeddings/image_embeddings.npy
data/index/image_index.npy
models/fashion-clip/
models/reranker/
outputs/eval_reports_four_way_q300/
frontend/node_modules/
frontend/dist/
.venv/
.env
```

## 不应误提交的大文件清单

- 图片数据集目录。
- `styles.csv` 原始元数据。
- `.npy` embedding 和 index。
- Hugging Face / Transformers 本地模型权重。
- `.joblib` 训练模型。
- 评估报告、训练报告和案例分析输出。
- 虚拟环境和前端依赖目录。

## 本地演示需要额外保留的产物

如果只想本地演示，不需要提交，但建议保留：

- `data/raw/`
- `data/processed/products.jsonl`
- `data/embeddings/`
- `data/index/`
- `data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/`
- `models/fashion-clip/`
- `models/reranker/`
- `models/reranker_pairwise/`
- `outputs/eval_reports_four_way_q300/`
