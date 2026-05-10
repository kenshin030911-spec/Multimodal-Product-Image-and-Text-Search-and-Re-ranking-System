# 多模态商品图文搜索与重排系统

一个面向服饰商品场景的多模态检索与重排项目。系统支持文本搜图、以图搜图、候选召回、特征重排、训练式 reranker、离线评估和前端可视化对比，覆盖从原始商品数据到 Web 演示的完整检索链路。

项目以 Fashion Product Images 类数据集为例，使用 CLIP/FashionCLIP 将商品图片和商品文本映射到同一向量空间，通过向量相似度完成初始召回，再使用规则特征、binary reranker 或 pairwise reranker 对候选商品进行二阶段排序。

## 项目特点

- 端到端检索流程：数据清洗、embedding 生成、索引构建、召回、重排、评估和前端展示。
- 多模态搜索：支持文本 query 检索商品图片，也支持上传图片检索相似商品。
- 多种排序策略：支持 `none`、`rule`、`trained`、`pairwise` 四种 reranker 模式。
- 可复现实验链路：提供训练集构造、binary reranker 训练、pairwise reranker 训练和四方离线评估脚本。
- 工程化组织：后端 API、前端页面、训练脚本、测试代码和 Git 发布说明分层管理。

## 前端页面展示
搜索页面
![img.png](img.png)
搜索结果
![img_1.png](img_1.png)
评估页面
![img_2.png](img_2.png)
评估结果
![img_3.png](img_3.png)

## 系统流程

```text
raw metadata + images
        |
        v
data preprocessing
        |
        v
products.jsonl
        |
        v
CLIP / FashionCLIP encoder
        |
        v
image embeddings + text embeddings
        |
        v
NumPy cosine index
        |
        v
candidate retrieval
        |
        v
rule / binary / pairwise reranker
        |
        v
ranked product results + evaluation reports
```

## 功能模块

| 模块 | 说明 |
| --- | --- |
| 数据处理 | 读取商品 CSV 和图片目录，生成标准化 `products.jsonl` |
| Embedding | 使用 CLIP/FashionCLIP 生成图片向量和文本向量 |
| 向量索引 | 基于图片 embedding 构建 NumPy flat cosine index |
| 文本搜图 | 将文本 query 编码为向量，与图片向量做相似度召回 |
| 以图搜图 | 将上传图片编码为向量，召回视觉相似商品 |
| 重排 | 支持规则重排、binary reranker 和 pairwise reranker |
| 离线评估 | 对 recall-only、rule-based、trained、pairwise 进行指标对比 |
| 前端展示 | 提供搜索页和单 query 多模式对比页 |

## 技术栈

| 层级 | 技术 |
| --- | --- |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| Multimodal Encoder | PyTorch, Transformers, CLIP/FashionCLIP |
| Retrieval | NumPy, cosine similarity |
| Reranker | scikit-learn LogisticRegression |
| Evaluation | NDCG@K, MRR@K, Recall@K |
| Frontend | React, Vite, CSS |

## 项目结构

```text
backend/
  app/                 FastAPI 应用、检索、embedding、reranker、评估和训练逻辑
  scripts/             数据准备、embedding、index、训练和评估脚本
  tests/               后端测试
frontend/
  src/                 React 页面、组件和样式
  package.json         前端依赖与构建脚本
data/
  raw/                 原始数据集目录，默认不提交 Git
  processed/           处理后数据和训练集，默认不提交 Git
  embeddings/          embedding 向量，默认不提交 Git
  index/               检索索引，默认不提交 Git
models/                本地模型和训练好的 reranker，默认不提交 Git
outputs/               训练、评估和报告输出，默认不提交 Git
docs/
  GIT_RELEASE_GUIDE.md Git 发布准备说明
```

## 数据与模型产物

仓库默认只保存源码、脚本、测试、配置样例和文档。以下内容属于可复现的大型运行产物，不直接提交到 Git：

- `data/raw/**`：原始图片和原始 CSV。
- `data/processed/**`：处理后的商品数据和 reranker 训练集。
- `data/embeddings/**`：图片向量、文本向量和 embedding 元数据。
- `data/index/**`：向量检索索引。
- `models/**`：本地下载模型和训练好的 reranker。
- `outputs/**`：训练报告、评估报告和案例分析输出。

这些目录已通过 `.gitignore` 忽略，并通过 `.gitkeep` 或目录 README 保留结构说明。

## 环境准备

Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

前端依赖：

```powershell
cd frontend
npm install
cd ..
```

原始数据放置路径：

```text
data/raw/metadata/styles.csv
data/raw/images/*.jpg
```

如使用本地 FashionCLIP 模型，放置路径为：

```text
models/fashion-clip/
```

也可以在 embedding 命令中将 `--model-name` 指向 Hugging Face 模型名，例如 `patrickjohncyh/fashion-clip`。

## 复现流程

1. 生成标准化商品数据：

```powershell
python backend/scripts/prepare_dataset.py
python backend/scripts/check_processed_data.py --write-json
```

2. 生成 embedding：

```powershell
python backend/scripts/build_embeddings.py --encoder-name fashion-clip --model-name models/fashion-clip --device cpu --batch-size 32 --overwrite
```

快速 smoke test 可使用 dummy encoder：

```powershell
python backend/scripts/build_embeddings.py --encoder-name dummy --overwrite
```

3. 构建检索索引：

```powershell
python backend/scripts/build_index.py --overwrite
```

4. 构造 reranker 训练集：

```powershell
python backend/scripts/build_reranker_dataset.py `
  --query-templates augmented `
  --queries-per-product 3 `
  --max-queries 500 `
  --candidate-k 150 `
  --max-positives-per-query 20 `
  --max-negatives-per-query 40 `
  --output-dir data/processed/reranker_dataset_aug_q500_c150_pos20_neg40 `
  --overwrite
```

5. 训练 binary reranker：

```powershell
python backend/scripts/train_reranker.py `
  --train-path data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/reranker_train.jsonl `
  --valid-path data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/reranker_valid.jsonl `
  --dataset-meta-path data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/reranker_dataset_meta.json `
  --model-output-dir models/reranker `
  --report-output-dir outputs/training `
  --overwrite
```

6. 训练 pairwise reranker：

```powershell
python backend/scripts/train_pairwise_reranker.py --overwrite
```

7. 运行四方离线评估：

```powershell
python backend/scripts/run_evaluation.py `
  --query-templates augmented `
  --queries-per-product 3 `
  --max-queries 300 `
  --candidate-k 150 `
  --include-trained-reranker `
  --trained-model-path models/reranker/trained_reranker.joblib `
  --trained-meta-path models/reranker/trained_reranker_meta.json `
  --include-pairwise-reranker `
  --pairwise-model-path models/reranker_pairwise/pairwise_reranker.joblib `
  --pairwise-meta-path models/reranker_pairwise/pairwise_reranker_meta.json `
  --output-dir outputs/eval_reports_four_way_q300
```

## 启动项目

后端服务：

```powershell
uvicorn backend.app.main:app --reload
```

前端服务：

```powershell
cd frontend
npm run dev
```

默认前端地址：`http://localhost:5173`

## reranker_type

| reranker_type | 排序方式 | 依赖 |
| --- | --- | --- |
| `none` | 仅使用向量召回分数排序 | `data/embeddings/`, `data/index/` |
| `rule` | 使用元数据规则特征重排 | 商品元数据 |
| `trained` | 使用 binary LogisticRegression reranker | `models/reranker/` |
| `pairwise` | 使用 pairwise LogisticRegression reranker | `models/reranker_pairwise/` |

后端 `trained` 模式默认加载 `models/reranker/`，`pairwise` 模式默认加载 `models/reranker_pairwise/`。

## 前端页面

- 搜索页：输入文本或上传图片，展示商品结果、相似度分数、商品属性和重排模式。
- 单 query 多模式对比页：对同一个 query 同时比较 recall-only、rule-based、trained 和 pairwise 的排序结果。

## 离线评估参考

本地四方评估目录为 `outputs/eval_reports_four_way_q300/`，该目录默认不提交 Git。一次参考结果如下：

| 模式 | NDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| recall-only | 0.6361 | 0.8145 | 0.5253 |
| rule-based | 0.9219 | 0.9967 | 0.8323 |
| trained binary | 0.9219 | 0.9967 | 0.8323 |
| pairwise | 0.9873 | 1.0000 | 0.9767 |

## 测试与构建

后端测试：

```powershell
python -m pytest backend/tests -q
```

前端构建：

```powershell
cd frontend
npm run build
```

## 项目局限

- 当前索引为 NumPy flat index，适合中小规模演示，不是生产级 ANN 检索方案。
- weak-supervised query 和标签主要由商品元数据构造，和真实用户行为日志仍有差距。
- binary reranker 和 pairwise reranker 使用轻量 sklearn 模型，表达能力有限。
- 数据集、模型权重、embedding、index 和训练输出需要按脚本本地生成。

## 后续方向

- 接入 FAISS、Milvus 或 Qdrant 支持更大规模向量检索。
- 使用真实点击、收藏或购买日志构造训练标签。
- 引入 cross-encoder 或 transformer reranker 提升排序能力。
- 增加筛选、收藏、错误案例分析和可视化评估面板。
- 增加 Docker Compose 和自动化数据流水线。

## 发布说明

仓库提交范围和大文件排除规则见 [docs/GIT_RELEASE_GUIDE.md](docs/GIT_RELEASE_GUIDE.md)。
