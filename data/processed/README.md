# data/processed

这里存放处理后的商品数据和 reranker 训练集，默认不提交 Git。

常见文件：

- `products.jsonl`
- `dataset_stats.json`
- `reranker_dataset_*/reranker_train.jsonl`
- `reranker_dataset_*/reranker_valid.jsonl`
- `reranker_dataset_*/reranker_dataset_meta.json`

生成商品数据：

```powershell
python backend/scripts/prepare_dataset.py
```

生成 reranker 训练集：

```powershell
python backend/scripts/build_reranker_dataset.py --query-templates augmented --max-queries 500 --output-dir data/processed/reranker_dataset_aug_q500_c150_pos20_neg40 --overwrite
```
