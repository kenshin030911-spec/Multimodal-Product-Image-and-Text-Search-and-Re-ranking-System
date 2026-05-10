# data/raw

这里存放原始商品数据集，不提交 Git。

期望结构：

```text
data/raw/metadata/styles.csv
data/raw/images/*.jpg
```

这个目录需要用户自行下载或准备数据。准备完成后运行：

```powershell
python backend/scripts/prepare_dataset.py
```

脚本会读取这里的 CSV 和图片，生成 `data/processed/products.jsonl` 与 `data/processed/dataset_stats.json`。
