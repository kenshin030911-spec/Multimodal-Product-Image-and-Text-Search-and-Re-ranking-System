# data/index

这里存放检索索引，默认不提交 Git。

常见文件：

- `image_index.npy`
- `index_meta.json`

索引由 embedding 结果构建：

```powershell
python backend/scripts/build_index.py --overwrite
```
