# data/embeddings

这里存放商品图片和文本 embedding，默认不提交 Git。

常见文件：

- `image_embeddings.npy`
- `text_embeddings.npy`
- `embedding_meta.json`

这些文件由 embedding 脚本生成：

```powershell
python backend/scripts/build_embeddings.py --encoder-name fashion-clip --model-name models/fashion-clip --device cpu --batch-size 32 --overwrite
```

如果只想快速跑通流程，可以使用：

```powershell
python backend/scripts/build_embeddings.py --encoder-name dummy --overwrite
```
