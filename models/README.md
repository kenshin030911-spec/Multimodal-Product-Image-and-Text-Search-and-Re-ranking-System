# models

这里存放本地模型和训练好的 reranker，默认不提交 Git。

常见目录：

- `fashion-clip/`：本地下载的 FashionCLIP 模型。
- `reranker/`：binary reranker，后端 `trained` 模式默认加载。
- `reranker_augmented/`：旧实验或对比用 binary reranker。
- `reranker_pairwise/`：pairwise reranker，后端 `pairwise` 模式默认加载。

训练 binary reranker：

```powershell
python backend/scripts/train_reranker.py --model-output-dir models/reranker --overwrite
```

训练 pairwise reranker：

```powershell
python backend/scripts/train_pairwise_reranker.py --overwrite
```
