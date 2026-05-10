"""NumPy flat cosine 向量索引构建。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative
from backend.app.embedding.embedding_store import EMBEDDING_META_FILE, load_embedding_bundle
from backend.app.index.index_store import IndexBundle, save_index_bundle


SUPPORTED_METRIC = "cosine"
INDEX_TYPE = "numpy_flat"


@dataclass(frozen=True)
class IndexBuildResult:
    """index 构建结果。"""

    bundle: IndexBundle
    product_count: int
    embedding_dim: int


def build_index(
    embedding_dir: Path | None = None,
    output_dir: Path | None = None,
    metric: str = SUPPORTED_METRIC,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> IndexBuildResult:
    """从 embedding bundle 构建归一化 image index。"""
    if metric != SUPPORTED_METRIC:
        raise ValueError("第一版 index metric 只支持 cosine。")

    settings = get_settings()
    project_root = project_root or settings.project_root
    embedding_dir = embedding_dir or settings.embeddings_dir
    output_dir = output_dir or settings.index_dir

    embedding_bundle = load_embedding_bundle(embedding_dir)
    image_embeddings = embedding_bundle.image_embeddings
    if image_embeddings is None:
        raise ValueError("embedding bundle 中缺少 image_embeddings.npy。")

    source_meta = embedding_bundle.meta
    product_ids = list(source_meta.get("product_ids", []))
    if not product_ids:
        raise ValueError("embedding_meta.json 中 product_ids 不能为空。")

    product_id_to_index = dict(source_meta.get("product_id_to_index", {}))
    expected_mapping = {product_id: index for index, product_id in enumerate(product_ids)}
    if product_id_to_index != expected_mapping:
        raise ValueError("embedding product_id_to_index 与 product_ids 顺序不一致。")

    embedding_dim = int(source_meta.get("embedding_dim", 0))
    if image_embeddings.shape != (len(product_ids), embedding_dim):
        raise ValueError("image_embeddings shape 与 product_ids/embedding_dim 不一致。")

    image_index = l2_normalize_matrix(image_embeddings.astype(np.float32, copy=False))
    meta: dict[str, Any] = {
        "index_type": INDEX_TYPE,
        "metric": metric,
        "embedding_source": to_project_relative(embedding_dir, project_root),
        "embedding_meta_file": to_project_relative(embedding_dir / EMBEDDING_META_FILE, project_root),
        "image_index_file": None,
        "embedding_dim": embedding_dim,
        "product_count": len(product_ids),
        "product_ids": product_ids,
        "product_id_to_index": expected_mapping,
        "source_encoder_name": source_meta.get("encoder_name"),
        "source_model_name": _normalize_source_model_name(source_meta.get("model_name"), project_root),
        "source_is_dummy": bool(source_meta.get("is_dummy", False)),
        "source_normalize_embeddings": bool(source_meta.get("normalize_embeddings", False)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    bundle = save_index_bundle(
        output_dir=output_dir,
        meta=meta,
        image_index=image_index,
        overwrite=overwrite,
        project_root=project_root,
    )
    return IndexBuildResult(
        bundle=bundle,
        product_count=len(product_ids),
        embedding_dim=embedding_dim,
    )


def l2_normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    """逐行 L2 normalize；零向量保持为零向量。"""
    if vectors.ndim != 2:
        raise ValueError("vectors 必须是二维数组。")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return (vectors / safe_norms).astype(np.float32, copy=False)


def build_index_placeholder() -> dict[str, bool]:
    """兼容旧调用：真实构建请使用 build_index。"""
    return {"built": False, "placeholder": True}


def _normalize_source_model_name(model_name: object, project_root: Path) -> str | None:
    """把本地模型绝对路径转成项目相对路径，远端模型名保持原样。"""
    if model_name is None:
        return None
    model_name_text = str(model_name)
    model_path = Path(model_name_text)
    if model_path.is_absolute():
        return to_project_relative(model_path, project_root)
    return model_name_text.replace("\\", "/")
