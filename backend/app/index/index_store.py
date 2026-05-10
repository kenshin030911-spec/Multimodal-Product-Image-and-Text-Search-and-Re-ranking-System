"""向量索引文件保存、加载和校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative


IMAGE_INDEX_FILE = "image_index.npy"
INDEX_META_FILE = "index_meta.json"


@dataclass(frozen=True)
class IndexBundle:
    """已保存或已加载的 numpy flat index bundle。"""

    image_index: np.ndarray
    meta: dict[str, Any]


def default_index_output_dir() -> Path:
    """返回默认 index 输出目录。"""
    return get_settings().index_dir


def save_index_bundle(
    output_dir: Path | None,
    meta: dict[str, Any],
    image_index: np.ndarray,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> IndexBundle:
    """保存 image_index.npy 和 index_meta.json，并校验映射一致。"""
    settings = get_settings()
    output_dir = output_dir or default_index_output_dir()
    project_root = project_root or settings.project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    image_index_path = output_dir / IMAGE_INDEX_FILE
    meta_path = output_dir / INDEX_META_FILE
    if not overwrite:
        existing = [path for path in (image_index_path, meta_path) if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"index 输出已存在，请使用 --overwrite 覆盖: {names}")

    normalized_meta = dict(meta)
    normalized_meta["image_index_file"] = to_project_relative(image_index_path, project_root)
    normalized_meta["index_meta_file"] = to_project_relative(meta_path, project_root)
    _validate_index_bundle(image_index, normalized_meta)

    np.save(image_index_path, image_index.astype(np.float32, copy=False))
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(normalized_meta, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return IndexBundle(image_index=image_index, meta=normalized_meta)


def load_index_bundle(output_dir: Path | None = None) -> IndexBundle:
    """加载 index_meta.json 和 image_index.npy。"""
    output_dir = output_dir or default_index_output_dir()
    meta_path = output_dir / INDEX_META_FILE
    image_index_path = output_dir / IMAGE_INDEX_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"index_meta.json 不存在: {meta_path}")
    if not image_index_path.is_file():
        raise FileNotFoundError(f"image_index.npy 不存在: {image_index_path}")

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    image_index = np.load(image_index_path)
    _validate_index_bundle(image_index, meta)
    return IndexBundle(image_index=image_index, meta=meta)


def load_index_placeholder() -> None:
    """兼容旧调用：真实读取请使用 load_index_bundle。"""
    return None


def _validate_index_bundle(image_index: np.ndarray, meta: dict[str, Any]) -> None:
    """校验 index shape、dtype 和 product_id 映射。"""
    product_ids = list(meta.get("product_ids", []))
    product_count = int(meta.get("product_count", -1))
    embedding_dim = int(meta.get("embedding_dim", 0))
    product_id_to_index = dict(meta.get("product_id_to_index", {}))

    if meta.get("index_type") != "numpy_flat":
        raise ValueError("index_type 必须是 numpy_flat。")
    if meta.get("metric") != "cosine":
        raise ValueError("metric 必须是 cosine。")
    if product_count != len(product_ids):
        raise ValueError("product_count 与 product_ids 数量不一致。")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim 必须大于 0。")
    expected_mapping = {product_id: index for index, product_id in enumerate(product_ids)}
    if product_id_to_index != expected_mapping:
        raise ValueError("product_id_to_index 与 product_ids 顺序不一致。")
    if image_index.ndim != 2:
        raise ValueError("image_index 必须是二维数组。")
    if image_index.shape != (product_count, embedding_dim):
        raise ValueError("image_index shape 与 product_count/embedding_dim 不一致。")
    if image_index.dtype != np.float32:
        raise ValueError("image_index dtype 必须是 float32。")
