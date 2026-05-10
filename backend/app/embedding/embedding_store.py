"""embedding 文件保存、加载和校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative


IMAGE_EMBEDDING_FILE = "image_embeddings.npy"
TEXT_EMBEDDING_FILE = "text_embeddings.npy"
EMBEDDING_META_FILE = "embedding_meta.json"


@dataclass(frozen=True)
class EmbeddingBundle:
    """已保存或已加载的 embedding bundle。"""

    image_embeddings: np.ndarray | None
    text_embeddings: np.ndarray | None
    meta: dict[str, Any]


def default_embedding_output_dir() -> Path:
    """返回默认 embedding 输出目录。"""
    return get_settings().embeddings_dir


def save_embedding_bundle(
    output_dir: Path | None,
    meta: dict[str, Any],
    image_embeddings: np.ndarray | None = None,
    text_embeddings: np.ndarray | None = None,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> EmbeddingBundle:
    """保存 npy 和 embedding_meta.json，并校验数量和维度。"""
    settings = get_settings()
    output_dir = output_dir or default_embedding_output_dir()
    project_root = project_root or settings.project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    product_ids = list(meta.get("product_ids", []))
    embedding_dim = int(meta.get("embedding_dim", 0))
    generated = dict(meta.get("generated", {}))
    product_id_to_index = dict(meta.get("product_id_to_index", {}))
    image_path = output_dir / IMAGE_EMBEDDING_FILE
    text_path = output_dir / TEXT_EMBEDDING_FILE
    meta_path = output_dir / EMBEDDING_META_FILE

    targets = [meta_path]
    if generated.get("image"):
        targets.append(image_path)
    if generated.get("text"):
        targets.append(text_path)
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"embedding 输出已存在，请使用 --overwrite 覆盖: {names}")

    _validate_product_mapping(product_ids, product_id_to_index)
    _validate_embeddings(product_ids, embedding_dim, image_embeddings, text_embeddings, generated)

    normalized_meta = dict(meta)
    normalized_meta["output_dir"] = to_project_relative(output_dir, project_root)
    normalized_meta["image_embedding_file"] = (
        to_project_relative(image_path, project_root) if generated.get("image") else None
    )
    normalized_meta["text_embedding_file"] = (
        to_project_relative(text_path, project_root) if generated.get("text") else None
    )
    normalized_meta["meta_file"] = to_project_relative(meta_path, project_root)

    if generated.get("image") and image_embeddings is not None:
        np.save(image_path, image_embeddings.astype(np.float32, copy=False))
    if generated.get("text") and text_embeddings is not None:
        np.save(text_path, text_embeddings.astype(np.float32, copy=False))

    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(normalized_meta, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return EmbeddingBundle(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        meta=normalized_meta,
    )


def load_embedding_bundle(output_dir: Path | None = None) -> EmbeddingBundle:
    """加载 embedding_meta.json 和已生成的 npy 文件。"""
    output_dir = output_dir or default_embedding_output_dir()
    meta_path = output_dir / EMBEDDING_META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"embedding_meta.json 不存在: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    generated = dict(meta.get("generated", {}))
    image_embeddings = np.load(output_dir / IMAGE_EMBEDDING_FILE) if generated.get("image") else None
    text_embeddings = np.load(output_dir / TEXT_EMBEDDING_FILE) if generated.get("text") else None

    _validate_embeddings(
        list(meta.get("product_ids", [])),
        int(meta.get("embedding_dim", 0)),
        image_embeddings,
        text_embeddings,
        generated,
    )
    _validate_product_mapping(
        list(meta.get("product_ids", [])),
        dict(meta.get("product_id_to_index", {})),
    )
    return EmbeddingBundle(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        meta=meta,
    )


def load_embeddings_placeholder() -> dict[str, list[float]]:
    """兼容旧调用：真实读取请使用 load_embedding_bundle。"""
    return {}


def _validate_embeddings(
    product_ids: list[str],
    embedding_dim: int,
    image_embeddings: np.ndarray | None,
    text_embeddings: np.ndarray | None,
    generated: dict[str, bool],
) -> None:
    """校验 embedding 行数、维度和 product_id 映射一致。"""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim 必须大于 0。")

    expected_count = len(product_ids)
    if generated.get("image"):
        _validate_single_array("image_embeddings", image_embeddings, expected_count, embedding_dim)
    if generated.get("text"):
        _validate_single_array("text_embeddings", text_embeddings, expected_count, embedding_dim)


def _validate_product_mapping(product_ids: list[str], product_id_to_index: dict[str, int]) -> None:
    """校验 product_id 到 embedding 行号的映射关系。"""
    expected = {product_id: index for index, product_id in enumerate(product_ids)}
    if product_id_to_index != expected:
        raise ValueError("product_id_to_index 与 product_ids 顺序不一致。")


def _validate_single_array(
    name: str,
    embeddings: np.ndarray | None,
    expected_count: int,
    embedding_dim: int,
) -> None:
    """校验单个 embedding 数组的 shape 和 dtype。"""
    if embeddings is None:
        raise ValueError(f"{name} 标记为生成，但数组为空。")
    if embeddings.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组。")
    if embeddings.shape[0] != expected_count:
        raise ValueError(f"{name} 行数与 product_ids 不一致。")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"{name} 维度与 embedding_dim 不一致。")
    if embeddings.dtype != np.float32:
        raise ValueError(f"{name} dtype 必须是 float32。")
