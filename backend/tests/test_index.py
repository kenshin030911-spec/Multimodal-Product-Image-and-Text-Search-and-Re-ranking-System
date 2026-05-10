"""NumPy 向量索引模块测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.embedding.embedding_store import EMBEDDING_META_FILE, IMAGE_EMBEDDING_FILE
from backend.app.embedding.embedding_store import save_embedding_bundle
from backend.app.index.index_builder import build_index
from backend.app.index.index_store import IMAGE_INDEX_FILE, INDEX_META_FILE, load_index_bundle
from backend.app.index.vector_searcher import VectorSearcher


def test_build_index_saves_normalized_image_index_and_meta(tmp_path: Path) -> None:
    """build_index 会保存归一化 image_index 和相对路径 meta。"""
    embedding_dir = _write_embedding_bundle(
        tmp_path,
        np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    output_dir = tmp_path / "data" / "index"

    result = build_index(
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        overwrite=True,
        project_root=tmp_path,
    )
    bundle = load_index_bundle(output_dir)

    assert result.product_count == 3
    assert result.embedding_dim == 2
    assert bundle.image_index.dtype == np.float32
    np.testing.assert_allclose(bundle.image_index[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(bundle.image_index[1], [0.0, 0.0], atol=1e-6)
    assert bundle.meta["index_type"] == "numpy_flat"
    assert bundle.meta["metric"] == "cosine"
    assert bundle.meta["embedding_source"] == "data/embeddings"
    assert bundle.meta["embedding_meta_file"] == "data/embeddings/embedding_meta.json"
    assert bundle.meta["image_index_file"] == "data/index/image_index.npy"
    assert bundle.meta["embedding_dim"] == 2
    assert bundle.meta["product_count"] == 3
    assert bundle.meta["product_ids"] == ["p1", "p2", "p3"]
    assert bundle.meta["product_id_to_index"] == {"p1": 0, "p2": 1, "p3": 2}
    assert bundle.meta["source_encoder_name"] == "dummy"
    assert bundle.meta["source_model_name"] is None
    assert bundle.meta["source_is_dummy"] is True
    assert bundle.meta["source_normalize_embeddings"] is False
    assert "created_at" in bundle.meta
    assert not Path(bundle.meta["embedding_source"]).is_absolute()
    assert not Path(bundle.meta["embedding_meta_file"]).is_absolute()
    assert not Path(bundle.meta["image_index_file"]).is_absolute()


def test_vector_search_top_k_mapping_and_negative_score(tmp_path: Path) -> None:
    """searcher 返回按 cosine 降序排列的 product_id、index 和原始分数。"""
    output_dir = _build_small_index(tmp_path)
    searcher = VectorSearcher(load_index_bundle(output_dir))

    results = searcher.search([1.0, 0.0], top_k=3)

    assert [result.product_id for result in results] == ["p1", "p2", "p3"]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.embedding_index for result in results] == [0, 1, 2]
    np.testing.assert_allclose([result.score for result in results], [1.0, 0.0, -1.0])
    assert results[-1].score < 0.0


def test_vector_search_excludes_product_id_and_embedding_index(tmp_path: Path) -> None:
    """图搜图场景可以按 product_id 或 embedding_index 排除自身。"""
    output_dir = _build_small_index(tmp_path)
    searcher = VectorSearcher(load_index_bundle(output_dir))

    by_product_id = searcher.search([1.0, 0.0], top_k=3, exclude_product_id="p1")
    by_embedding_index = searcher.search([1.0, 0.0], top_k=3, exclude_embedding_index=0)

    assert [result.product_id for result in by_product_id] == ["p2", "p3"]
    assert [result.product_id for result in by_embedding_index] == ["p2", "p3"]


def test_build_index_rejects_shape_mismatch(tmp_path: Path) -> None:
    """embedding 行数与 product_ids 不一致时给出清晰错误。"""
    embedding_dir = tmp_path / "data" / "embeddings"
    embedding_dir.mkdir(parents=True)
    np.save(
        embedding_dir / IMAGE_EMBEDDING_FILE,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    _write_embedding_meta(
        embedding_dir,
        product_ids=["p1", "p2", "p3"],
        embedding_dim=2,
    )

    with pytest.raises(ValueError, match="行数|shape"):
        build_index(
            embedding_dir=embedding_dir,
            output_dir=tmp_path / "data" / "index",
            overwrite=True,
            project_root=tmp_path,
        )


def test_build_index_overwrite_guard(tmp_path: Path) -> None:
    """默认不覆盖已有 index 输出，显式 overwrite 才覆盖。"""
    embedding_dir = _write_embedding_bundle(
        tmp_path,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    output_dir = tmp_path / "data" / "index"

    build_index(
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        overwrite=True,
        project_root=tmp_path,
    )

    with pytest.raises(FileExistsError):
        build_index(
            embedding_dir=embedding_dir,
            output_dir=output_dir,
            overwrite=False,
            project_root=tmp_path,
        )


def test_build_index_rejects_unsupported_metric(tmp_path: Path) -> None:
    """第一版只支持 cosine metric。"""
    embedding_dir = _write_embedding_bundle(
        tmp_path,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="cosine"):
        build_index(
            embedding_dir=embedding_dir,
            output_dir=tmp_path / "data" / "index",
            metric="inner_product",
            overwrite=True,
            project_root=tmp_path,
        )


def test_vector_search_validates_exclude_and_query_shape(tmp_path: Path) -> None:
    """searcher 会校验 query 维度和排除参数。"""
    output_dir = _build_small_index(tmp_path)
    searcher = VectorSearcher(load_index_bundle(output_dir))

    with pytest.raises(ValueError, match="exclude_product_id"):
        searcher.search([1.0, 0.0], exclude_product_id="missing")
    with pytest.raises(ValueError, match="exclude_embedding_index"):
        searcher.search([1.0, 0.0], exclude_embedding_index=99)
    with pytest.raises(ValueError, match="维度"):
        searcher.search([1.0, 0.0, 0.0])


def _build_small_index(tmp_path: Path) -> Path:
    """创建三条二维向量并构建 index。"""
    embedding_dir = _write_embedding_bundle(
        tmp_path,
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )
    output_dir = tmp_path / "data" / "index"
    build_index(
        embedding_dir=embedding_dir,
        output_dir=output_dir,
        overwrite=True,
        project_root=tmp_path,
    )
    return output_dir


def _write_embedding_bundle(tmp_path: Path, image_embeddings: np.ndarray) -> Path:
    """写入测试用 embedding bundle。"""
    embedding_dir = tmp_path / "data" / "embeddings"
    product_ids = [f"p{index}" for index in range(1, image_embeddings.shape[0] + 1)]
    meta = _embedding_meta(product_ids=product_ids, embedding_dim=image_embeddings.shape[1])
    save_embedding_bundle(
        output_dir=embedding_dir,
        meta=meta,
        image_embeddings=image_embeddings.astype(np.float32, copy=False),
        overwrite=True,
        project_root=tmp_path,
    )
    return embedding_dir


def _write_embedding_meta(
    embedding_dir: Path,
    product_ids: list[str],
    embedding_dim: int,
) -> None:
    """直接写入 meta，用于构造坏 shape 场景。"""
    with (embedding_dir / EMBEDDING_META_FILE).open("w", encoding="utf-8") as file:
        json.dump(_embedding_meta(product_ids=product_ids, embedding_dim=embedding_dim), file)
        file.write("\n")


def _embedding_meta(product_ids: list[str], embedding_dim: int) -> dict[str, object]:
    """创建最小 embedding_meta。"""
    return {
        "encoder_name": "dummy",
        "encoder_version": "dummy-v1",
        "embedding_dim": embedding_dim,
        "is_dummy": True,
        "framework": "dummy",
        "model_name": None,
        "normalize_embeddings": False,
        "product_count": len(product_ids),
        "product_ids": product_ids,
        "product_id_to_index": {
            product_id: index for index, product_id in enumerate(product_ids)
        },
        "generated": {"image": True, "text": False},
        "failed_images": [],
        "failed_texts": [],
        "skipped_products": [],
        "batch_size": 2,
        "limit": None,
    }
