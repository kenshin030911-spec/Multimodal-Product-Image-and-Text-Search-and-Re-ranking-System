"""离线检索服务层测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from backend.app.data.dataset_loader import write_products
from backend.app.embedding.image_encoder import DummyImageEncoder
from backend.app.embedding.text_encoder import DummyTextEncoder
from backend.app.index.index_builder import l2_normalize_matrix
from backend.app.index.index_store import save_index_bundle
from backend.app.retrieval.image_search import search_image_to_image
from backend.app.retrieval.text_search import search_text_to_image
from backend.app.schemas.product import ProductItem


def test_text_to_image_returns_products_and_relative_paths(tmp_path: Path) -> None:
    """text-to-image 会编码 query、检索 image_index，并回查商品。"""
    query_text = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query_text])[0]
    index_dir = _write_index(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=_vectors_for_query(query_vector),
    )
    products_path = _write_products(
        tmp_path,
        [
            _product("p1", title="Black Shirt", image_path=str(tmp_path / "data/raw/images/p1.jpg")),
            _product("p2", title="White Shoes", article_type="Shoes"),
            _product("p3", title="Opposite Item"),
        ],
    )

    response = search_text_to_image(
        query_text=query_text,
        top_k=2,
        index_dir=index_dir,
        products_path=products_path,
        project_root=tmp_path,
    )

    assert response.query_type == "text"
    assert response.query == query_text
    assert response.top_k == 2
    assert response.missing_product_ids == []
    assert [result.product_id for result in response.results] == ["p1", "p2"]
    assert response.results[0].title == "Black Shirt"
    assert response.results[0].article_type == "Shirts"
    assert response.results[0].base_colour == "Black"
    assert response.results[0].recall_rank == 1
    assert response.results[0].final_rank == 1
    assert response.results[0].embedding_index == 0
    assert response.results[0].image_path == "data/raw/images/p1.jpg"
    assert not Path(response.results[0].image_path).is_absolute()


def test_image_to_image_supports_exclude_product_id(tmp_path: Path) -> None:
    """image-to-image 支持按 product_id 排除自身。"""
    query_image = tmp_path / "query.jpg"
    query_image.write_bytes(b"fake-image")
    query_vector = DummyImageEncoder().encode_batch([query_image.resolve()])[0]
    index_dir = _write_index(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=_vectors_for_query(query_vector),
    )
    products_path = _write_products(
        tmp_path,
        [_product("p1"), _product("p2"), _product("p3")],
    )

    response = search_image_to_image(
        query_image_path=query_image,
        top_k=3,
        index_dir=index_dir,
        products_path=products_path,
        exclude_product_id="p1",
        project_root=tmp_path,
    )

    assert response.query_type == "image"
    assert response.query == "query.jpg"
    assert [result.product_id for result in response.results] == ["p2", "p3"]
    assert response.results[0].score == 0.0
    assert response.results[1].score < 0.0


def test_image_to_image_supports_exclude_embedding_index(tmp_path: Path) -> None:
    """image-to-image 支持按 embedding_index 排除自身。"""
    query_image = tmp_path / "query.jpg"
    query_image.write_bytes(b"fake-image")
    query_vector = DummyImageEncoder().encode_batch([query_image.resolve()])[0]
    index_dir = _write_index(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=_vectors_for_query(query_vector),
    )
    products_path = _write_products(
        tmp_path,
        [_product("p1"), _product("p2"), _product("p3")],
    )

    response = search_image_to_image(
        query_image_path=query_image,
        top_k=3,
        index_dir=index_dir,
        products_path=products_path,
        exclude_embedding_index=0,
        project_root=tmp_path,
    )

    assert [result.product_id for result in response.results] == ["p2", "p3"]


def test_retrieval_records_missing_product_ids(tmp_path: Path) -> None:
    """index 命中但 products.jsonl 缺失时，记录 missing_product_ids 并跳过。"""
    query_text = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query_text])[0]
    index_dir = _write_index(
        tmp_path=tmp_path,
        product_ids=["missing", "p1"],
        image_index=l2_normalize_matrix(
            np.vstack([query_vector, np.zeros_like(query_vector)]).astype(np.float32)
        ),
    )
    products_path = _write_products(tmp_path, [_product("p1")])

    response = search_text_to_image(
        query_text=query_text,
        top_k=2,
        index_dir=index_dir,
        products_path=products_path,
        project_root=tmp_path,
    )

    assert response.missing_product_ids == ["missing"]
    assert [result.product_id for result in response.results] == ["p1"]
    assert response.results[0].recall_rank == 2
    assert response.results[0].final_rank == 2


def test_search_demo_help() -> None:
    """search_demo.py --help 可用，且不会加载真实模型。"""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/search_demo.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--query-text" in completed.stdout
    assert "--query-image" in completed.stdout


def _vectors_for_query(query_vector: np.ndarray) -> np.ndarray:
    """构造与 query 同向、零向量、反向的归一化 index。"""
    return l2_normalize_matrix(
        np.vstack(
            [
                query_vector,
                np.zeros_like(query_vector),
                -query_vector,
            ]
        ).astype(np.float32)
    )


def _write_index(tmp_path: Path, product_ids: list[str], image_index: np.ndarray) -> Path:
    """写入 dummy index bundle。"""
    index_dir = tmp_path / "data" / "index"
    meta = {
        "index_type": "numpy_flat",
        "metric": "cosine",
        "embedding_source": "data/embeddings",
        "embedding_meta_file": "data/embeddings/embedding_meta.json",
        "image_index_file": "data/index/image_index.npy",
        "embedding_dim": image_index.shape[1],
        "product_count": len(product_ids),
        "product_ids": product_ids,
        "product_id_to_index": {
            product_id: index for index, product_id in enumerate(product_ids)
        },
        "source_encoder_name": "dummy",
        "source_model_name": None,
        "source_is_dummy": True,
        "source_normalize_embeddings": False,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    save_index_bundle(
        output_dir=index_dir,
        meta=meta,
        image_index=image_index.astype(np.float32, copy=False),
        overwrite=True,
        project_root=tmp_path,
    )
    return index_dir


def _write_products(tmp_path: Path, products: list[ProductItem]) -> Path:
    """写入测试用 products.jsonl。"""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    write_products(products, products_path)
    return products_path


def _product(
    product_id: str,
    title: str | None = None,
    article_type: str = "Shirts",
    image_path: str | None = None,
) -> ProductItem:
    """创建测试商品。"""
    return ProductItem(
        product_id=product_id,
        title=title or f"Product {product_id}",
        gender="Men",
        master_category="Apparel",
        sub_category="Topwear",
        article_type=article_type,
        base_colour="Black",
        season="Fall",
        year=2011,
        usage="Casual",
        image_path=image_path or f"data/raw/images/{product_id}.jpg",
        freshness_score=0.5,
    )
