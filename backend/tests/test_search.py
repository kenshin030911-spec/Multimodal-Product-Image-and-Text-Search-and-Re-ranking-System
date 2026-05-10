"""搜索 API 测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import write_products
from backend.app.embedding.text_encoder import DummyTextEncoder
from backend.app.index.index_builder import l2_normalize_matrix
from backend.app.index.index_store import save_index_bundle
from backend.app.main import create_app
from backend.app.api import routes_search
from backend.app.reranker import pairwise_rerank_service
from backend.app.reranker.pairwise_rerank_service import PairwiseRerankerBundle
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.schemas.product import ProductItem
from backend.app.training.feature_exporter import FEATURE_NAMES


def test_app_can_import_and_health_is_accessible() -> None:
    """应用可以 import，健康检查可以访问。"""
    client = TestClient(create_app())
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_loaded"] is False
    assert data["reranker_loaded"] is False
    assert data["placeholder"] is True


def test_text_search_returns_real_vector_results(tmp_path: Path, monkeypatch) -> None:
    """文本搜索 API 接入离线 text-to-image service。"""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=_vectors_for_query(query_vector),
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 2, "use_rerank": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "text"
    assert data["query"] == query
    assert data["top_k"] == 2
    assert data["use_rerank"] is False
    assert data["reranker_type"] == "none"
    assert data["reranker_message"] == data["message"]
    assert data["placeholder"] is False
    assert "reranker 未启用" in data["message"]
    assert [result["product_id"] for result in data["results"]] == ["p1", "p2"]
    first = data["results"][0]
    assert first["title"] == "Product p1"
    assert first["image_path"] == "data/raw/images/p1.jpg"
    assert first["image_url"] == "/static/images/p1.jpg"
    assert ":\\" not in first["image_url"]
    assert first["article_type"] == "Shirts"
    assert first["base_colour"] == "Black"
    assert first["recall_rank"] == 1
    assert first["final_rank"] == 1
    assert first["recall_score"] == first["rerank_score"]
    assert first["freshness_score"] == 0.7
    assert not Path(first["image_path"]).is_absolute()


def test_text_search_can_use_rule_reranker(tmp_path: Path, monkeypatch) -> None:
    """use_rerank=true 时，API 使用规则 reranker 并保留原召回 rank。"""
    query = "black casual shirts men topwear"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=np.vstack(
            [
                _vector_with_target_cosine(query_vector, 0.70),
                _vector_with_target_cosine(query_vector, 0.64),
                _vector_with_target_cosine(query_vector, -0.10),
            ]
        ).astype(np.float32),
        products=[
            _product(
                "p1",
                title="Red Shoes",
                article_type="Shoes",
                base_colour="Red",
                gender="Women",
                usage="Sports",
                sub_category="Footwear",
            ),
            _product(
                "p2",
                title="Black Casual Shirts",
                article_type="Shirts",
                base_colour="Black",
                gender="Men",
                usage="Casual",
                sub_category="Topwear",
            ),
            _product("p3", title="Other Item", article_type="Accessories"),
        ],
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 2, "use_rerank": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["use_rerank"] is True
    assert data["reranker_type"] == "rule"
    assert data["reranker_message"] == data["message"]
    assert "已启用规则 reranker baseline" in data["message"]
    assert [result["product_id"] for result in data["results"]] == ["p2", "p1"]
    assert data["results"][0]["recall_rank"] == 2
    assert data["results"][0]["final_rank"] == 1
    assert data["results"][0]["rerank_score"] > data["results"][0]["recall_score"]
    assert data["results"][1]["recall_rank"] == 1
    assert data["results"][1]["final_rank"] == 2


def test_image_search_returns_real_vector_results(tmp_path: Path, monkeypatch) -> None:
    """图片搜索 API 保存临时文件并接入 image-to-image service。"""
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=l2_normalize_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        ),
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/image",
        data={"top_k": "2", "use_rerank": "true"},
        files={"file": ("query.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "image"
    assert data["query"] == "query.png"
    assert data["reranker_type"] == "rule"
    assert data["placeholder"] is False
    assert "已启用规则 reranker baseline" in data["message"]
    assert len(data["results"]) == 2
    assert data["results"][0]["recall_rank"] == 1
    assert data["results"][0]["final_rank"] == 1
    assert all(not Path(result["image_path"]).is_absolute() for result in data["results"])
    assert list((tmp_path / "uploads" / "tmp").glob("*")) == []


def test_static_image_file_is_served(tmp_path: Path, monkeypatch) -> None:
    """data/raw/images 下的图片通过 /static/images 访问。"""
    _write_search_assets(tmp_path=tmp_path, product_ids=["p1"])
    client = _make_client(tmp_path, monkeypatch)

    response = client.get("/static/images/p1.jpg")

    assert response.status_code == 200
    assert response.content == b"fake-image-p1"


def test_static_image_missing_returns_404(tmp_path: Path, monkeypatch) -> None:
    """不存在的静态图片返回 404。"""
    _write_search_assets(tmp_path=tmp_path, product_ids=["p1"])
    client = _make_client(tmp_path, monkeypatch)

    response = client.get("/static/images/not-found.jpg")

    assert response.status_code == 404


def test_static_images_do_not_allow_path_traversal(tmp_path: Path, monkeypatch) -> None:
    """路径穿越不能通过 /static/images 访问 raw_images_dir 之外的文件。"""
    _write_search_assets(tmp_path=tmp_path, product_ids=["p1"])
    client = _make_client(tmp_path, monkeypatch)

    plain_response = client.get("/static/images/../processed/products.jsonl")
    encoded_response = client.get("/static/images/%2E%2E/processed/products.jsonl")

    assert plain_response.status_code == 404
    assert encoded_response.status_code == 404


def test_search_image_url_ignores_traversal_image_path(tmp_path: Path, monkeypatch) -> None:
    """商品 image_path 包含 .. 时不生成 image_url，也不影响搜索响应。"""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
        products=[
            _product(
                "p1",
                image_path="data/raw/images/../processed/products.jsonl",
            )
        ],
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 1, "use_rerank": False},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["image_path"] == "data/raw/images/../processed/products.jsonl"
    assert result["image_url"] is None


def test_uploads_tmp_is_not_served_from_static_images(tmp_path: Path, monkeypatch) -> None:
    """uploads/tmp 不挂载到 /static/images。"""
    uploads_tmp_dir = tmp_path / "uploads" / "tmp"
    uploads_tmp_dir.mkdir(parents=True)
    (uploads_tmp_dir / "query.jpg").write_bytes(b"temporary-upload")
    _write_search_assets(tmp_path=tmp_path, product_ids=["p1"])
    client = _make_client(tmp_path, monkeypatch)

    response = client.get("/static/images/query.jpg")

    assert response.status_code == 404


def test_image_search_excludes_product_id(tmp_path: Path, monkeypatch) -> None:
    """图片搜索 API 支持 exclude_product_id。"""
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2", "p3"],
        image_index=l2_normalize_matrix(np.eye(3, 8, dtype=np.float32)),
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/image",
        data={"top_k": "3", "use_rerank": "false", "exclude_product_id": "p1"},
        files={"file": ("query.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    product_ids = [result["product_id"] for result in response.json()["results"]]
    assert "p1" not in product_ids
    assert set(product_ids) == {"p2", "p3"}


def test_text_search_explicit_none_overrides_use_rerank_true(tmp_path: Path, monkeypatch) -> None:
    """reranker_type=none takes precedence over legacy use_rerank=true."""
    query = "black casual shirts men topwear"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2"],
        image_index=np.vstack(
            [
                _vector_with_target_cosine(query_vector, 0.70),
                _vector_with_target_cosine(query_vector, 0.64),
            ]
        ).astype(np.float32),
        products=[
            _product("p1", title="Red Shoes", article_type="Shoes", base_colour="Red"),
            _product("p2", title="Black Casual Shirts"),
        ],
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={
            "query": query,
            "top_k": 2,
            "use_rerank": True,
            "reranker_type": "none",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["use_rerank"] is False
    assert data["reranker_type"] == "none"
    assert [result["product_id"] for result in data["results"]] == ["p1", "p2"]
    assert data["results"][0]["recall_score"] == data["results"][0]["rerank_score"]


def test_text_search_explicit_trained_reranker_changes_rank(tmp_path: Path, monkeypatch) -> None:
    """reranker_type=trained calls the trained service and returns trained ranks."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2"],
        image_index=np.vstack(
            [
                _vector_with_target_cosine(query_vector, 0.70),
                _vector_with_target_cosine(query_vector, 0.64),
            ]
        ).astype(np.float32),
    )
    called = {"trained": False}

    def fake_trained_rerank(retrieval_response, query_text):
        called["trained"] = True
        assert query_text == query
        ranked = [
            replace(
                retrieval_response.results[1],
                rank=1,
                final_rank=1,
                rerank_score=0.99,
            ),
            replace(
                retrieval_response.results[0],
                rank=2,
                final_rank=2,
                rerank_score=0.10,
            ),
        ]
        return replace(retrieval_response, results=ranked)

    monkeypatch.setattr(
        routes_search,
        "rerank_retrieval_response_with_trained_model",
        fake_trained_rerank,
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 2, "reranker_type": "trained"},
    )

    assert response.status_code == 200
    data = response.json()
    assert called["trained"] is True
    assert data["use_rerank"] is True
    assert data["reranker_type"] == "trained"
    assert "experimental trained reranker" in data["reranker_message"]
    assert [result["product_id"] for result in data["results"]] == ["p2", "p1"]
    assert data["results"][0]["recall_rank"] == 2
    assert data["results"][0]["final_rank"] == 1
    assert data["results"][0]["recall_score"] != data["results"][0]["rerank_score"]


def test_pairwise_service_calls_score_items_and_preserves_recall(monkeypatch) -> None:
    """Pairwise service scores candidates and keeps recall fields intact."""
    query = "black shirt"
    response = RetrievalResponse(
        query_type="text",
        query=query,
        top_k=2,
        results=[
            _retrieval_result("p1", recall_rank=1, recall_score=0.70),
            _retrieval_result("p2", recall_rank=2, recall_score=0.64),
        ],
        missing_product_ids=[],
    )

    class FakePairwiseModel:
        def __init__(self) -> None:
            self.seen_rows = None

        def score_items(self, x_rows):
            self.seen_rows = x_rows
            return [0.10, 0.99]

    model = FakePairwiseModel()

    monkeypatch.setattr(
        pairwise_rerank_service,
        "get_pairwise_reranker_bundle",
        lambda: PairwiseRerankerBundle(
            model=model,
            meta={"feature_names": list(FEATURE_NAMES)},
            feature_names=FEATURE_NAMES,
        ),
    )

    reranked = pairwise_rerank_service.rerank_retrieval_response_with_pairwise_model(
        response,
        query_text=query,
    )

    assert model.seen_rows is not None
    assert len(model.seen_rows) == 2
    assert [result.product_id for result in reranked.results] == ["p2", "p1"]
    assert reranked.results[0].recall_rank == 2
    assert reranked.results[0].score == 0.64
    assert reranked.results[0].rerank_score == 0.99
    assert reranked.results[0].final_rank == 1
    assert reranked.results[1].recall_rank == 1


def test_text_search_explicit_pairwise_reranker_changes_rank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """reranker_type=pairwise calls the pairwise service and returns pairwise ranks."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1", "p2"],
        image_index=np.vstack(
            [
                _vector_with_target_cosine(query_vector, 0.70),
                _vector_with_target_cosine(query_vector, 0.64),
            ]
        ).astype(np.float32),
    )
    called = {"pairwise": False}

    def fake_pairwise_rerank(retrieval_response, query_text):
        called["pairwise"] = True
        assert query_text == query
        ranked = [
            replace(
                retrieval_response.results[1],
                rank=1,
                final_rank=1,
                rerank_score=1.25,
            ),
            replace(
                retrieval_response.results[0],
                rank=2,
                final_rank=2,
                rerank_score=-0.10,
            ),
        ]
        return replace(retrieval_response, results=ranked)

    monkeypatch.setattr(
        routes_search,
        "rerank_retrieval_response_with_pairwise_model",
        fake_pairwise_rerank,
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={
            "query": query,
            "top_k": 2,
            "use_rerank": False,
            "reranker_type": "pairwise",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert called["pairwise"] is True
    assert data["use_rerank"] is True
    assert data["reranker_type"] == "pairwise"
    assert "experimental pairwise reranker" in data["reranker_message"]
    assert "not calibrated probability" in data["reranker_message"]
    assert [result["product_id"] for result in data["results"]] == ["p2", "p1"]
    assert data["results"][0]["recall_rank"] == 2
    assert data["results"][0]["final_rank"] == 1
    assert data["results"][0]["recall_score"] != data["results"][0]["rerank_score"]


def test_text_search_pairwise_overrides_use_rerank_true(tmp_path: Path, monkeypatch) -> None:
    """Explicit pairwise mode takes precedence over legacy use_rerank=true."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
    )

    def fake_pairwise_rerank(retrieval_response, query_text):
        assert query_text == query
        return retrieval_response

    monkeypatch.setattr(
        routes_search,
        "rerank_retrieval_response_with_pairwise_model",
        fake_pairwise_rerank,
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={
            "query": query,
            "top_k": 1,
            "use_rerank": True,
            "reranker_type": "pairwise",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["use_rerank"] is True
    assert data["reranker_type"] == "pairwise"


def test_text_search_trained_missing_model_returns_503(tmp_path: Path, monkeypatch) -> None:
    """Missing trained model artifacts return a clear 503 error."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 1, "reranker_type": "trained"},
    )

    assert response.status_code == 503
    assert "train_reranker.py" in response.json()["detail"]


def test_text_search_trained_feature_mismatch_returns_503(tmp_path: Path, monkeypatch) -> None:
    """Trained service configuration errors are returned as clear 503 responses."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
    )

    def fake_trained_rerank(_retrieval_response, query_text):
        assert query_text == query
        raise RuntimeError("trained reranker feature_names mismatch")

    monkeypatch.setattr(
        routes_search,
        "rerank_retrieval_response_with_trained_model",
        fake_trained_rerank,
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 1, "reranker_type": "trained"},
    )

    assert response.status_code == 503
    assert "feature_names mismatch" in response.json()["detail"]


def test_text_search_pairwise_missing_model_returns_503(tmp_path: Path, monkeypatch) -> None:
    """Missing pairwise model artifacts return a clear 503 error."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
    )
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    pairwise_rerank_service.clear_pairwise_reranker_cache()
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 1, "reranker_type": "pairwise"},
    )

    assert response.status_code == 503
    assert "train_pairwise_reranker.py" in response.json()["detail"]


def test_text_search_pairwise_feature_mismatch_returns_503(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pairwise service configuration errors are returned as clear 503 responses."""
    query = "black shirt"
    query_vector = DummyTextEncoder().encode_batch([query])[0]
    _write_search_assets(
        tmp_path=tmp_path,
        product_ids=["p1"],
        image_index=l2_normalize_matrix(query_vector.reshape(1, -1)),
    )

    def fake_pairwise_rerank(_retrieval_response, query_text):
        assert query_text == query
        raise RuntimeError(
            "pairwise reranker feature_names mismatch; "
            "请先运行 python backend/scripts/train_pairwise_reranker.py。"
        )

    monkeypatch.setattr(
        routes_search,
        "rerank_retrieval_response_with_pairwise_model",
        fake_pairwise_rerank,
    )
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": query, "top_k": 1, "reranker_type": "pairwise"},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "feature_names mismatch" in detail
    assert "train_pairwise_reranker.py" in detail


def test_image_search_rejects_trained_reranker_type(tmp_path: Path, monkeypatch) -> None:
    """Image search does not silently downgrade trained reranker requests."""
    _write_search_assets(tmp_path=tmp_path, product_ids=["p1"])
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/image",
        data={"top_k": "1", "reranker_type": "trained"},
        files={"file": ("query.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 400
    assert "text search only" in response.json()["detail"]


def test_image_search_rejects_pairwise_reranker_type(tmp_path: Path, monkeypatch) -> None:
    """Image search does not silently downgrade pairwise reranker requests."""
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/image",
        data={"top_k": "1", "reranker_type": "pairwise"},
        files={"file": ("query.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 400
    assert "pairwise reranker currently supports text search only" in response.json()["detail"]


def test_text_search_rejects_invalid_top_k(tmp_path: Path, monkeypatch) -> None:
    """非法 top_k 由 FastAPI/Pydantic 返回 422。"""
    _write_search_assets(tmp_path=tmp_path)
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": "black shirt", "top_k": 0, "use_rerank": True},
    )

    assert response.status_code == 422


def test_image_search_rejects_invalid_content_type(tmp_path: Path, monkeypatch) -> None:
    """上传非法文件类型返回 400。"""
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/image",
        data={"top_k": "5", "use_rerank": "true"},
        files={"file": ("query.txt", b"not-image", "text/plain")},
    )

    assert response.status_code == 400
    assert "不支持的图片类型" in response.json()["detail"]


def test_image_search_rejects_oversized_file(tmp_path: Path, monkeypatch) -> None:
    """上传超过大小限制返回 413。"""
    client = _make_client(tmp_path, monkeypatch, max_upload_size_mb="0")

    response = client.post(
        "/api/search/image",
        data={"top_k": "5", "use_rerank": "true"},
        files={"file": ("query.png", b"x", "image/png")},
    )

    assert response.status_code == 413


def test_search_returns_controlled_error_when_index_missing(tmp_path: Path, monkeypatch) -> None:
    """index 缺失时返回受控错误，不暴露堆栈。"""
    _write_products(tmp_path, [_product("p1")])
    client = _make_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/search/text",
        json={"query": "black shirt", "top_k": 5, "use_rerank": False},
    )

    assert response.status_code == 503
    assert "搜索依赖文件不存在" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_dataset_status_returns_placeholder(tmp_path: Path, monkeypatch) -> None:
    """未准备数据集时，状态接口返回非 placeholder 的基础状态。"""
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    client = TestClient(create_app())

    try:
        response = client.get("/api/dataset/status")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is False
    assert data["placeholder"] is False


def _make_client(
    tmp_path: Path,
    monkeypatch,
    max_upload_size_mb: str | None = None,
) -> TestClient:
    """创建使用 tmp_path 配置的新 TestClient。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UPLOADS_TMP_DIR", str(tmp_path / "uploads" / "tmp"))
    if max_upload_size_mb is not None:
        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", max_upload_size_mb)
    get_settings.cache_clear()
    return TestClient(create_app())


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


def _vector_with_target_cosine(query_vector: np.ndarray, target_cosine: float) -> np.ndarray:
    """构造与 query_vector 具有指定 cosine 的单位向量。"""
    query = query_vector.astype(np.float32, copy=False)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise ValueError("query_vector 不能是零向量。")
    query_unit = query / query_norm

    basis = np.zeros_like(query_unit)
    basis[0] = 1.0
    orthogonal = basis - float(np.dot(basis, query_unit)) * query_unit
    if float(np.linalg.norm(orthogonal)) < 1e-6:
        basis = np.zeros_like(query_unit)
        basis[1] = 1.0
        orthogonal = basis - float(np.dot(basis, query_unit)) * query_unit
    orthogonal = orthogonal / float(np.linalg.norm(orthogonal))

    target_cosine = float(np.clip(target_cosine, -1.0, 1.0))
    vector = target_cosine * query_unit + np.sqrt(max(0.0, 1.0 - target_cosine**2)) * orthogonal
    return vector.astype(np.float32, copy=False)


def _write_search_assets(
    tmp_path: Path,
    product_ids: list[str] | None = None,
    image_index: np.ndarray | None = None,
    products: list[ProductItem] | None = None,
) -> None:
    """写入 dummy products 和 index。"""
    product_ids = product_ids or ["p1", "p2", "p3"]
    if image_index is None:
        image_index = l2_normalize_matrix(np.eye(len(product_ids), 8, dtype=np.float32))
    _write_index(tmp_path, product_ids, image_index)
    _write_raw_images(tmp_path, product_ids)
    _write_products(tmp_path, products or [_product(product_id) for product_id in product_ids])


def _write_index(tmp_path: Path, product_ids: list[str], image_index: np.ndarray) -> None:
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


def _write_products(tmp_path: Path, products: list[ProductItem]) -> None:
    """写入测试用 products.jsonl。"""
    write_products(products, tmp_path / "data" / "processed" / "products.jsonl")


def _retrieval_result(
    product_id: str,
    recall_rank: int,
    recall_score: float,
) -> RetrievalResult:
    """创建测试用 RetrievalResult。"""
    return RetrievalResult(
        product_id=product_id,
        title=f"Product {product_id}",
        image_path=f"data/raw/images/{product_id}.jpg",
        article_type="Shirts",
        base_colour="Black",
        gender="Men",
        usage="Casual",
        sub_category="Topwear",
        freshness_score=0.7,
        score=recall_score,
        rank=recall_rank,
        embedding_index=recall_rank - 1,
        recall_rank=recall_rank,
        rerank_score=recall_score,
        final_rank=recall_rank,
    )


def _write_raw_images(tmp_path: Path, product_ids: list[str]) -> None:
    """写入测试用 raw images。"""
    raw_images_dir = tmp_path / "data" / "raw" / "images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    for product_id in product_ids:
        (raw_images_dir / f"{product_id}.jpg").write_bytes(f"fake-image-{product_id}".encode())


def _product(
    product_id: str,
    title: str | None = None,
    article_type: str = "Shirts",
    base_colour: str = "Black",
    gender: str = "Men",
    usage: str = "Casual",
    sub_category: str = "Topwear",
    image_path: str | None = None,
) -> ProductItem:
    """创建测试商品。"""
    return ProductItem(
        product_id=product_id,
        title=title or f"Product {product_id}",
        gender=gender,
        master_category="Apparel",
        sub_category=sub_category,
        article_type=article_type,
        base_colour=base_colour,
        season="Fall",
        year=2011,
        usage=usage,
        image_path=image_path or f"data/raw/images/{product_id}.jpg",
        freshness_score=0.7,
    )
