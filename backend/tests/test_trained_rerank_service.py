"""Tests for the experimental trained reranker service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.core.config import get_settings
from backend.app.reranker import trained_rerank_service
from backend.app.reranker.trained_rerank_service import (
    clear_trained_reranker_cache,
    rerank_retrieval_response_with_trained_model,
)
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.training.feature_exporter import FEATURE_NAMES
from backend.app.training.model_store import save_trained_reranker


class FakeModel:
    """Serializable model that scores by color_match."""

    def predict_proba(self, rows):
        color_index = list(FEATURE_NAMES).index("color_match")
        return [[1.0 - float(row[color_index]), float(row[color_index])] for row in rows]


def test_trained_rerank_service_sorts_and_preserves_recall_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Trained service sorts by predict_proba score and keeps recall metadata."""
    _write_fake_model(tmp_path, FakeModel(), list(FEATURE_NAMES))
    _configure_model_dir(tmp_path, monkeypatch)
    response = _response(
        [
            _result("p1", rank=1, score=0.90, base_colour="Red"),
            _result("p2", rank=2, score=0.40, base_colour="Black"),
        ]
    )

    reranked = rerank_retrieval_response_with_trained_model(
        response,
        query_text="black shirt",
    )

    assert [result.product_id for result in reranked.results] == ["p2", "p1"]
    assert reranked.results[0].recall_rank == 2
    assert reranked.results[0].score == pytest.approx(0.40)
    assert reranked.results[0].final_rank == 1
    assert reranked.results[0].rerank_score == pytest.approx(1.0)


def test_trained_rerank_service_missing_model_has_train_hint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Missing artifacts produce a clear train_reranker.py hint."""
    _configure_model_dir(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="train_reranker.py"):
        rerank_retrieval_response_with_trained_model(
            _response([_result("p1")]),
            query_text="black shirt",
        )


def test_trained_rerank_service_feature_mismatch_has_clear_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Feature-name mismatch fails before inference."""
    _write_fake_model(tmp_path, FakeModel(), ["wrong_feature"])
    _configure_model_dir(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="feature_names mismatch"):
        rerank_retrieval_response_with_trained_model(
            _response([_result("p1")]),
            query_text="black shirt",
        )


def _configure_model_dir(tmp_path: Path, monkeypatch) -> None:
    """Point settings at tmp_path model artifacts and clear caches."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    clear_trained_reranker_cache()


def _write_fake_model(tmp_path: Path, model: FakeModel, feature_names: list[str]) -> None:
    """Write fake trained model artifacts."""
    model_dir = tmp_path / "models" / "reranker"
    save_trained_reranker(
        model=model,
        model_path=model_dir / trained_rerank_service.TRAINED_RERANKER_MODEL_FILE,
        meta={
            "model_type": "fake",
            "framework": "test",
            "feature_names": feature_names,
        },
        meta_path=model_dir / trained_rerank_service.TRAINED_RERANKER_META_FILE,
        overwrite=True,
    )


def _response(results: list[RetrievalResult]) -> RetrievalResponse:
    return RetrievalResponse(
        query_type="text",
        query="black shirt",
        top_k=len(results),
        results=results,
        missing_product_ids=[],
    )


def _result(
    product_id: str,
    *,
    rank: int = 1,
    score: float = 0.5,
    base_colour: str = "Black",
) -> RetrievalResult:
    return RetrievalResult(
        product_id=product_id,
        title=f"Product {product_id}",
        image_path=f"data/raw/images/{product_id}.jpg",
        article_type="Shirts",
        base_colour=base_colour,
        gender="Men",
        usage="Casual",
        sub_category="Topwear",
        freshness_score=0.5,
        score=score,
        rank=rank,
        embedding_index=rank - 1,
        recall_rank=rank,
        rerank_score=score,
        final_rank=rank,
    )
