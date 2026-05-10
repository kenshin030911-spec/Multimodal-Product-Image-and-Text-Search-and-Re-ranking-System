"""Experimental trained reranker service for text search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from backend.app.core.config import get_settings
from backend.app.reranker.feature_builder import build_rerank_feature
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.training.feature_exporter import FEATURE_NAMES, rerank_feature_to_dict
from backend.app.training.model_store import load_trained_reranker


TRAINED_RERANKER_MODEL_FILE = "trained_reranker.joblib"
TRAINED_RERANKER_META_FILE = "trained_reranker_meta.json"
TRAIN_RERANKER_HINT = "请先运行 python backend/scripts/train_reranker.py。"


@dataclass(frozen=True)
class TrainedRerankerBundle:
    """Loaded trained reranker model and metadata."""

    model: Any
    meta: dict[str, Any]
    feature_names: tuple[str, ...]


def get_trained_reranker_bundle() -> TrainedRerankerBundle:
    """Load and cache the experimental trained reranker bundle."""
    settings = get_settings()
    return _load_trained_reranker_bundle(str(settings.reranker_dir.resolve()))


def clear_trained_reranker_cache() -> None:
    """Clear the cached trained reranker, mainly for tests."""
    _load_trained_reranker_bundle.cache_clear()


def rerank_retrieval_response_with_trained_model(
    response: RetrievalResponse,
    query_text: str,
) -> RetrievalResponse:
    """Rerank text-search candidates with the trained sklearn model."""
    if not query_text.strip():
        raise ValueError("trained reranker requires non-empty query_text.")
    if not response.results:
        return response

    bundle = get_trained_reranker_bundle()
    x_rows = _build_feature_matrix(
        results=response.results,
        query_text=query_text,
        feature_names=bundle.feature_names,
    )
    probabilities = bundle.model.predict_proba(x_rows)
    trained_scores = [float(row[1]) for row in probabilities]
    scored_results = list(zip(response.results, trained_scores))
    scored_results.sort(key=lambda item: (-item[1], item[0].recall_rank))

    reranked_results: list[RetrievalResult] = []
    for final_rank, (result, score) in enumerate(scored_results, start=1):
        reranked_results.append(
            replace(
                result,
                rank=final_rank,
                rerank_score=score,
                final_rank=final_rank,
            )
        )
    return replace(response, results=reranked_results)


@lru_cache(maxsize=1)
def _load_trained_reranker_bundle(reranker_dir_value: str) -> TrainedRerankerBundle:
    """Load trained artifacts from a reranker directory with clear errors."""
    reranker_dir = Path(reranker_dir_value)
    model_path = reranker_dir / TRAINED_RERANKER_MODEL_FILE
    meta_path = reranker_dir / TRAINED_RERANKER_META_FILE
    try:
        model, meta = load_trained_reranker(
            model_path=model_path,
            meta_path=meta_path,
            expected_feature_names=FEATURE_NAMES,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"trained reranker model or meta not found: {exc} {TRAIN_RERANKER_HINT}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"trained reranker feature_names mismatch: {exc} {TRAIN_RERANKER_HINT}"
        ) from exc

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("trained reranker model does not provide predict_proba().")

    return TrainedRerankerBundle(
        model=model,
        meta=meta,
        feature_names=tuple(str(name) for name in meta.get("feature_names", FEATURE_NAMES)),
    )


def _build_feature_matrix(
    results: Sequence[RetrievalResult],
    query_text: str,
    feature_names: Sequence[str],
) -> list[list[float]]:
    """Build feature rows in the exact order used for training."""
    rows: list[list[float]] = []
    for result in results:
        feature = build_rerank_feature(result, query_text=query_text)
        feature_dict = rerank_feature_to_dict(feature)
        rows.append([float(feature_dict[name]) for name in feature_names])
    return rows
