"""Experimental pairwise reranker service for text search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from backend.app.core.config import get_settings
from backend.app.reranker.feature_builder import build_rerank_feature
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.training.feature_exporter import FEATURE_NAMES, rerank_feature_to_dict
from backend.app.training.pairwise_trainer import load_pairwise_reranker


PAIRWISE_RERANKER_DIR = "reranker_pairwise"
PAIRWISE_RERANKER_MODEL_FILE = "pairwise_reranker.joblib"
PAIRWISE_RERANKER_META_FILE = "pairwise_reranker_meta.json"
TRAIN_PAIRWISE_RERANKER_HINT = (
    "请先运行 python backend/scripts/train_pairwise_reranker.py。"
)


@dataclass(frozen=True)
class PairwiseRerankerBundle:
    """Loaded pairwise reranker model and metadata."""

    model: Any
    meta: dict[str, Any]
    feature_names: tuple[str, ...]


def get_pairwise_reranker_bundle() -> PairwiseRerankerBundle:
    """Load and cache the experimental pairwise reranker bundle."""
    settings = get_settings()
    pairwise_dir = settings.model_dir / PAIRWISE_RERANKER_DIR
    return _load_pairwise_reranker_bundle(str(pairwise_dir.resolve()))


def clear_pairwise_reranker_cache() -> None:
    """Clear the cached pairwise reranker, mainly for tests."""
    _load_pairwise_reranker_bundle.cache_clear()


def rerank_retrieval_response_with_pairwise_model(
    response: RetrievalResponse,
    query_text: str,
) -> RetrievalResponse:
    """Rerank text-search candidates with the pairwise ranking model."""
    if not query_text.strip():
        raise ValueError("pairwise reranker requires non-empty query_text.")
    if not response.results:
        return response

    bundle = get_pairwise_reranker_bundle()
    x_rows = _build_feature_matrix(
        results=response.results,
        query_text=query_text,
        feature_names=bundle.feature_names,
    )
    pairwise_scores = [float(score) for score in bundle.model.score_items(x_rows)]
    scored_results = list(zip(response.results, pairwise_scores))
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
def _load_pairwise_reranker_bundle(pairwise_dir_value: str) -> PairwiseRerankerBundle:
    """Load pairwise artifacts from a model directory with clear errors."""
    pairwise_dir = Path(pairwise_dir_value)
    model_path = pairwise_dir / PAIRWISE_RERANKER_MODEL_FILE
    meta_path = pairwise_dir / PAIRWISE_RERANKER_META_FILE
    try:
        model, meta = load_pairwise_reranker(
            model_path=model_path,
            meta_path=meta_path,
            expected_feature_names=FEATURE_NAMES,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"pairwise reranker model or meta not found: {exc} "
            f"{TRAIN_PAIRWISE_RERANKER_HINT}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"pairwise reranker feature_names mismatch: {exc} "
            f"{TRAIN_PAIRWISE_RERANKER_HINT}"
        ) from exc

    if not hasattr(model, "score_items"):
        raise RuntimeError(
            "pairwise reranker model does not provide score_items(). "
            f"{TRAIN_PAIRWISE_RERANKER_HINT}"
        )

    return PairwiseRerankerBundle(
        model=model,
        meta=meta,
        feature_names=tuple(str(name) for name in meta.get("feature_names", FEATURE_NAMES)),
    )


def _build_feature_matrix(
    results: Sequence[RetrievalResult],
    query_text: str,
    feature_names: Sequence[str],
) -> list[list[float]]:
    """Build feature rows in the exact order used for pairwise training."""
    rows: list[list[float]] = []
    for result in results:
        feature = build_rerank_feature(result, query_text=query_text)
        feature_dict = rerank_feature_to_dict(feature)
        rows.append([float(feature_dict[name]) for name in feature_names])
    return rows
