"""Stable feature export helpers for reranker training data."""

from __future__ import annotations

from backend.app.reranker.feature_builder import RerankFeature


FEATURE_NAMES = (
    "recall_score",
    "freshness_score",
    "title_match",
    "article_type_match",
    "color_match",
    "gender_match",
    "usage_match",
    "sub_category_match",
    "text_match_score",
    "metadata_match_score",
)


def rerank_feature_to_dict(feature: RerankFeature) -> dict[str, float]:
    """Export a RerankFeature using a fixed feature order and JSON-safe floats."""
    return {
        feature_name: float(getattr(feature, feature_name, 0.0))
        for feature_name in FEATURE_NAMES
    }
