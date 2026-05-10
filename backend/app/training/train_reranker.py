"""Compatibility entry points for reranker training."""

from __future__ import annotations

from backend.app.training.reranker_trainer import (
    RerankerTrainingResult,
    train_reranker_model,
)


def train_reranker(**kwargs) -> RerankerTrainingResult:
    """Train the lightweight reranker using the new trainer implementation."""
    return train_reranker_model(**kwargs)


def train_reranker_placeholder() -> dict[str, bool]:
    """Compatibility helper retained for old imports."""
    return {"trained": False, "placeholder": False}
