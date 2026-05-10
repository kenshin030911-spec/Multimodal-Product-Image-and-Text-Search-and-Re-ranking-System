"""Query-level train/valid split helpers for reranker datasets."""

from __future__ import annotations

import random
from collections.abc import Iterable


def split_query_ids(
    query_ids: Iterable[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, str]:
    """Assign each query_id to train or valid with a deterministic shuffle."""
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio 必须在 0 和 1 之间。")

    unique_query_ids = sorted(set(query_ids))
    if not unique_query_ids:
        return {}

    rng = random.Random(seed)
    shuffled = list(unique_query_ids)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        train_count = 1
    else:
        train_count = int(len(shuffled) * train_ratio)
        train_count = max(1, min(train_count, len(shuffled) - 1))

    train_query_ids = set(shuffled[:train_count])
    return {
        query_id: "train" if query_id in train_query_ids else "valid"
        for query_id in unique_query_ids
    }
