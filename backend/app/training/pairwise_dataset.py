"""Pairwise ranking dataset helpers."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from backend.app.training.feature_exporter import FEATURE_NAMES


PairSamplingStrategy = Literal["hard", "random"]


@dataclass(frozen=True)
class PairwiseItem:
    """One query-product row loaded from reranker training JSONL."""

    index: int
    query_id: str
    product_id: str
    relevance_grade: int
    recall_rank: int
    recall_score: float
    features: dict[str, float]


@dataclass(frozen=True)
class PairwiseItemDataset:
    """Item-level training split used to build pairwise rows."""

    path: Path
    items: list[PairwiseItem]
    feature_names: list[str]
    x: np.ndarray

    @property
    def item_count(self) -> int:
        """Return item count."""
        return len(self.items)

    @property
    def query_count(self) -> int:
        """Return unique query count."""
        return len(group_items_by_query(self.items))


@dataclass(frozen=True)
class PairwiseRows:
    """Expanded pairwise diff rows and metadata."""

    x_diff: np.ndarray
    y: np.ndarray
    rows: list[dict[str, Any]]
    query_pair_counts: dict[str, int]
    skipped_no_pair_query_count: int

    @property
    def sample_count(self) -> int:
        """Return expanded pairwise row count."""
        return int(self.y.size)


@dataclass(frozen=True)
class PairwiseDataBundle:
    """Train/valid item splits and source metadata."""

    train: PairwiseItemDataset
    valid: PairwiseItemDataset
    dataset_meta: dict[str, Any]
    feature_names: list[str]


@dataclass(frozen=True)
class PreferredPair:
    """One preferred-vs-less-preferred item pair before positive/negative expansion."""

    query_id: str
    preferred: PairwiseItem
    less_preferred: PairwiseItem
    grade_gap: int
    rank_distance: int
    score_distance: float


def load_pairwise_data(
    train_path: Path,
    valid_path: Path,
    dataset_meta_path: Path,
) -> PairwiseDataBundle:
    """Load train/valid item JSONL and resolve feature order."""
    dataset_meta = read_json_object(dataset_meta_path, label="dataset meta")
    feature_names = resolve_feature_names(dataset_meta)
    train = load_pairwise_item_dataset(train_path, feature_names=feature_names)
    valid = load_pairwise_item_dataset(valid_path, feature_names=feature_names)
    return PairwiseDataBundle(
        train=train,
        valid=valid,
        dataset_meta=dataset_meta,
        feature_names=feature_names,
    )


def load_pairwise_item_dataset(path: Path, feature_names: Sequence[str]) -> PairwiseItemDataset:
    """Read one JSONL split and validate required item-level fields."""
    raw_rows = read_jsonl_objects(path)
    if not raw_rows:
        raise ValueError(f"pairwise item split is empty: {path}")

    stable_feature_names = [str(name) for name in feature_names]
    items: list[PairwiseItem] = []
    x_rows: list[list[float]] = []
    for line_number, row in enumerate(raw_rows, start=1):
        item, x_row = sample_to_pairwise_item(
            row,
            feature_names=stable_feature_names,
            index=line_number - 1,
            path=path,
            line_number=line_number,
        )
        items.append(item)
        x_rows.append(x_row)

    return PairwiseItemDataset(
        path=path,
        items=items,
        feature_names=stable_feature_names,
        x=np.asarray(x_rows, dtype=float),
    )


def sample_to_pairwise_item(
    sample: Mapping[str, Any],
    feature_names: Sequence[str],
    index: int,
    path: Path,
    line_number: int,
) -> tuple[PairwiseItem, list[float]]:
    """Validate one training row and convert features to a stable numeric vector."""
    for field_name in (
        "query_id",
        "product_id",
        "relevance_grade",
        "recall_rank",
        "recall_score",
        "features",
    ):
        if field_name not in sample:
            raise ValueError(f"{path}:{line_number} missing required field {field_name}.")

    query_id = str(sample["query_id"])
    product_id = str(sample["product_id"])
    features = sample["features"]
    if not isinstance(features, Mapping):
        raise ValueError(f"sample {query_id}/{product_id} features must be an object.")

    feature_values: list[float] = []
    feature_dict: dict[str, float] = {}
    for feature_name in feature_names:
        if feature_name not in features:
            raise ValueError(
                f"sample {query_id}/{product_id} missing feature {feature_name}",
            )
        try:
            value = float(features[feature_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample {query_id}/{product_id} feature {feature_name} must be numeric.",
            ) from exc
        feature_values.append(value)
        feature_dict[str(feature_name)] = value

    item = PairwiseItem(
        index=index,
        query_id=query_id,
        product_id=product_id,
        relevance_grade=_parse_int(
            sample["relevance_grade"],
            label="relevance_grade",
            query_id=query_id,
            product_id=product_id,
        ),
        recall_rank=_parse_int(
            sample["recall_rank"],
            label="recall_rank",
            query_id=query_id,
            product_id=product_id,
        ),
        recall_score=_parse_float(
            sample["recall_score"],
            label="recall_score",
            query_id=query_id,
            product_id=product_id,
        ),
        features=feature_dict,
    )
    return item, feature_values


def build_pairwise_rows(
    dataset: PairwiseItemDataset,
    x_scaled: np.ndarray,
    max_pairs_per_query: int,
    min_grade_gap: int,
    pair_sampling_strategy: PairSamplingStrategy = "hard",
    seed: int = 2026,
) -> PairwiseRows:
    """Build positive and reverse pairwise diff rows from scaled item features."""
    _validate_pair_args(
        max_pairs_per_query=max_pairs_per_query,
        min_grade_gap=min_grade_gap,
        pair_sampling_strategy=pair_sampling_strategy,
    )
    x_scaled = np.asarray(x_scaled, dtype=float)
    if x_scaled.shape != dataset.x.shape:
        raise ValueError(
            "x_scaled shape must match dataset.x shape: "
            f"expected {dataset.x.shape}, got {x_scaled.shape}",
        )

    rng = random.Random(seed)
    x_diff_rows: list[np.ndarray] = []
    targets: list[int] = []
    pair_rows: list[dict[str, Any]] = []
    query_pair_counts: dict[str, int] = {}
    skipped_no_pair_query_count = 0

    for query_id, query_items in group_items_by_query(dataset.items).items():
        preferred_pairs = _candidate_preferred_pairs(
            query_items,
            min_grade_gap=min_grade_gap,
        )
        if not preferred_pairs:
            skipped_no_pair_query_count += 1
            query_pair_counts[query_id] = 0
            continue

        selected_full_pairs, extra_positive_pair = _select_preferred_pairs(
            preferred_pairs=preferred_pairs,
            max_pairs_per_query=max_pairs_per_query,
            pair_sampling_strategy=pair_sampling_strategy,
            rng=rng,
        )
        query_row_count = 0
        for pair in selected_full_pairs:
            _append_pair_row(
                pair=pair,
                x_scaled=x_scaled,
                target=1,
                x_diff_rows=x_diff_rows,
                targets=targets,
                pair_rows=pair_rows,
            )
            _append_pair_row(
                pair=pair,
                x_scaled=x_scaled,
                target=0,
                x_diff_rows=x_diff_rows,
                targets=targets,
                pair_rows=pair_rows,
            )
            query_row_count += 2
        if extra_positive_pair is not None:
            _append_pair_row(
                pair=extra_positive_pair,
                x_scaled=x_scaled,
                target=1,
                x_diff_rows=x_diff_rows,
                targets=targets,
                pair_rows=pair_rows,
            )
            query_row_count += 1
        query_pair_counts[query_id] = query_row_count

    if x_diff_rows:
        x_diff = np.vstack(x_diff_rows).astype(float)
    else:
        x_diff = np.empty((0, len(dataset.feature_names)), dtype=float)
    return PairwiseRows(
        x_diff=x_diff,
        y=np.asarray(targets, dtype=int),
        rows=pair_rows,
        query_pair_counts=query_pair_counts,
        skipped_no_pair_query_count=skipped_no_pair_query_count,
    )


def group_items_by_query(items: Sequence[PairwiseItem]) -> dict[str, list[PairwiseItem]]:
    """Group item rows by query_id, preserving input order within each query."""
    grouped: dict[str, list[PairwiseItem]] = defaultdict(list)
    for item in items:
        grouped[item.query_id].append(item)
    return dict(grouped)


def resolve_feature_names(dataset_meta: Mapping[str, Any]) -> list[str]:
    """Resolve stable feature order from dataset metadata or exporter fallback."""
    raw_feature_names = dataset_meta.get("feature_names") or list(FEATURE_NAMES)
    feature_names = [str(name) for name in raw_feature_names]
    if not feature_names:
        raise ValueError("feature_names must not be empty.")
    return feature_names


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"pairwise JSONL not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _candidate_preferred_pairs(
    query_items: Sequence[PairwiseItem],
    min_grade_gap: int,
) -> list[PreferredPair]:
    """Enumerate preferred item pairs inside one query."""
    pairs: list[PreferredPair] = []
    for left_index, left_item in enumerate(query_items):
        for right_item in query_items[left_index + 1 :]:
            grade_delta = left_item.relevance_grade - right_item.relevance_grade
            if abs(grade_delta) < min_grade_gap:
                continue
            if grade_delta > 0:
                preferred = left_item
                less_preferred = right_item
                grade_gap = grade_delta
            else:
                preferred = right_item
                less_preferred = left_item
                grade_gap = -grade_delta
            pairs.append(
                PreferredPair(
                    query_id=preferred.query_id,
                    preferred=preferred,
                    less_preferred=less_preferred,
                    grade_gap=int(grade_gap),
                    rank_distance=abs(preferred.recall_rank - less_preferred.recall_rank),
                    score_distance=abs(preferred.recall_score - less_preferred.recall_score),
                )
            )
    return pairs


def _select_preferred_pairs(
    preferred_pairs: list[PreferredPair],
    max_pairs_per_query: int,
    pair_sampling_strategy: PairSamplingStrategy,
    rng: random.Random,
) -> tuple[list[PreferredPair], PreferredPair | None]:
    """Select preferred pairs before positive/reverse expansion."""
    pairs = list(preferred_pairs)
    if pair_sampling_strategy == "hard":
        pairs.sort(key=_hard_pair_sort_key)
    elif pair_sampling_strategy == "random":
        rng.shuffle(pairs)

    full_pair_limit = max_pairs_per_query // 2
    selected_full_pairs = pairs[:full_pair_limit]
    extra_positive_pair = None
    if max_pairs_per_query % 2 == 1 and len(pairs) > full_pair_limit:
        extra_positive_pair = pairs[full_pair_limit]
    return selected_full_pairs, extra_positive_pair


def _append_pair_row(
    pair: PreferredPair,
    x_scaled: np.ndarray,
    target: int,
    x_diff_rows: list[np.ndarray],
    targets: list[int],
    pair_rows: list[dict[str, Any]],
) -> None:
    """Append one positive or reverse pair row."""
    if target == 1:
        left = pair.preferred
        right = pair.less_preferred
        direction = "preferred_minus_less"
    else:
        left = pair.less_preferred
        right = pair.preferred
        direction = "less_minus_preferred"
    x_diff_rows.append(x_scaled[left.index] - x_scaled[right.index])
    targets.append(int(target))
    pair_rows.append(
        {
            "query_id": pair.query_id,
            "left_product_id": left.product_id,
            "right_product_id": right.product_id,
            "preferred_product_id": pair.preferred.product_id,
            "less_preferred_product_id": pair.less_preferred.product_id,
            "target": int(target),
            "direction": direction,
            "grade_gap": int(pair.grade_gap),
            "rank_distance": int(pair.rank_distance),
            "score_distance": float(pair.score_distance),
        }
    )


def _hard_pair_sort_key(pair: PreferredPair) -> tuple[int, int, float, str, str]:
    """Sort by strong but close hard pairs, with stable product-id tie-breaks."""
    return (
        -int(pair.grade_gap),
        int(pair.rank_distance),
        float(pair.score_distance),
        pair.preferred.product_id,
        pair.less_preferred.product_id,
    )


def _validate_pair_args(
    max_pairs_per_query: int,
    min_grade_gap: int,
    pair_sampling_strategy: str,
) -> None:
    """Validate pair construction parameters."""
    if max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be greater than 0.")
    if min_grade_gap <= 0:
        raise ValueError("min_grade_gap must be greater than 0.")
    if pair_sampling_strategy not in {"hard", "random"}:
        raise ValueError("pair_sampling_strategy must be 'hard' or 'random'.")


def _parse_int(value: Any, label: str, query_id: str, product_id: str) -> int:
    """Parse an integer item field."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {query_id}/{product_id} {label} must be an integer.") from exc


def _parse_float(value: Any, label: str, query_id: str, product_id: str) -> float:
    """Parse a float item field."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {query_id}/{product_id} {label} must be numeric.") from exc
