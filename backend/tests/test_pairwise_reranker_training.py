"""Pairwise reranker training tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.app.training.feature_exporter import FEATURE_NAMES
from backend.app.training.pairwise_dataset import (
    build_pairwise_rows,
    group_items_by_query,
    load_pairwise_data,
    load_pairwise_item_dataset,
)
from backend.app.training.pairwise_ranker import PairwiseLogisticRanker, feature_dicts_to_matrix
from backend.app.training.pairwise_trainer import (
    MODEL_FILE,
    MODEL_META_FILE,
    REPORT_JSON_FILE,
    REPORT_TXT_FILE,
    calculate_valid_ranking_metrics,
    load_pairwise_reranker,
    train_pairwise_reranker,
)


def test_load_pairwise_item_dataset_preserves_feature_order(tmp_path: Path) -> None:
    """JSONL rows are validated and converted with stable feature order."""
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            _sample("Q1", "p1", grade=3, recall_rank=1, feature_value=0.9),
            _sample("Q1", "p2", grade=0, recall_rank=2, feature_value=0.1),
        ],
    )

    dataset = load_pairwise_item_dataset(path, feature_names=FEATURE_NAMES)

    assert dataset.item_count == 2
    assert dataset.query_count == 1
    assert dataset.feature_names == list(FEATURE_NAMES)
    assert dataset.x.shape == (2, len(FEATURE_NAMES))
    assert dataset.x[0, list(FEATURE_NAMES).index("metadata_match_score")] == 0.9


def test_group_and_pair_construction_with_forward_reverse_rows(tmp_path: Path) -> None:
    """Higher relevance_grade items create positive and reverse pair rows."""
    dataset = _item_dataset(
        tmp_path,
        [
            _sample("Q1", "p_high", grade=3, recall_rank=1, feature_value=3.0),
            _sample("Q1", "p_low", grade=0, recall_rank=2, feature_value=1.0),
        ],
    )

    grouped = group_items_by_query(dataset.items)
    pairs = build_pairwise_rows(
        dataset,
        x_scaled=dataset.x,
        max_pairs_per_query=2,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
    )

    assert set(grouped) == {"Q1"}
    assert pairs.sample_count == 2
    assert pairs.y.tolist() == [1, 0]
    assert pairs.rows[0]["preferred_product_id"] == "p_high"
    assert pairs.rows[1]["direction"] == "less_minus_preferred"
    assert np.allclose(pairs.x_diff[0], dataset.x[0] - dataset.x[1])
    assert np.allclose(pairs.x_diff[1], dataset.x[1] - dataset.x[0])


def test_min_grade_gap_and_same_grade_skip_pairs(tmp_path: Path) -> None:
    """Same-grade rows are skipped and min_grade_gap filters close grades."""
    dataset = _item_dataset(
        tmp_path,
        [
            _sample("Q1", "p3", grade=3, recall_rank=1, feature_value=3.0),
            _sample("Q1", "p2", grade=2, recall_rank=2, feature_value=2.0),
            _sample("Q1", "p1", grade=1, recall_rank=3, feature_value=1.0),
            _sample("Q2", "pa", grade=1, recall_rank=1, feature_value=1.0),
            _sample("Q2", "pb", grade=1, recall_rank=2, feature_value=1.0),
        ],
    )

    pairs = build_pairwise_rows(
        dataset,
        x_scaled=dataset.x,
        max_pairs_per_query=10,
        min_grade_gap=2,
        pair_sampling_strategy="hard",
        seed=2026,
    )

    assert pairs.skipped_no_pair_query_count == 1
    assert pairs.query_pair_counts["Q2"] == 0
    assert pairs.sample_count == 2
    assert {row["grade_gap"] for row in pairs.rows} == {2}


def test_hard_sampling_max_pairs_and_reproducibility(tmp_path: Path) -> None:
    """Hard sampling respects expanded row limit and keeps deterministic order."""
    dataset = _item_dataset(
        tmp_path,
        [
            _sample("Q1", "p1", grade=3, recall_rank=1, recall_score=0.90, feature_value=0.9),
            _sample("Q1", "p2", grade=2, recall_rank=2, recall_score=0.89, feature_value=0.8),
            _sample("Q1", "p3", grade=1, recall_rank=3, recall_score=0.88, feature_value=0.7),
            _sample("Q1", "p4", grade=0, recall_rank=4, recall_score=0.87, feature_value=0.6),
        ],
    )

    first = build_pairwise_rows(
        dataset,
        x_scaled=dataset.x,
        max_pairs_per_query=3,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
    )
    second = build_pairwise_rows(
        dataset,
        x_scaled=dataset.x,
        max_pairs_per_query=3,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
    )

    assert first.sample_count == 3
    assert first.rows == second.rows
    assert first.rows[0]["grade_gap"] == 3
    assert first.y.tolist() == [1, 0, 1]


def test_pairwise_ranker_fit_and_item_scores_rank_high_grade_first(tmp_path: Path) -> None:
    """PairwiseLogisticRanker learns an item-level score from pair diffs."""
    dataset = _item_dataset(
        tmp_path,
        [
            _sample("Q1", "p_high", grade=3, recall_rank=1, feature_value=0.95),
            _sample("Q1", "p_mid", grade=2, recall_rank=2, feature_value=0.60),
            _sample("Q1", "p_low", grade=0, recall_rank=3, feature_value=0.10),
        ],
    )
    ranker = PairwiseLogisticRanker(feature_names=list(FEATURE_NAMES), seed=2026)
    ranker.fit_item_scaler(dataset.x)
    x_scaled = ranker.transform_items(dataset.x)
    pairs = build_pairwise_rows(
        dataset,
        x_scaled=x_scaled,
        max_pairs_per_query=10,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
    )

    ranker.fit_classifier(pairs.x_diff, pairs.y)
    scores = ranker.score_items(dataset.x)

    assert scores[0] > scores[1] > scores[2]
    assert ranker.coef_ is not None
    assert ranker.predict_pairwise(pairs.x_diff).shape == pairs.y.shape


def test_feature_dicts_to_matrix_validates_feature_names() -> None:
    """Feature dict conversion uses stable names and rejects missing fields."""
    matrix = feature_dicts_to_matrix(
        [_features(0.7)],
        feature_names=FEATURE_NAMES,
    )

    assert matrix.shape == (1, len(FEATURE_NAMES))
    with pytest.raises(ValueError, match="missing feature"):
        feature_dicts_to_matrix([{"recall_score": 0.1}], feature_names=FEATURE_NAMES)


def test_valid_query_level_ranking_metrics_are_computed(tmp_path: Path) -> None:
    """Valid ranking metrics compare recall ordering and pairwise scores."""
    dataset = _item_dataset(
        tmp_path,
        [
            _sample("Q1", "p_low", grade=0, recall_rank=1, feature_value=0.1),
            _sample("Q1", "p_high", grade=3, recall_rank=2, feature_value=0.9),
            _sample("Q1", "p_mid", grade=1, recall_rank=3, feature_value=0.5),
        ],
    )
    ranker = PairwiseLogisticRanker(feature_names=list(FEATURE_NAMES), seed=2026)
    ranker.fit_item_scaler(dataset.x)
    pairs = build_pairwise_rows(
        dataset,
        x_scaled=ranker.transform_items(dataset.x),
        max_pairs_per_query=10,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
    )
    ranker.fit_classifier(pairs.x_diff, pairs.y)

    metrics = calculate_valid_ranking_metrics(dataset, ranker, k=2)

    assert metrics["evaluated_query_count"] == 1
    assert metrics["valid_pairwise_ranking_metrics"]["ndcg_at_k"] > metrics[
        "valid_recall_ranking_metrics"
    ]["ndcg_at_k"]
    assert "mrr" in metrics["pairwise_vs_recall_delta"]


def test_train_pairwise_reranker_writes_model_meta_and_reports(tmp_path: Path) -> None:
    """End-to-end trainer saves pairwise model, meta, and reports."""
    paths = _write_pairwise_training_files(tmp_path)

    result = train_pairwise_reranker(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
        model_output_dir=tmp_path / "models" / "reranker_pairwise",
        report_output_dir=tmp_path / "outputs" / "training_pairwise",
        max_pairs_per_query=6,
        min_grade_gap=1,
        pair_sampling_strategy="hard",
        seed=2026,
        overwrite=True,
        project_root=tmp_path,
    )

    model_path = tmp_path / "models" / "reranker_pairwise" / MODEL_FILE
    meta_path = tmp_path / "models" / "reranker_pairwise" / MODEL_META_FILE
    report_json_path = tmp_path / "outputs" / "training_pairwise" / REPORT_JSON_FILE
    report_txt_path = tmp_path / "outputs" / "training_pairwise" / REPORT_TXT_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert model_path.is_file()
    assert meta_path.is_file()
    assert report_json_path.is_file()
    assert report_txt_path.is_file()
    assert meta["model_type"] == "pairwise_logistic_ranker"
    assert meta["pairwise_train_pair_count"] > 0
    assert meta["score_definition"] == "coef dot standardized features"
    assert "valid_ranking" in meta["metrics"]
    assert result.report["metrics"]["valid_pairwise"]["pairwise_sample_count"] > 0
    assert "Pairwise Reranker Training Report" in report_txt_path.read_text(encoding="utf-8")


def test_pairwise_model_loads_and_rejects_feature_mismatch(tmp_path: Path) -> None:
    """Saved pairwise model can score items and validates feature_names on load."""
    paths = _write_pairwise_training_files(tmp_path)
    result = train_pairwise_reranker(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
        model_output_dir=tmp_path / "models" / "reranker_pairwise",
        report_output_dir=tmp_path / "outputs" / "training_pairwise",
        max_pairs_per_query=6,
        min_grade_gap=1,
        seed=2026,
        overwrite=True,
        project_root=tmp_path,
    )

    loaded_model, loaded_meta = load_pairwise_reranker(
        result.output_paths["model_path"],
        result.output_paths["model_meta_path"],
        expected_feature_names=FEATURE_NAMES,
    )
    scores = loaded_model.score_items(np.asarray([list(_features(0.8).values())]))

    assert loaded_meta["model_type"] == "pairwise_logistic_ranker"
    assert scores.shape == (1,)
    with pytest.raises(ValueError, match="feature_names mismatch"):
        load_pairwise_reranker(
            result.output_paths["model_path"],
            result.output_paths["model_meta_path"],
            expected_feature_names=["wrong_feature"],
        )


def test_train_pairwise_reranker_rejects_existing_outputs(tmp_path: Path) -> None:
    """Trainer refuses to overwrite pairwise model outputs without --overwrite."""
    paths = _write_pairwise_training_files(tmp_path)
    model_output_dir = tmp_path / "models" / "reranker_pairwise"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir / MODEL_FILE).write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="use --overwrite"):
        train_pairwise_reranker(
            train_path=paths["train"],
            valid_path=paths["valid"],
            dataset_meta_path=paths["meta"],
            model_output_dir=model_output_dir,
            report_output_dir=tmp_path / "outputs" / "training_pairwise",
            project_root=tmp_path,
            overwrite=False,
        )


def test_load_pairwise_data_reads_train_valid_and_meta(tmp_path: Path) -> None:
    """Bundle loader reads train/valid splits with dataset meta feature_names."""
    paths = _write_pairwise_training_files(tmp_path)

    bundle = load_pairwise_data(paths["train"], paths["valid"], paths["meta"])

    assert bundle.feature_names == list(FEATURE_NAMES)
    assert bundle.train.query_count == 2
    assert bundle.valid.query_count == 1
    assert bundle.dataset_meta["note"] == "test weak labels"


def test_train_pairwise_reranker_help() -> None:
    """train_pairwise_reranker.py --help is available without loading models."""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/train_pairwise_reranker.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--max-pairs-per-query" in completed.stdout
    assert "--min-grade-gap" in completed.stdout
    assert "--pair-sampling-strategy" in completed.stdout
    assert "--overwrite" in completed.stdout


def _item_dataset(tmp_path: Path, rows: list[dict]) -> object:
    path = tmp_path / "items.jsonl"
    _write_jsonl(path, rows)
    return load_pairwise_item_dataset(path, feature_names=FEATURE_NAMES)


def _write_pairwise_training_files(tmp_path: Path) -> dict[str, Path]:
    data_dir = tmp_path / "data" / "processed" / "pairwise"
    train_path = data_dir / "reranker_train.jsonl"
    valid_path = data_dir / "reranker_valid.jsonl"
    meta_path = data_dir / "reranker_dataset_meta.json"
    _write_jsonl(
        train_path,
        [
            _sample("Q1", "q1_high", grade=3, recall_rank=1, feature_value=0.95),
            _sample("Q1", "q1_mid", grade=2, recall_rank=2, feature_value=0.70),
            _sample("Q1", "q1_low", grade=0, recall_rank=3, feature_value=0.10),
            _sample("Q2", "q2_high", grade=3, recall_rank=1, feature_value=0.90),
            _sample("Q2", "q2_low", grade=0, recall_rank=2, feature_value=0.20),
        ],
    )
    _write_jsonl(
        valid_path,
        [
            _sample("QV", "v_low", grade=0, recall_rank=1, feature_value=0.15),
            _sample("QV", "v_high", grade=3, recall_rank=2, feature_value=0.85),
            _sample("QV", "v_mid", grade=1, recall_rank=3, feature_value=0.45),
        ],
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "feature_names": list(FEATURE_NAMES),
                "label_policy": "test policy",
                "note": "test weak labels",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {"train": train_path, "valid": valid_path, "meta": meta_path}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")


def _sample(
    query_id: str,
    product_id: str,
    *,
    grade: int,
    recall_rank: int,
    recall_score: float | None = None,
    feature_value: float,
) -> dict:
    recall_score = feature_value if recall_score is None else recall_score
    return {
        "query_id": query_id,
        "query_text": f"query {query_id}",
        "product_id": product_id,
        "label": 1 if grade >= 2 else 0,
        "relevance_grade": grade,
        "recall_rank": recall_rank,
        "recall_score": recall_score,
        "features": _features(feature_value),
        "split": "train",
    }


def _features(value: float) -> dict[str, float]:
    return {
        "recall_score": float(value),
        "freshness_score": float(value),
        "title_match": float(value),
        "article_type_match": float(value),
        "color_match": float(value),
        "gender_match": float(value),
        "usage_match": float(value),
        "sub_category_match": float(value),
        "text_match_score": float(value),
        "metadata_match_score": float(value),
    }
