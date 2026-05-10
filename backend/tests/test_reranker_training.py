"""Tests for lightweight reranker training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backend.app.training.model_store import load_trained_reranker
from backend.app.training.reranker_trainer import (
    MODEL_FILE,
    MODEL_META_FILE,
    REPORT_JSON_FILE,
    REPORT_TXT_FILE,
    build_logistic_regression_pipeline,
    load_training_data,
    train_reranker_model,
)
from backend.app.training.training_metrics import calculate_classification_metrics


def test_load_training_data_uses_meta_feature_order(tmp_path: Path) -> None:
    """Training data loader follows dataset meta feature_names order."""
    paths = _write_dataset(
        tmp_path,
        feature_names=["second", "first"],
        train_rows=[
            _sample("Q1", "p1", 1, {"first": 1.0, "second": 2.0}),
            _sample("Q1", "p2", 0, {"first": 3.0, "second": 4.0}),
        ],
        valid_rows=[
            _sample("Q2", "p3", 1, {"first": 5.0, "second": 6.0}),
            _sample("Q2", "p4", 0, {"first": 7.0, "second": 8.0}),
        ],
    )

    data = load_training_data(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
    )

    assert data.feature_names == ["second", "first"]
    assert data.train.x[0].tolist() == [2.0, 1.0]
    assert data.train.y.tolist() == [1, 0]


def test_load_training_data_rejects_missing_feature(tmp_path: Path) -> None:
    """Missing features fail with a query/product specific error."""
    paths = _write_dataset(
        tmp_path,
        feature_names=["recall_score", "color_match"],
        train_rows=[
            _sample("Q001", "p123", 1, {"recall_score": 0.9}),
            _sample("Q001", "p124", 0, {"recall_score": 0.1, "color_match": 0.0}),
        ],
        valid_rows=[
            _sample("Q002", "p125", 1, {"recall_score": 0.8, "color_match": 1.0}),
            _sample("Q002", "p126", 0, {"recall_score": 0.2, "color_match": 0.0}),
        ],
    )

    with pytest.raises(ValueError, match="sample Q001/p123 missing feature color_match"):
        load_training_data(
            train_path=paths["train"],
            valid_path=paths["valid"],
            dataset_meta_path=paths["meta"],
        )


def test_load_training_data_rejects_bad_label(tmp_path: Path) -> None:
    """Labels must be binary 0/1 values."""
    paths = _write_dataset(
        tmp_path,
        feature_names=["recall_score"],
        train_rows=[
            _sample("Q1", "p1", 2, {"recall_score": 0.9}),
            _sample("Q1", "p2", 0, {"recall_score": 0.1}),
        ],
        valid_rows=[
            _sample("Q2", "p3", 1, {"recall_score": 0.8}),
            _sample("Q2", "p4", 0, {"recall_score": 0.2}),
        ],
    )

    with pytest.raises(ValueError, match="sample Q1/p1 label must be 0 or 1"):
        load_training_data(
            train_path=paths["train"],
            valid_path=paths["valid"],
            dataset_meta_path=paths["meta"],
        )


def test_training_metrics_required_fields_and_single_class_safe() -> None:
    """Metric helper returns required fields and does not crash on one class."""
    metrics = calculate_classification_metrics(
        y_true=[1, 1],
        y_pred=[1, 0],
        y_score=[0.9, 0.2],
    )

    assert {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
        "positive_rate",
        "sample_count",
    }.issubset(metrics)
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["confusion_matrix"] == [[0, 0], [1, 1]]
    assert metrics["warnings"]


def test_train_logistic_regression_writes_model_report_and_loads(tmp_path: Path) -> None:
    """Trainer writes model/meta/report and loaded model supports predict_proba."""
    paths = _write_linearly_separable_dataset(tmp_path)

    result = train_reranker_model(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
        model_output_dir=tmp_path / "models" / "reranker",
        report_output_dir=tmp_path / "outputs" / "training",
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        seed=7,
        overwrite=True,
        project_root=tmp_path,
    )

    model_path = tmp_path / "models" / "reranker" / MODEL_FILE
    meta_path = tmp_path / "models" / "reranker" / MODEL_META_FILE
    report_json_path = tmp_path / "outputs" / "training" / REPORT_JSON_FILE
    report_txt_path = tmp_path / "outputs" / "training" / REPORT_TXT_FILE
    loaded_model, loaded_meta = load_trained_reranker(
        model_path=model_path,
        meta_path=meta_path,
        expected_feature_names=["recall_score", "color_match"],
    )

    assert model_path.is_file()
    assert meta_path.is_file()
    assert report_json_path.is_file()
    assert report_txt_path.is_file()
    assert result.report["valid_metrics"]["sample_count"] == 4
    assert result.model_meta["class_weight"] == "balanced"
    assert result.model.named_steps["classifier"].class_weight == "balanced"
    assert set(result.report["feature_coefficients"]) == {"recall_score", "color_match"}
    assert loaded_meta["trained_rerank_score_definition"] == "predict_proba(features)[1]"
    assert loaded_model.predict_proba(np.asarray([[0.95, 1.0]], dtype=float)).shape == (1, 2)


def test_train_logistic_regression_supports_none_class_weight(tmp_path: Path) -> None:
    """class_weight='none' maps to sklearn None and still trains."""
    paths = _write_linearly_separable_dataset(tmp_path)

    result = train_reranker_model(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
        model_output_dir=tmp_path / "models" / "reranker",
        report_output_dir=tmp_path / "outputs" / "training",
        class_weight="none",
        overwrite=True,
        project_root=tmp_path,
    )

    assert result.model.named_steps["classifier"].class_weight is None
    assert result.model_meta["class_weight"] == "none"


def test_build_pipeline_rejects_unknown_class_weight() -> None:
    """Only balanced and none are valid class_weight values."""
    with pytest.raises(ValueError, match="class_weight"):
        build_logistic_regression_pipeline(class_weight="heavy")


def test_load_trained_reranker_rejects_feature_mismatch(tmp_path: Path) -> None:
    """Model loading validates expected feature names."""
    paths = _write_linearly_separable_dataset(tmp_path)
    train_reranker_model(
        train_path=paths["train"],
        valid_path=paths["valid"],
        dataset_meta_path=paths["meta"],
        model_output_dir=tmp_path / "models" / "reranker",
        report_output_dir=tmp_path / "outputs" / "training",
        overwrite=True,
        project_root=tmp_path,
    )

    with pytest.raises(ValueError, match="feature_names mismatch"):
        load_trained_reranker(
            model_path=tmp_path / "models" / "reranker" / MODEL_FILE,
            meta_path=tmp_path / "models" / "reranker" / MODEL_META_FILE,
            expected_feature_names=["other_feature"],
        )


def test_train_rejects_existing_model_without_overwrite(tmp_path: Path) -> None:
    """Trainer refuses to overwrite existing model outputs by default."""
    paths = _write_linearly_separable_dataset(tmp_path)
    model_dir = tmp_path / "models" / "reranker"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / MODEL_FILE).write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="use --overwrite"):
        train_reranker_model(
            train_path=paths["train"],
            valid_path=paths["valid"],
            dataset_meta_path=paths["meta"],
            model_output_dir=model_dir,
            report_output_dir=tmp_path / "outputs" / "training",
            overwrite=False,
            project_root=tmp_path,
        )


def test_train_reranker_help() -> None:
    """train_reranker.py --help is available without loading models or data."""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/train_reranker.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--class-weight" in completed.stdout
    assert "--solver" in completed.stdout
    assert "--overwrite" in completed.stdout


def _write_linearly_separable_dataset(tmp_path: Path) -> dict[str, Path]:
    """Write a small train/valid dataset with both labels."""
    feature_names = ["recall_score", "color_match"]
    return _write_dataset(
        tmp_path,
        feature_names=feature_names,
        train_rows=[
            _sample("Q1", "p1", 1, {"recall_score": 0.95, "color_match": 1.0}),
            _sample("Q1", "p2", 1, {"recall_score": 0.90, "color_match": 1.0}),
            _sample("Q2", "p3", 1, {"recall_score": 0.85, "color_match": 1.0}),
            _sample("Q2", "p4", 1, {"recall_score": 0.80, "color_match": 1.0}),
            _sample("Q3", "p5", 0, {"recall_score": 0.20, "color_match": 0.0}),
            _sample("Q3", "p6", 0, {"recall_score": 0.15, "color_match": 0.0}),
            _sample("Q4", "p7", 0, {"recall_score": 0.10, "color_match": 0.0}),
            _sample("Q4", "p8", 0, {"recall_score": 0.05, "color_match": 0.0}),
        ],
        valid_rows=[
            _sample("Q5", "p9", 1, {"recall_score": 0.88, "color_match": 1.0}),
            _sample("Q5", "p10", 1, {"recall_score": 0.82, "color_match": 1.0}),
            _sample("Q6", "p11", 0, {"recall_score": 0.18, "color_match": 0.0}),
            _sample("Q6", "p12", 0, {"recall_score": 0.12, "color_match": 0.0}),
        ],
    )


def _write_dataset(
    tmp_path: Path,
    feature_names: list[str],
    train_rows: list[dict],
    valid_rows: list[dict],
) -> dict[str, Path]:
    """Write train/valid JSONL and dataset meta files."""
    data_dir = tmp_path / "data" / "processed" / "reranker_dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "reranker_train.jsonl"
    valid_path = data_dir / "reranker_valid.jsonl"
    meta_path = data_dir / "reranker_dataset_meta.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    meta = {
        "feature_names": feature_names,
        "label_policy": "weak labels for tests",
        "note": "weak-supervised labels, not large-scale manual annotations",
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return {"train": train_path, "valid": valid_path, "meta": meta_path}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")


def _sample(
    query_id: str,
    product_id: str,
    label: int,
    features: dict[str, float],
) -> dict:
    """Build one training sample."""
    return {
        "query_id": query_id,
        "query_text": "query",
        "product_id": product_id,
        "label": label,
        "relevance_grade": 3 if label else 0,
        "label_source": "weak_metadata",
        "recall_rank": 1,
        "recall_score": float(features.get("recall_score", 0.5)),
        "features": features,
        "split": "train",
    }
