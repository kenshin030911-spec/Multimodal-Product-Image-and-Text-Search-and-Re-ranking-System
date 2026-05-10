"""Train a pairwise logistic ranking reranker."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative
from backend.app.evaluation.metrics import aggregate_metrics, calculate_ranking_metrics
from backend.app.training.model_store import load_model_artifacts, save_model_artifacts
from backend.app.training.pairwise_dataset import (
    PairSamplingStrategy,
    PairwiseDataBundle,
    PairwiseItem,
    PairwiseItemDataset,
    PairwiseRows,
    build_pairwise_rows,
    group_items_by_query,
    load_pairwise_data,
)
from backend.app.training.pairwise_ranker import PairwiseLogisticRanker
from backend.app.training.reranker_trainer import normalize_class_weight
from backend.app.training.training_metrics import calculate_classification_metrics


DEFAULT_DATASET_DIR = Path("data/processed/reranker_dataset_aug_q500_c150_pos20_neg40")
DEFAULT_TRAIN_FILE = "reranker_train.jsonl"
DEFAULT_VALID_FILE = "reranker_valid.jsonl"
DEFAULT_DATASET_META_FILE = "reranker_dataset_meta.json"
DEFAULT_MAX_PAIRS_PER_QUERY = 200
DEFAULT_MIN_GRADE_GAP = 1
DEFAULT_PAIR_SAMPLING_STRATEGY: PairSamplingStrategy = "hard"
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_MAX_ITER = 1000
DEFAULT_SOLVER = "lbfgs"
DEFAULT_SEED = 2026
DEFAULT_RANKING_METRIC_K = 10

MODEL_TYPE = "pairwise_logistic_ranker"
MODEL_FILE = "pairwise_reranker.joblib"
MODEL_META_FILE = "pairwise_reranker_meta.json"
REPORT_JSON_FILE = "pairwise_training_report.json"
REPORT_TXT_FILE = "pairwise_training_report.txt"
PAIRWISE_SCORE_DEFINITION = "coef dot standardized features"
PAIRWISE_SCORE_NOTE = (
    "pairwise_rerank_score is an ordering score, not calibrated probability"
)
RELEVANCE_POLICY = "higher relevance_grade should rank earlier within the same query"
PAIRWISE_NOTE = "weak-supervised pairwise ranking labels, not human annotations"


@dataclass(frozen=True)
class PairwiseTrainingResult:
    """Artifacts and metrics returned by a pairwise training run."""

    model: PairwiseLogisticRanker
    model_meta: dict[str, Any]
    report: dict[str, Any]
    output_paths: dict[str, Path]


def default_train_path(project_root: Path | None = None) -> Path:
    """Return the default augmented reranker train JSONL path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_TRAIN_FILE


def default_valid_path(project_root: Path | None = None) -> Path:
    """Return the default augmented reranker valid JSONL path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_VALID_FILE


def default_dataset_meta_path(project_root: Path | None = None) -> Path:
    """Return the default augmented reranker dataset meta path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_DATASET_META_FILE


def train_pairwise_reranker(
    train_path: Path | None = None,
    valid_path: Path | None = None,
    dataset_meta_path: Path | None = None,
    model_output_dir: Path | None = None,
    report_output_dir: Path | None = None,
    max_pairs_per_query: int = DEFAULT_MAX_PAIRS_PER_QUERY,
    min_grade_gap: int = DEFAULT_MIN_GRADE_GAP,
    pair_sampling_strategy: PairSamplingStrategy = DEFAULT_PAIR_SAMPLING_STRATEGY,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: str = DEFAULT_SOLVER,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> PairwiseTrainingResult:
    """Train, validate, and save a pairwise logistic ranking reranker."""
    _validate_train_args(
        max_pairs_per_query=max_pairs_per_query,
        min_grade_gap=min_grade_gap,
        pair_sampling_strategy=pair_sampling_strategy,
        max_iter=max_iter,
    )
    settings = get_settings()
    project_root = project_root or settings.project_root
    train_path = train_path or default_train_path(project_root)
    valid_path = valid_path or default_valid_path(project_root)
    dataset_meta_path = dataset_meta_path or default_dataset_meta_path(project_root)
    model_output_dir = model_output_dir or (settings.project_root / "models" / "reranker_pairwise")
    report_output_dir = report_output_dir or (settings.output_dir / "training_pairwise")
    output_paths = {
        "model_path": model_output_dir / MODEL_FILE,
        "model_meta_path": model_output_dir / MODEL_META_FILE,
        "report_json_path": report_output_dir / REPORT_JSON_FILE,
        "report_txt_path": report_output_dir / REPORT_TXT_FILE,
    }
    _ensure_model_outputs_can_write(
        model_path=output_paths["model_path"],
        model_meta_path=output_paths["model_meta_path"],
        overwrite=overwrite,
    )

    data = load_pairwise_data(
        train_path=train_path,
        valid_path=valid_path,
        dataset_meta_path=dataset_meta_path,
    )
    ranker = PairwiseLogisticRanker(
        feature_names=list(data.feature_names),
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        seed=seed,
    )
    ranker.fit_item_scaler(data.train.x)
    train_x_scaled = ranker.transform_items(data.train.x)
    valid_x_scaled = ranker.transform_items(data.valid.x)
    train_pairs = build_pairwise_rows(
        dataset=data.train,
        x_scaled=train_x_scaled,
        max_pairs_per_query=max_pairs_per_query,
        min_grade_gap=min_grade_gap,
        pair_sampling_strategy=pair_sampling_strategy,
        seed=seed,
    )
    valid_pairs = build_pairwise_rows(
        dataset=data.valid,
        x_scaled=valid_x_scaled,
        max_pairs_per_query=max_pairs_per_query,
        min_grade_gap=min_grade_gap,
        pair_sampling_strategy=pair_sampling_strategy,
        seed=seed,
    )
    ranker.fit_classifier(train_pairs.x_diff, train_pairs.y)

    train_pairwise_metrics = _pairwise_metrics(ranker, train_pairs)
    valid_pairwise_metrics = _pairwise_metrics(ranker, valid_pairs)
    valid_ranking_metrics = calculate_valid_ranking_metrics(
        dataset=data.valid,
        ranker=ranker,
        k=DEFAULT_RANKING_METRIC_K,
    )
    metrics = {
        "train_pairwise": train_pairwise_metrics,
        "valid_pairwise": valid_pairwise_metrics,
        "valid_ranking": valid_ranking_metrics,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    normalized_class_weight = normalize_class_weight(class_weight)
    class_weight_for_meta = normalized_class_weight or "none"
    feature_coefficients = _feature_coefficients(ranker, data.feature_names)
    scaler_info = _scaler_info(ranker)
    query_counts = {
        "train_query_count": data.train.query_count,
        "valid_query_count": data.valid.query_count,
    }
    pair_counts = {
        "pairwise_train_pair_count": train_pairs.sample_count,
        "pairwise_valid_pair_count": valid_pairs.sample_count,
    }

    model_meta = {
        "model_type": MODEL_TYPE,
        "framework": "sklearn",
        "feature_names": data.feature_names,
        "train_path": to_project_relative(train_path, project_root),
        "valid_path": to_project_relative(valid_path, project_root),
        "dataset_meta_path": to_project_relative(dataset_meta_path, project_root),
        **query_counts,
        "train_item_count": data.train.item_count,
        "valid_item_count": data.valid.item_count,
        "skipped_no_pair_train_query_count": train_pairs.skipped_no_pair_query_count,
        "skipped_no_pair_valid_query_count": valid_pairs.skipped_no_pair_query_count,
        **pair_counts,
        "max_pairs_per_query": max_pairs_per_query,
        "min_grade_gap": min_grade_gap,
        "pair_sampling_strategy": pair_sampling_strategy,
        "class_weight": class_weight_for_meta,
        "max_iter": max_iter,
        "solver": solver,
        "seed": seed,
        "created_at": created_at,
        "scaler_info": scaler_info,
        "feature_coefficients": feature_coefficients,
        "score_definition": PAIRWISE_SCORE_DEFINITION,
        "score_note": PAIRWISE_SCORE_NOTE,
        "relevance_policy": RELEVANCE_POLICY,
        "metrics": metrics,
        "dataset_note": data.dataset_meta.get("note"),
        "note": PAIRWISE_NOTE,
    }
    report = {
        **model_meta,
        "query_pair_counts": {
            "train": train_pairs.query_pair_counts,
            "valid": valid_pairs.query_pair_counts,
        },
    }

    save_model_artifacts(
        model=ranker,
        model_path=output_paths["model_path"],
        meta=model_meta,
        meta_path=output_paths["model_meta_path"],
        overwrite=overwrite,
    )
    _write_reports(report=report, output_paths=output_paths)
    return PairwiseTrainingResult(
        model=ranker,
        model_meta=model_meta,
        report=report,
        output_paths=output_paths,
    )


def load_pairwise_reranker(
    model_path: Path,
    meta_path: Path,
    expected_feature_names: Sequence[str] | None = None,
) -> tuple[PairwiseLogisticRanker, dict[str, Any]]:
    """Load a saved pairwise ranker and validate its feature names."""
    model, meta = load_model_artifacts(
        model_path=model_path,
        meta_path=meta_path,
        expected_feature_names=expected_feature_names,
        artifact_name="pairwise reranker",
    )
    if meta.get("model_type") != MODEL_TYPE:
        raise ValueError(f"pairwise reranker model_type mismatch: {meta.get('model_type')}")
    if not isinstance(model, PairwiseLogisticRanker):
        raise ValueError("pairwise reranker model artifact has unexpected type.")
    return model, meta


def calculate_valid_ranking_metrics(
    dataset: PairwiseItemDataset,
    ranker: PairwiseLogisticRanker,
    k: int = DEFAULT_RANKING_METRIC_K,
) -> dict[str, Any]:
    """Compare recall ordering and pairwise ranking on a valid item split."""
    pairwise_scores = ranker.score_items(dataset.x)
    score_by_index = {
        item.index: float(pairwise_scores[item.index])
        for item in dataset.items
    }
    recall_rows: list[dict[str, float]] = []
    pairwise_rows: list[dict[str, float]] = []
    evaluated_query_count = 0

    for query_items in group_items_by_query(dataset.items).values():
        if not query_items:
            continue
        evaluated_query_count += 1
        ideal_grades = [item.relevance_grade for item in query_items]
        positive_count = sum(1 for item in query_items if item.relevance_grade > 0)

        recall_sorted = sorted(
            query_items,
            key=lambda item: (item.recall_rank, -item.recall_score, item.product_id),
        )
        pairwise_sorted = sorted(
            query_items,
            key=lambda item: (-score_by_index[item.index], item.recall_rank, item.product_id),
        )
        recall_rows.append(
            calculate_ranking_metrics(
                [item.relevance_grade for item in recall_sorted],
                positive_count=positive_count,
                k=k,
                ideal_relevance_grades=ideal_grades,
            )
        )
        pairwise_rows.append(
            calculate_ranking_metrics(
                [item.relevance_grade for item in pairwise_sorted],
                positive_count=positive_count,
                k=k,
                ideal_relevance_grades=ideal_grades,
            )
        )

    recall_metrics = aggregate_metrics(recall_rows)
    pairwise_metrics = aggregate_metrics(pairwise_rows)
    return {
        "metric_k": k,
        "evaluated_query_count": evaluated_query_count,
        "valid_recall_ranking_metrics": recall_metrics,
        "valid_pairwise_ranking_metrics": pairwise_metrics,
        "pairwise_vs_recall_delta": _metric_delta(pairwise_metrics, recall_metrics),
    }


def _pairwise_metrics(
    ranker: PairwiseLogisticRanker,
    pairs: PairwiseRows,
) -> dict[str, Any]:
    """Calculate pairwise classification metrics with pairwise-prefixed names."""
    if pairs.sample_count == 0:
        base_metrics = calculate_classification_metrics([], [], [])
    else:
        y_pred = ranker.predict_pairwise(pairs.x_diff)
        y_score = ranker.predict_pairwise_scores(pairs.x_diff)
        base_metrics = calculate_classification_metrics(pairs.y, y_pred, y_score)
    return {
        "pairwise_accuracy": base_metrics["accuracy"],
        "pairwise_precision": base_metrics["precision"],
        "pairwise_recall": base_metrics["recall"],
        "pairwise_f1": base_metrics["f1"],
        "pairwise_sample_count": base_metrics["sample_count"],
        "positive_rate": base_metrics["positive_rate"],
        "confusion_matrix": base_metrics["confusion_matrix"],
        "warnings": base_metrics.get("warnings", []),
    }


def _metric_delta(
    target_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Return target minus baseline for observed metric keys."""
    return {
        key: float(target_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0))
        for key in sorted(set(target_metrics) | set(baseline_metrics))
    }


def _feature_coefficients(
    ranker: PairwiseLogisticRanker,
    feature_names: Sequence[str],
) -> dict[str, float]:
    """Return item-level scoring coefficients keyed by feature name."""
    if ranker.coef_ is None:
        raise ValueError("pairwise ranker coefficients are not available.")
    return {
        feature_name: float(coef)
        for feature_name, coef in zip(feature_names, ranker.coef_, strict=True)
    }


def _scaler_info(ranker: PairwiseLogisticRanker) -> dict[str, Any]:
    """Return lightweight scaler metadata."""
    if ranker.scaler is None:
        raise ValueError("pairwise ranker scaler is not available.")
    return {
        "type": "StandardScaler",
        "with_mean": bool(ranker.scaler.with_mean),
        "with_std": bool(ranker.scaler.with_std),
        "n_features_in": int(ranker.scaler.n_features_in_),
    }


def _write_reports(report: dict[str, Any], output_paths: Mapping[str, Path]) -> None:
    """Write JSON and text training reports."""
    report_json_path = output_paths["report_json_path"]
    report_txt_path = output_paths["report_txt_path"]
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with report_json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    report_txt_path.write_text(_format_text_report(report), encoding="utf-8")


def _format_text_report(report: Mapping[str, Any]) -> str:
    """Create a readable pairwise training summary."""
    metrics = report["metrics"]
    train_pairwise = metrics["train_pairwise"]
    valid_pairwise = metrics["valid_pairwise"]
    valid_ranking = metrics["valid_ranking"]
    recall_ranking = valid_ranking["valid_recall_ranking_metrics"]
    pairwise_ranking = valid_ranking["valid_pairwise_ranking_metrics"]
    delta = valid_ranking["pairwise_vs_recall_delta"]
    lines = [
        "Pairwise Reranker Training Report",
        "",
        f"model_type: {report['model_type']}",
        f"framework: {report['framework']}",
        f"train_query_count: {report['train_query_count']}",
        f"valid_query_count: {report['valid_query_count']}",
        f"train_item_count: {report['train_item_count']}",
        f"valid_item_count: {report['valid_item_count']}",
        f"pairwise_train_pair_count: {report['pairwise_train_pair_count']}",
        f"pairwise_valid_pair_count: {report['pairwise_valid_pair_count']}",
        f"max_pairs_per_query: {report['max_pairs_per_query']}",
        f"min_grade_gap: {report['min_grade_gap']}",
        f"pair_sampling_strategy: {report['pair_sampling_strategy']}",
        f"class_weight: {report['class_weight']}",
        f"max_iter: {report['max_iter']}",
        f"solver: {report['solver']}",
        f"seed: {report['seed']}",
        "",
        "Train Pairwise Metrics",
        _format_pairwise_metrics(train_pairwise),
        "",
        "Valid Pairwise Metrics",
        _format_pairwise_metrics(valid_pairwise),
        "",
        "Valid Query-Level Ranking Metrics",
        f"metric_k: {valid_ranking['metric_k']}",
        f"evaluated_query_count: {valid_ranking['evaluated_query_count']}",
        (
            "recall: "
            f"ndcg={recall_ranking.get('ndcg_at_k', 0.0):.6f}, "
            f"mrr={recall_ranking.get('mrr', 0.0):.6f}, "
            f"hit={recall_ranking.get('hit_at_k', 0.0):.6f}"
        ),
        (
            "pairwise: "
            f"ndcg={pairwise_ranking.get('ndcg_at_k', 0.0):.6f}, "
            f"mrr={pairwise_ranking.get('mrr', 0.0):.6f}, "
            f"hit={pairwise_ranking.get('hit_at_k', 0.0):.6f}"
        ),
        (
            "pairwise_vs_recall: "
            f"ndcg={delta.get('ndcg_at_k', 0.0):.6f}, "
            f"mrr={delta.get('mrr', 0.0):.6f}, "
            f"hit={delta.get('hit_at_k', 0.0):.6f}"
        ),
        "",
        "Feature Coefficients",
    ]
    lines.extend(
        f"- {name}: {value:.6f}"
        for name, value in report["feature_coefficients"].items()
    )
    lines.extend(
        [
            "",
            f"score_definition: {report['score_definition']}",
            f"score_note: {report['score_note']}",
            f"relevance_policy: {report['relevance_policy']}",
            f"note: {report['note']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_pairwise_metrics(metrics: Mapping[str, Any]) -> str:
    """Format pairwise metrics for text report."""
    return "\n".join(
        [
            f"pairwise_accuracy: {metrics['pairwise_accuracy']:.6f}",
            f"pairwise_precision: {metrics['pairwise_precision']:.6f}",
            f"pairwise_recall: {metrics['pairwise_recall']:.6f}",
            f"pairwise_f1: {metrics['pairwise_f1']:.6f}",
            f"pairwise_sample_count: {metrics['pairwise_sample_count']}",
            f"positive_rate: {metrics['positive_rate']:.6f}",
            f"confusion_matrix: {metrics['confusion_matrix']}",
        ],
    )


def _validate_train_args(
    max_pairs_per_query: int,
    min_grade_gap: int,
    pair_sampling_strategy: str,
    max_iter: int,
) -> None:
    """Validate top-level pairwise training options."""
    if max_pairs_per_query <= 0:
        raise ValueError("max_pairs_per_query must be greater than 0.")
    if min_grade_gap <= 0:
        raise ValueError("min_grade_gap must be greater than 0.")
    if pair_sampling_strategy not in {"hard", "random"}:
        raise ValueError("pair_sampling_strategy must be 'hard' or 'random'.")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0.")


def _ensure_model_outputs_can_write(
    model_path: Path,
    model_meta_path: Path,
    overwrite: bool,
) -> None:
    """Fail before training when pairwise model outputs already exist."""
    if overwrite:
        return
    existing = [path for path in (model_path, model_meta_path) if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            "pairwise reranker output already exists; use --overwrite to replace: "
            f"{names}",
        )
