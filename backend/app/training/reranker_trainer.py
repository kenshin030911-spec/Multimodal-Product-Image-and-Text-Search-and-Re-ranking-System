"""Train a lightweight sklearn reranker on exported weak-supervised samples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative
from backend.app.training.feature_exporter import FEATURE_NAMES
from backend.app.training.model_store import save_trained_reranker
from backend.app.training.training_metrics import calculate_classification_metrics


DEFAULT_DATASET_DIR = Path("data/processed/reranker_dataset_q300_c150_pos20_neg40")
DEFAULT_TRAIN_FILE = "reranker_train.jsonl"
DEFAULT_VALID_FILE = "reranker_valid.jsonl"
DEFAULT_DATASET_META_FILE = "reranker_dataset_meta.json"
DEFAULT_MODEL_TYPE = "logistic_regression"
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_MAX_ITER = 1000
DEFAULT_SOLVER = "lbfgs"
DEFAULT_SEED = 42

MODEL_FILE = "trained_reranker.joblib"
MODEL_META_FILE = "trained_reranker_meta.json"
REPORT_JSON_FILE = "reranker_training_report.json"
REPORT_TXT_FILE = "reranker_training_report.txt"
WEAK_LABEL_NOTE = "weak-supervised labels, not large-scale manual annotations"
MODEL_NOTE = (
    "Model trained on weak-supervised labels; not a human-labeled relevance model."
)
TRAINED_SCORE_DEFINITION = "predict_proba(features)[1]"


@dataclass(frozen=True)
class TrainingDataset:
    """Loaded JSONL samples converted to model-ready arrays."""

    path: Path
    samples: list[dict[str, Any]]
    feature_names: list[str]
    x: np.ndarray
    y: np.ndarray

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""
        return int(self.y.size)

    @property
    def positive_rate(self) -> float:
        """Return the positive label ratio."""
        return float(np.mean(self.y == 1)) if self.sample_count else 0.0


@dataclass(frozen=True)
class TrainingDataBundle:
    """Train/valid datasets plus source metadata."""

    train: TrainingDataset
    valid: TrainingDataset
    dataset_meta: dict[str, Any]
    feature_names: list[str]


@dataclass(frozen=True)
class RerankerTrainingResult:
    """Artifacts and metrics returned by a training run."""

    model: Pipeline
    model_meta: dict[str, Any]
    report: dict[str, Any]
    output_paths: dict[str, Path]


def default_train_path(project_root: Path | None = None) -> Path:
    """Return the default generated reranker train JSONL path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_TRAIN_FILE


def default_valid_path(project_root: Path | None = None) -> Path:
    """Return the default generated reranker valid JSONL path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_VALID_FILE


def default_dataset_meta_path(project_root: Path | None = None) -> Path:
    """Return the default generated reranker dataset meta path."""
    root = project_root or get_settings().project_root
    return root / DEFAULT_DATASET_DIR / DEFAULT_DATASET_META_FILE


def load_training_data(
    train_path: Path,
    valid_path: Path,
    dataset_meta_path: Path,
) -> TrainingDataBundle:
    """Read train/valid JSONL files and convert them to X/y arrays."""
    dataset_meta = _read_json(dataset_meta_path)
    feature_names = _resolve_feature_names(dataset_meta)
    train = load_training_dataset(train_path, feature_names=feature_names)
    valid = load_training_dataset(valid_path, feature_names=feature_names)
    return TrainingDataBundle(
        train=train,
        valid=valid,
        dataset_meta=dataset_meta,
        feature_names=feature_names,
    )


def load_training_dataset(path: Path, feature_names: Sequence[str]) -> TrainingDataset:
    """Load one JSONL split and validate all samples."""
    samples = _read_jsonl(path)
    if not samples:
        raise ValueError(f"training split is empty: {path}")

    x_rows: list[list[float]] = []
    labels: list[int] = []
    stable_feature_names = list(feature_names)
    for line_number, sample in enumerate(samples, start=1):
        x_row, label = _sample_to_xy(
            sample=sample,
            feature_names=stable_feature_names,
            line_number=line_number,
            path=path,
        )
        x_rows.append(x_row)
        labels.append(label)

    return TrainingDataset(
        path=path,
        samples=samples,
        feature_names=stable_feature_names,
        x=np.asarray(x_rows, dtype=float),
        y=np.asarray(labels, dtype=int),
    )


def build_logistic_regression_pipeline(
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: str = DEFAULT_SOLVER,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """Create the StandardScaler + LogisticRegression baseline pipeline."""
    normalized_class_weight = normalize_class_weight(class_weight)
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0.")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=normalized_class_weight,
                    max_iter=max_iter,
                    solver=solver,
                    random_state=seed,
                ),
            ),
        ],
    )


def normalize_class_weight(value: str | None) -> str | None:
    """Normalize CLI class_weight values into sklearn-compatible values."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "balanced":
        return "balanced"
    if normalized == "none":
        return None
    raise ValueError("class_weight must be 'balanced' or 'none'.")


def train_reranker_model(
    train_path: Path | None = None,
    valid_path: Path | None = None,
    dataset_meta_path: Path | None = None,
    model_output_dir: Path | None = None,
    report_output_dir: Path | None = None,
    model_type: str = DEFAULT_MODEL_TYPE,
    class_weight: str | None = DEFAULT_CLASS_WEIGHT,
    max_iter: int = DEFAULT_MAX_ITER,
    solver: str = DEFAULT_SOLVER,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> RerankerTrainingResult:
    """Train, evaluate, and save the lightweight reranker model."""
    settings = get_settings()
    project_root = project_root or settings.project_root
    train_path = train_path or default_train_path(project_root)
    valid_path = valid_path or default_valid_path(project_root)
    dataset_meta_path = dataset_meta_path or default_dataset_meta_path(project_root)
    model_output_dir = model_output_dir or settings.reranker_dir
    report_output_dir = report_output_dir or (settings.output_dir / "training")
    _validate_training_args(model_type=model_type, max_iter=max_iter)

    model_path = model_output_dir / MODEL_FILE
    model_meta_path = model_output_dir / MODEL_META_FILE
    output_paths = {
        "model_path": model_path,
        "model_meta_path": model_meta_path,
        "report_json_path": report_output_dir / REPORT_JSON_FILE,
        "report_txt_path": report_output_dir / REPORT_TXT_FILE,
    }
    _ensure_model_outputs_can_write(
        model_path=model_path,
        model_meta_path=model_meta_path,
        overwrite=overwrite,
    )

    data = load_training_data(
        train_path=train_path,
        valid_path=valid_path,
        dataset_meta_path=dataset_meta_path,
    )
    _ensure_binary_train_labels(data.train.y)
    model = build_logistic_regression_pipeline(
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        seed=seed,
    )
    model.fit(data.train.x, data.train.y)

    train_metrics = _predict_and_score(model, data.train)
    valid_metrics = _predict_and_score(model, data.valid)
    feature_coefficients = _feature_coefficients(model, data.feature_names)
    warnings = _collect_warnings(train_metrics, valid_metrics)
    created_at = datetime.now(timezone.utc).isoformat()
    normalized_class_weight = normalize_class_weight(class_weight)
    class_weight_for_meta = normalized_class_weight or "none"
    label_policy = data.dataset_meta.get("label_policy")
    dataset_note = data.dataset_meta.get("note", WEAK_LABEL_NOTE)

    report = {
        "model_type": model_type,
        "framework": "sklearn",
        "feature_names": data.feature_names,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "train_sample_count": data.train.sample_count,
        "valid_sample_count": data.valid.sample_count,
        "train_positive_rate": data.train.positive_rate,
        "valid_positive_rate": data.valid.positive_rate,
        "class_weight": class_weight_for_meta,
        "max_iter": max_iter,
        "solver": solver,
        "seed": seed,
        "feature_coefficients": feature_coefficients,
        "warnings": warnings,
        "label_policy": label_policy,
        "note": MODEL_NOTE,
    }
    model_meta = {
        "model_type": model_type,
        "framework": "sklearn",
        "feature_names": data.feature_names,
        "train_path": to_project_relative(train_path, project_root),
        "valid_path": to_project_relative(valid_path, project_root),
        "dataset_meta_path": to_project_relative(dataset_meta_path, project_root),
        "train_sample_count": data.train.sample_count,
        "valid_sample_count": data.valid.sample_count,
        "train_positive_rate": data.train.positive_rate,
        "valid_positive_rate": data.valid.positive_rate,
        "class_weight": class_weight_for_meta,
        "max_iter": max_iter,
        "solver": solver,
        "seed": seed,
        "created_at": created_at,
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
        },
        "feature_coefficients": feature_coefficients,
        "label_policy": label_policy,
        "dataset_note": dataset_note,
        "trained_rerank_score_definition": TRAINED_SCORE_DEFINITION,
        "note": WEAK_LABEL_NOTE,
    }

    save_trained_reranker(
        model=model,
        model_path=model_path,
        meta=model_meta,
        meta_path=model_meta_path,
        overwrite=overwrite,
    )
    _write_reports(report=report, output_paths=output_paths)
    return RerankerTrainingResult(
        model=model,
        model_meta=model_meta,
        report=report,
        output_paths=output_paths,
    )


def _predict_and_score(model: Pipeline, dataset: TrainingDataset) -> dict[str, Any]:
    """Predict labels and probabilities for a split, then compute metrics."""
    y_score = model.predict_proba(dataset.x)[:, 1]
    y_pred = model.predict(dataset.x)
    return calculate_classification_metrics(dataset.y, y_pred, y_score)


def _feature_coefficients(model: Pipeline, feature_names: Sequence[str]) -> dict[str, float]:
    """Extract logistic regression coefficients keyed by feature name."""
    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    return {
        feature_name: float(coef)
        for feature_name, coef in zip(feature_names, coefficients, strict=True)
    }


def _write_reports(report: dict[str, Any], output_paths: Mapping[str, Path]) -> None:
    """Write machine-readable and text training reports."""
    report_json_path = output_paths["report_json_path"]
    report_txt_path = output_paths["report_txt_path"]
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with report_json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    report_txt_path.write_text(_format_text_report(report), encoding="utf-8")


def _format_text_report(report: Mapping[str, Any]) -> str:
    """Create a concise human-readable training summary."""
    train_metrics = report["train_metrics"]
    valid_metrics = report["valid_metrics"]
    lines = [
        "Reranker Training Report",
        "",
        f"model_type: {report['model_type']}",
        f"framework: {report['framework']}",
        f"train_sample_count: {report['train_sample_count']}",
        f"valid_sample_count: {report['valid_sample_count']}",
        f"train_positive_rate: {report['train_positive_rate']:.6f}",
        f"valid_positive_rate: {report['valid_positive_rate']:.6f}",
        f"class_weight: {report['class_weight']}",
        f"max_iter: {report['max_iter']}",
        f"solver: {report['solver']}",
        f"seed: {report['seed']}",
        "",
        "Train Metrics",
        _format_metrics(train_metrics),
        "",
        "Valid Metrics",
        _format_metrics(valid_metrics),
        "",
        "Feature Coefficients",
    ]
    lines.extend(
        f"- {name}: {value:.6f}"
        for name, value in report["feature_coefficients"].items()
    )
    if report["warnings"]:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.extend(["", f"note: {report['note']}"])
    return "\n".join(lines) + "\n"


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    """Format core metrics for the text report."""
    return "\n".join(
        [
            f"accuracy: {metrics['accuracy']:.6f}",
            f"precision: {metrics['precision']:.6f}",
            f"recall: {metrics['recall']:.6f}",
            f"f1: {metrics['f1']:.6f}",
            f"roc_auc: {_format_optional_float(metrics['roc_auc'])}",
            f"average_precision: {_format_optional_float(metrics['average_precision'])}",
            f"positive_rate: {metrics['positive_rate']:.6f}",
            f"sample_count: {metrics['sample_count']}",
            f"confusion_matrix: {metrics['confusion_matrix']}",
        ],
    )


def _format_optional_float(value: float | None) -> str:
    """Format nullable metrics."""
    return "null" if value is None else f"{value:.6f}"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset meta not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"dataset meta must be a JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"training JSONL not found: {path}")
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


def _resolve_feature_names(dataset_meta: Mapping[str, Any]) -> list[str]:
    """Resolve stable feature order from dataset meta or exporter fallback."""
    raw_feature_names = dataset_meta.get("feature_names") or list(FEATURE_NAMES)
    feature_names = [str(name) for name in raw_feature_names]
    if not feature_names:
        raise ValueError("feature_names must not be empty.")
    return feature_names


def _sample_to_xy(
    sample: Mapping[str, Any],
    feature_names: Sequence[str],
    line_number: int,
    path: Path,
) -> tuple[list[float], int]:
    """Validate one sample and return its feature row plus label."""
    for required_field in ("query_id", "product_id", "label", "features"):
        if required_field not in sample:
            raise ValueError(f"{path}:{line_number} missing required field {required_field}.")

    query_id = str(sample["query_id"])
    product_id = str(sample["product_id"])
    label = _parse_label(sample["label"], query_id=query_id, product_id=product_id)
    features = sample["features"]
    if not isinstance(features, Mapping):
        raise ValueError(f"sample {query_id}/{product_id} features must be an object.")

    row: list[float] = []
    for feature_name in feature_names:
        if feature_name not in features:
            raise ValueError(
                f"sample {query_id}/{product_id} missing feature {feature_name}",
            )
        try:
            row.append(float(features[feature_name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sample {query_id}/{product_id} feature {feature_name} "
                "must be numeric.",
            ) from exc
    return row, label


def _parse_label(value: Any, query_id: str, product_id: str) -> int:
    """Parse and validate a binary label."""
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample {query_id}/{product_id} label must be 0 or 1.") from exc
    if label not in {0, 1}:
        raise ValueError(f"sample {query_id}/{product_id} label must be 0 or 1.")
    return label


def _ensure_binary_train_labels(y_train: np.ndarray) -> None:
    """LogisticRegression requires both classes in the training split."""
    classes = set(int(value) for value in np.unique(y_train))
    if classes != {0, 1}:
        raise ValueError(
            "train labels must contain both 0 and 1 for LogisticRegression.",
        )


def _collect_warnings(
    train_metrics: Mapping[str, Any],
    valid_metrics: Mapping[str, Any],
) -> list[str]:
    """Collect split-level metric warnings into the top-level report."""
    warnings: list[str] = []
    warnings.extend(f"train: {warning}" for warning in train_metrics.get("warnings", []))
    warnings.extend(f"valid: {warning}" for warning in valid_metrics.get("warnings", []))
    return warnings


def _validate_training_args(model_type: str, max_iter: int) -> None:
    """Validate high-level training options."""
    if model_type != DEFAULT_MODEL_TYPE:
        raise ValueError("model_type must be logistic_regression.")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0.")


def _ensure_model_outputs_can_write(
    model_path: Path,
    model_meta_path: Path,
    overwrite: bool,
) -> None:
    """Fail before training when model outputs already exist."""
    if overwrite:
        return
    existing = [path for path in (model_path, model_meta_path) if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            "trained reranker output already exists; use --overwrite to replace: "
            f"{names}",
        )
