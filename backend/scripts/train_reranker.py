"""Train the lightweight sklearn reranker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution with python backend/scripts/train_reranker.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.training.reranker_trainer import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_ITER,
    DEFAULT_MODEL_TYPE,
    DEFAULT_SEED,
    DEFAULT_SOLVER,
    default_dataset_meta_path,
    default_train_path,
    default_valid_path,
    train_reranker_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the reranker training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a lightweight sklearn LogisticRegression reranker.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=default_train_path(),
        help=(
            "train JSONL path; defaults to "
            "data/processed/reranker_dataset_q300_c150_pos20_neg40/reranker_train.jsonl."
        ),
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=default_valid_path(),
        help=(
            "valid JSONL path; defaults to "
            "data/processed/reranker_dataset_q300_c150_pos20_neg40/reranker_valid.jsonl."
        ),
    )
    parser.add_argument(
        "--dataset-meta-path",
        type=Path,
        default=default_dataset_meta_path(),
        help=(
            "dataset meta JSON path; defaults to "
            "data/processed/reranker_dataset_q300_c150_pos20_neg40/"
            "reranker_dataset_meta.json."
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "reranker",
        help="directory for trained_reranker.joblib and trained_reranker_meta.json.",
    )
    parser.add_argument(
        "--report-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training",
        help="directory for reranker_training_report.json/txt.",
    )
    parser.add_argument(
        "--model-type",
        choices=(DEFAULT_MODEL_TYPE,),
        default=DEFAULT_MODEL_TYPE,
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default=DEFAULT_CLASS_WEIGHT,
    )
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="overwrite model outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run training and print the core validation summary."""
    args = build_parser().parse_args(argv)
    try:
        result = train_reranker_model(
            train_path=args.train_path,
            valid_path=args.valid_path,
            dataset_meta_path=args.dataset_meta_path,
            model_output_dir=args.model_output_dir,
            report_output_dir=args.report_output_dir,
            model_type=args.model_type,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            solver=args.solver,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"reranker training failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    valid_metrics = report["valid_metrics"]
    print("reranker training completed.")
    print(f"train_sample_count: {report['train_sample_count']}")
    print(f"valid_sample_count: {report['valid_sample_count']}")
    print(f"train_positive_rate: {report['train_positive_rate']:.6f}")
    print(f"valid_positive_rate: {report['valid_positive_rate']:.6f}")
    print(f"valid_precision: {valid_metrics['precision']:.6f}")
    print(f"valid_recall: {valid_metrics['recall']:.6f}")
    print(f"valid_f1: {valid_metrics['f1']:.6f}")
    print(f"valid_roc_auc: {_format_metric(valid_metrics['roc_auc'])}")
    print(f"valid_average_precision: {_format_metric(valid_metrics['average_precision'])}")
    print("Output files:")
    print(f"- model_path: {result.output_paths['model_path']}")
    print(f"- model_meta_path: {result.output_paths['model_meta_path']}")
    print(f"- report_json_path: {result.output_paths['report_json_path']}")
    print(f"- report_txt_path: {result.output_paths['report_txt_path']}")
    return 0


def _format_metric(value: float | None) -> str:
    """Format nullable CLI metric values."""
    return "null" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
