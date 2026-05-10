"""Train the pairwise logistic ranking reranker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution with python backend/scripts/train_pairwise_reranker.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.training.pairwise_trainer import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_PAIRS_PER_QUERY,
    DEFAULT_MIN_GRADE_GAP,
    DEFAULT_PAIR_SAMPLING_STRATEGY,
    DEFAULT_SEED,
    DEFAULT_SOLVER,
    default_dataset_meta_path,
    default_train_path,
    default_valid_path,
    train_pairwise_reranker,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the pairwise reranker training CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a pairwise sklearn LogisticRegression ranking reranker.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=default_train_path(),
        help=(
            "train JSONL path; defaults to "
            "data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/"
            "reranker_train.jsonl."
        ),
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=default_valid_path(),
        help=(
            "valid JSONL path; defaults to "
            "data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/"
            "reranker_valid.jsonl."
        ),
    )
    parser.add_argument(
        "--dataset-meta-path",
        type=Path,
        default=default_dataset_meta_path(),
        help=(
            "dataset meta JSON path; defaults to "
            "data/processed/reranker_dataset_aug_q500_c150_pos20_neg40/"
            "reranker_dataset_meta.json."
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "reranker_pairwise",
        help="directory for pairwise_reranker.joblib and pairwise_reranker_meta.json.",
    )
    parser.add_argument(
        "--report-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "training_pairwise",
        help="directory for pairwise_training_report.json/txt.",
    )
    parser.add_argument(
        "--max-pairs-per-query",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_QUERY,
    )
    parser.add_argument("--min-grade-gap", type=int, default=DEFAULT_MIN_GRADE_GAP)
    parser.add_argument(
        "--pair-sampling-strategy",
        choices=("hard", "random"),
        default=DEFAULT_PAIR_SAMPLING_STRATEGY,
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
    """Run pairwise training and print core validation metrics."""
    args = build_parser().parse_args(argv)
    try:
        result = train_pairwise_reranker(
            train_path=args.train_path,
            valid_path=args.valid_path,
            dataset_meta_path=args.dataset_meta_path,
            model_output_dir=args.model_output_dir,
            report_output_dir=args.report_output_dir,
            max_pairs_per_query=args.max_pairs_per_query,
            min_grade_gap=args.min_grade_gap,
            pair_sampling_strategy=args.pair_sampling_strategy,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            solver=args.solver,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"pairwise reranker training failed: {exc}", file=sys.stderr)
        return 1

    report = result.report
    metrics = report["metrics"]
    valid_pairwise = metrics["valid_pairwise"]
    valid_ranking = metrics["valid_ranking"]
    pairwise_ranking = valid_ranking["valid_pairwise_ranking_metrics"]
    delta = valid_ranking["pairwise_vs_recall_delta"]
    print("pairwise reranker training completed.")
    print(f"train_query_count: {report['train_query_count']}")
    print(f"valid_query_count: {report['valid_query_count']}")
    print(f"train_item_count: {report['train_item_count']}")
    print(f"valid_item_count: {report['valid_item_count']}")
    print(f"pairwise_train_pair_count: {report['pairwise_train_pair_count']}")
    print(f"pairwise_valid_pair_count: {report['pairwise_valid_pair_count']}")
    print(f"valid_pairwise_accuracy: {valid_pairwise['pairwise_accuracy']:.6f}")
    print(f"valid_pairwise_f1: {valid_pairwise['pairwise_f1']:.6f}")
    print(f"valid_pairwise_ndcg_at_10: {pairwise_ranking.get('ndcg_at_k', 0.0):.6f}")
    print(f"pairwise_vs_recall_ndcg_at_10: {delta.get('ndcg_at_k', 0.0):.6f}")
    print("Output files:")
    print(f"- model_path: {result.output_paths['model_path']}")
    print(f"- model_meta_path: {result.output_paths['model_meta_path']}")
    print(f"- report_json_path: {result.output_paths['report_json_path']}")
    print(f"- report_txt_path: {result.output_paths['report_txt_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
