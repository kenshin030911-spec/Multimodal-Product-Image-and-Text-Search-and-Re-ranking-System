"""Offline evaluation script entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution with python backend/scripts/run_evaluation.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.eval_runner import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_MAX_QUERY_VARIANTS,
    DEFAULT_MAX_QUERIES,
    DEFAULT_METRIC_K,
    DEFAULT_QUERIES_PER_PRODUCT,
    DEFAULT_QUERY_TEMPLATES,
    DEFAULT_SEED,
    default_pairwise_meta_path,
    default_pairwise_model_path,
    default_trained_meta_path,
    default_trained_model_path,
    run_evaluation,
)
from backend.app.evaluation.report_writer import format_text_summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for offline retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Run offline weak-supervised text retrieval evaluation.",
    )
    parser.add_argument(
        "--eval-queries-path",
        type=Path,
        default=None,
        help="eval_queries.jsonl path; defaults to data/processed/eval_queries.jsonl.",
    )
    parser.add_argument(
        "--products-path",
        type=Path,
        default=None,
        help="products.jsonl path; defaults to data/processed/products.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report output directory; defaults to outputs/eval_reports.",
    )
    parser.add_argument("--metric-k", type=int, default=DEFAULT_METRIC_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument(
        "--query-templates",
        choices=("basic", "augmented"),
        default=DEFAULT_QUERY_TEMPLATES,
        help="weak query generation mode used only when eval_queries.jsonl is absent.",
    )
    parser.add_argument(
        "--queries-per-product",
        type=int,
        default=DEFAULT_QUERIES_PER_PRODUCT,
        help="max generated queries per product in augmented mode.",
    )
    parser.add_argument(
        "--max-query-variants",
        type=int,
        default=DEFAULT_MAX_QUERY_VARIANTS,
        help="max rendered template variants considered per product.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="query encoder device for real FashionCLIP indexes.",
    )
    parser.add_argument(
        "--include-trained-reranker",
        action="store_true",
        help="include trained sklearn reranker in the offline ranking comparison.",
    )
    parser.add_argument(
        "--trained-model-path",
        type=Path,
        default=default_trained_model_path(),
        help="trained reranker joblib path; defaults to models/reranker/trained_reranker.joblib.",
    )
    parser.add_argument(
        "--trained-meta-path",
        type=Path,
        default=default_trained_meta_path(),
        help="trained reranker meta path; defaults to models/reranker/trained_reranker_meta.json.",
    )
    parser.add_argument(
        "--include-pairwise-reranker",
        action="store_true",
        help="include pairwise sklearn reranker in the offline ranking comparison.",
    )
    parser.add_argument(
        "--pairwise-model-path",
        type=Path,
        default=default_pairwise_model_path(),
        help="pairwise reranker joblib path; defaults to models/reranker_pairwise/pairwise_reranker.joblib.",
    )
    parser.add_argument(
        "--pairwise-meta-path",
        type=Path,
        default=default_pairwise_meta_path(),
        help="pairwise reranker meta path; defaults to models/reranker_pairwise/pairwise_reranker_meta.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run evaluation and print a readable summary plus report paths."""
    args = build_parser().parse_args(argv)
    try:
        result = run_evaluation(
            eval_queries_path=args.eval_queries_path,
            products_path=args.products_path,
            output_dir=args.output_dir,
            metric_k=args.metric_k,
            candidate_k=args.candidate_k,
            max_queries=args.max_queries,
            seed=args.seed,
            device=args.device,
            query_templates=args.query_templates,
            queries_per_product=args.queries_per_product,
            max_query_variants=args.max_query_variants,
            include_trained_reranker=args.include_trained_reranker,
            trained_model_path=args.trained_model_path,
            trained_meta_path=args.trained_meta_path,
            include_pairwise_reranker=args.include_pairwise_reranker,
            pairwise_model_path=args.pairwise_model_path,
            pairwise_meta_path=args.pairwise_meta_path,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(format_text_summary(result.summary), end="")
    print("Output files:")
    for label, path in result.report_paths.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
