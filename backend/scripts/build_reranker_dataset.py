"""Build weak-supervised reranker training dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution with python backend/scripts/build_reranker_dataset.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.training.sample_builder import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_MAX_QUERY_VARIANTS,
    DEFAULT_MAX_NEGATIVES_PER_QUERY,
    DEFAULT_MAX_POSITIVES_PER_QUERY,
    DEFAULT_MAX_QUERIES,
    DEFAULT_MIN_POSITIVES_PER_QUERY,
    DEFAULT_QUERIES_PER_PRODUCT,
    DEFAULT_QUERY_TEMPLATES,
    DEFAULT_SEED,
    DEFAULT_TRAIN_RATIO,
    build_reranker_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the reranker dataset builder CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build weak-supervised JSONL samples for reranker training.",
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
        help="output directory; defaults to data/processed.",
    )
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
    parser.add_argument(
        "--max-positives-per-query",
        type=int,
        default=DEFAULT_MAX_POSITIVES_PER_QUERY,
    )
    parser.add_argument(
        "--max-negatives-per-query",
        type=int,
        default=DEFAULT_MAX_NEGATIVES_PER_QUERY,
    )
    parser.add_argument(
        "--min-positives-per-query",
        type=int,
        default=DEFAULT_MIN_POSITIVES_PER_QUERY,
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="query encoder device for real FashionCLIP indexes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dataset builder and print output paths plus core stats."""
    args = build_parser().parse_args(argv)
    try:
        result = build_reranker_dataset(
            eval_queries_path=args.eval_queries_path,
            products_path=args.products_path,
            output_dir=args.output_dir,
            candidate_k=args.candidate_k,
            max_queries=args.max_queries,
            max_positives_per_query=args.max_positives_per_query,
            max_negatives_per_query=args.max_negatives_per_query,
            min_positives_per_query=args.min_positives_per_query,
            train_ratio=args.train_ratio,
            seed=args.seed,
            device=args.device,
            query_templates=args.query_templates,
            queries_per_product=args.queries_per_product,
            max_query_variants=args.max_query_variants,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"reranker dataset build failed: {exc}", file=sys.stderr)
        return 1

    meta = result.meta
    print("reranker dataset build completed.")
    print(f"query_count: {meta['query_count']}")
    print(f"used_query_count: {meta['used_query_count']}")
    print(f"train_sample_count: {meta['train_sample_count']}")
    print(f"valid_sample_count: {meta['valid_sample_count']}")
    print(f"positive_count: {meta['positive_count']}")
    print(f"negative_count: {meta['negative_count']}")
    print(f"positive_rate: {meta['positive_rate']:.6f}")
    print("Output files:")
    for label, path in result.output_paths.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
