"""Analyze trained reranker evaluation cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution with python backend/scripts/analyze_evaluation_cases.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.evaluation.case_analysis import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_N,
    analyze_evaluation_cases,
    default_details_path,
    default_output_dir,
    default_summary_path,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the case-analysis CLI parser."""
    parser = argparse.ArgumentParser(
        description="Analyze trained reranker stability from offline evaluation reports.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=default_summary_path(),
        help="evaluation_summary.json path; defaults to outputs/eval_reports/evaluation_summary.json.",
    )
    parser.add_argument(
        "--details-path",
        type=Path,
        default=default_details_path(),
        help="evaluation_details.jsonl path; defaults to outputs/eval_reports/evaluation_details.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="output directory; defaults to outputs/eval_reports.",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run case analysis and print the core recommendation."""
    args = build_parser().parse_args(argv)
    try:
        result = analyze_evaluation_cases(
            summary_path=args.summary_path,
            details_path=args.details_path,
            output_dir=args.output_dir,
            top_n=args.top_n,
            threshold=args.threshold,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"case analysis failed: {exc}", file=sys.stderr)
        return 1

    stats = result.report["summary_stats"]
    counts = result.report["counts_by_category"]
    improved_ids = [case["query_id"] for case in result.report["top_improved_cases"]]
    degraded_ids = [case["query_id"] for case in result.report["top_degraded_cases"]]
    print("rerank case analysis completed.")
    print(f"stability_label: {stats['stability_label']}")
    print(f"recommendation: {stats['recommendation']}")
    print("counts_by_category:")
    for category, count in counts.items():
        print(f"- {category}: {count}")
    print(f"top_improved_query_ids: {improved_ids}")
    print(f"top_degraded_query_ids: {degraded_ids}")
    print("Output files:")
    for label, path in result.output_paths.items():
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
