"""检查 processed 商品数据的脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许使用 python backend/scripts/check_processed_data.py 直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.dataset_loader import default_data_check_report_path
from backend.app.data.processed_checker import (
    build_processed_data_report,
    format_processed_data_report,
    write_processed_data_report,
)


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="检查 data/processed/products.jsonl 是否适合后续向量生成。",
    )
    parser.add_argument(
        "--products-path",
        type=Path,
        default=None,
        help="products.jsonl 路径，默认 data/processed/products.jsonl。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON 报告路径，默认 outputs/data_checks/data_check_report.json。",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多检查多少条有效商品。")
    parser.add_argument("--sample-size", type=int, default=5, help="报告中展示的商品样本数。")
    parser.add_argument("--top-n", type=int, default=10, help="类别统计 Top-N。")
    parser.add_argument(
        "--image-check-size",
        type=int,
        default=20,
        help="抽样检查 image_path 的商品数量。",
    )
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子。")
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="写入 JSON 检查报告。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 processed 数据检查。"""
    args = build_parser().parse_args(argv)
    report_path = args.report_path or default_data_check_report_path()
    report = build_processed_data_report(
        products_path=args.products_path,
        report_path=report_path,
        limit=args.limit,
        sample_size=args.sample_size,
        top_n=args.top_n,
        image_check_size=args.image_check_size,
        seed=args.seed,
    )

    print(format_processed_data_report(report))
    if args.write_json:
        write_processed_data_report(report, report_path)
        print(f"\nJSON report written to: {report['report_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
