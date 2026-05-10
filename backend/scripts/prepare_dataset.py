"""数据集准备脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许使用 python backend/scripts/prepare_dataset.py 直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.processor import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="准备 Fashion Product Images Small 商品元数据。",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="metadata CSV 路径，默认 data/raw/metadata/styles.csv。",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="图片目录，默认 data/raw/images/。",
    )
    parser.add_argument(
        "--products-path",
        type=Path,
        default=None,
        help="标准化商品输出路径，默认 data/processed/products.jsonl。",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="数据统计输出路径，默认 data/processed/dataset_stats.json。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行数据集准备，缺少真实数据时给出清晰错误。"""
    args = build_parser().parse_args(argv)
    try:
        stats = prepare_dataset(
            metadata_path=args.metadata_path,
            images_dir=args.images_dir,
            products_path=args.products_path,
            stats_path=args.stats_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"数据集准备失败: {exc}", file=sys.stderr)
        return 1

    print("数据集准备完成。")
    print(f"输出商品数: {stats['output_product_count']}")
    print(f"缺失图片数: {stats['missing_image_count']}")
    print(f"输出文件: {stats['processed_products_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
