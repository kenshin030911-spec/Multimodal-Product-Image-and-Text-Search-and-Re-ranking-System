"""索引构建脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许使用 python backend/scripts/build_index.py 直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.index.index_builder import SUPPORTED_METRIC, build_index


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="构建 NumPy flat cosine image index。")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="embedding 输入目录，默认 data/embeddings。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="index 输出目录，默认 data/index。",
    )
    parser.add_argument(
        "--metric",
        default=SUPPORTED_METRIC,
        help="相似度指标，第一版只支持 cosine。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 index 输出。")
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 index 构建。"""
    args = build_parser().parse_args(argv)
    try:
        result = build_index(
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            metric=args.metric,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"index 构建失败: {exc}", file=sys.stderr)
        return 1

    meta = result.bundle.meta
    print("index 构建完成。")
    print(f"index_type: {meta['index_type']}")
    print(f"metric: {meta['metric']}")
    print(f"embedding_source: {meta['embedding_source']}")
    print(f"embedding_dim: {meta['embedding_dim']}")
    print(f"product_count: {meta['product_count']}")
    print(f"image_index_file: {meta['image_index_file']}")
    print(f"index_meta_file: {meta['index_meta_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
