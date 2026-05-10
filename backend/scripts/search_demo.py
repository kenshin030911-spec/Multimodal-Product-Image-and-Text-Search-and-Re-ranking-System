"""离线检索 demo 脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许使用 python backend/scripts/search_demo.py 直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.retrieval.image_search import search_image_to_image
from backend.app.retrieval.text_search import search_text_to_image


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="离线 text-to-image / image-to-image 检索 demo。")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-text", default=None, help="文本查询。")
    query_group.add_argument("--query-image", type=Path, default=None, help="图片查询路径。")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量。")
    parser.add_argument("--index-dir", type=Path, default=None, help="index 目录，默认 data/index。")
    parser.add_argument(
        "--products-path",
        type=Path,
        default=None,
        help="products.jsonl 路径，默认 data/processed/products.jsonl。",
    )
    parser.add_argument("--exclude-product-id", default=None, help="图搜图时排除指定 product_id。")
    parser.add_argument(
        "--exclude-embedding-index",
        type=int,
        default=None,
        help="图搜图时排除指定 embedding index。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="真实 encoder 运行设备。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行离线检索 demo。"""
    args = build_parser().parse_args(argv)
    try:
        if args.query_text is not None:
            response = search_text_to_image(
                query_text=args.query_text,
                top_k=args.top_k,
                index_dir=args.index_dir,
                products_path=args.products_path,
                device=args.device,
            )
        else:
            response = search_image_to_image(
                query_image_path=args.query_image,
                top_k=args.top_k,
                index_dir=args.index_dir,
                products_path=args.products_path,
                exclude_product_id=args.exclude_product_id,
                exclude_embedding_index=args.exclude_embedding_index,
                device=args.device,
            )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"检索失败: {exc}", file=sys.stderr)
        return 1

    print(f"query_type: {response.query_type}")
    print(f"query: {response.query}")
    print(f"top_k: {response.top_k}")
    if response.missing_product_ids:
        print(f"missing_product_ids: {', '.join(response.missing_product_ids)}")
    print("rank\tscore\tproduct_id\ttitle\tarticle_type\tbase_colour\timage_path")
    for result in response.results:
        print(
            f"{result.final_rank}\t"
            f"{result.score:.6f}\t"
            f"{result.product_id}\t"
            f"{result.title}\t"
            f"{result.article_type or ''}\t"
            f"{result.base_colour or ''}\t"
            f"{result.image_path or ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
