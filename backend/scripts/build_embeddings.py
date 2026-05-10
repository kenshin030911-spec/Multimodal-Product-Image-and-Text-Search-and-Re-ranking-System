"""embedding 生成脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 允许使用 python backend/scripts/build_embeddings.py 直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.embedding.embedding_builder import build_embeddings
from backend.app.embedding.versioning import (
    DEFAULT_DUMMY_ENCODER_NAME,
    DEFAULT_REAL_MODEL_NAME,
    SUPPORTED_ENCODER_NAMES,
)


def _parse_bool(raw_value: str | bool) -> bool:
    """解析 CLI 布尔值。"""
    if isinstance(raw_value, bool):
        return raw_value
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("布尔值只支持 true/false、1/0、yes/no、on/off。")


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="生成图片/文本 embedding；默认 dummy，真实 CLIP/FashionCLIP 需显式启用。",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多读取多少条商品。")
    parser.add_argument("--batch-size", type=int, default=16, help="批量编码大小。")
    parser.add_argument(
        "--encoder-name",
        default=DEFAULT_DUMMY_ENCODER_NAME,
        choices=SUPPORTED_ENCODER_NAMES,
        help="编码器名称，默认 dummy；真实模型可选 fashion-clip 或 clip。",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_REAL_MODEL_NAME,
        help="Transformers 模型名，仅真实 encoder 使用。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="真实 encoder 运行设备：auto 优先 CUDA，否则 CPU。",
    )
    parser.add_argument(
        "--normalize",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="是否对真实 embedding 做 L2 normalize，默认 true。",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="关闭真实 embedding L2 normalize。",
    )
    parser.add_argument("--image-only", action="store_true", help="只生成图片 embedding。")
    parser.add_argument("--text-only", action="store_true", help="只生成文本 embedding。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 embedding 输出。")
    parser.add_argument(
        "--products-path",
        type=Path,
        default=None,
        help="products.jsonl 路径，默认 data/processed/products.jsonl。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="embedding 输出目录，默认 data/embeddings/。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 dummy embedding 生成。"""
    args = build_parser().parse_args(argv)
    try:
        result = build_embeddings(
            products_path=args.products_path,
            output_dir=args.output_dir,
            limit=args.limit,
            batch_size=args.batch_size,
            encoder_name=args.encoder_name,
            model_name=args.model_name,
            device=args.device,
            normalize=args.normalize,
            image_only=args.image_only,
            text_only=args.text_only,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ImportError, RuntimeError, ValueError) as exc:
        print(f"embedding 生成失败: {exc}", file=sys.stderr)
        return 1

    meta = result.bundle.meta
    print("embedding 生成完成。")
    print(f"encoder: {meta['encoder_name']} ({meta['encoder_version']})")
    print(f"framework: {meta.get('framework')}")
    print(f"model_name: {meta.get('model_name')}")
    print(f"device: {meta.get('device')}")
    print(f"torch_dtype: {meta.get('torch_dtype')}")
    print(f"normalize_embeddings: {meta.get('normalize_embeddings')}")
    print(f"embedding_dim: {meta['embedding_dim']}")
    print(f"product_count: {meta['product_count']}")
    print(f"generated: image={meta['generated']['image']}, text={meta['generated']['text']}")
    print(f"failed_images: {len(meta['failed_images'])}")
    print(f"failed_texts: {len(meta['failed_texts'])}")
    print(f"skipped_products: {len(meta['skipped_products'])}")
    print(f"output_dir: {meta['output_dir']}")
    print(f"meta_file: {meta['meta_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
