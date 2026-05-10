"""数据集准备主流程。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import (
    default_dataset_stats_path,
    default_metadata_path,
    default_products_path,
    write_dataset_stats,
    write_products,
)
from backend.app.data.freshness import add_freshness_scores
from backend.app.data.metadata_loader import load_raw_metadata, standardize_metadata_row
from backend.app.data.validators import build_image_path, to_project_relative, validate_image_path
from backend.app.schemas.product import ProductItem


def prepare_dataset(
    metadata_path: Path | None = None,
    images_dir: Path | None = None,
    products_path: Path | None = None,
    stats_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """读取原始 metadata，输出标准化 products.jsonl 和 dataset_stats.json。"""
    settings = get_settings()
    project_root = project_root or settings.project_root
    metadata_path = metadata_path or default_metadata_path(settings)
    images_dir = images_dir or settings.raw_images_dir
    products_path = products_path or default_products_path(settings)
    stats_path = stats_path or default_dataset_stats_path(settings)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata 文件不存在，请放置到 {to_project_relative(metadata_path, project_root)}"
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"图片目录不存在，请放置到 {to_project_relative(images_dir, project_root)}"
        )

    rows = load_raw_metadata(metadata_path)
    stats = _initial_stats(metadata_path, images_dir, products_path, project_root, len(rows))
    candidate_products: list[dict[str, Any]] = []

    for row in rows:
        try:
            standardized = standardize_metadata_row(row)
        except Exception:
            stats["skipped_invalid_row_count"] += 1
            continue

        if standardized.product_data is None:
            if standardized.skipped_reason == "missing_id":
                stats["skipped_missing_id_count"] += 1
            else:
                stats["skipped_invalid_id_count"] += 1
            continue

        if standardized.missing_year:
            stats["missing_year_count"] += 1
        if standardized.invalid_year:
            stats["invalid_year_count"] += 1

        product_id = standardized.product_data["product_id"]
        image_path = build_image_path(images_dir, product_id)
        if not validate_image_path(image_path):
            stats["missing_image_count"] += 1
            stats["skipped_missing_image_count"] += 1
            continue

        standardized.product_data["image_path"] = to_project_relative(image_path, project_root)
        candidate_products.append(standardized.product_data)

    products = _validate_products(add_freshness_scores(candidate_products), stats)
    stats["product_count"] = len(products)
    stats["output_product_count"] = len(products)
    stats["valid_image_count"] = len(products)
    stats["prepared"] = len(products) > 0
    stats["message"] = (
        "数据集准备完成。"
        if stats["prepared"]
        else "未生成有效商品，请检查 metadata 和图片目录。"
    )

    write_products(products, products_path)
    write_dataset_stats(stats, stats_path)
    return stats


def _validate_products(
    product_dicts: list[dict[str, Any]],
    stats: dict[str, Any],
) -> list[ProductItem]:
    """用 ProductItem schema 校验每个输出商品。"""
    products: list[ProductItem] = []
    for product_data in product_dicts:
        try:
            products.append(ProductItem.model_validate(product_data))
        except ValidationError:
            stats["skipped_invalid_row_count"] += 1
    return products


def _initial_stats(
    metadata_path: Path,
    images_dir: Path,
    products_path: Path,
    project_root: Path,
    raw_row_count: int,
) -> dict[str, Any]:
    """创建统计字段，路径只保存相对路径。"""
    return {
        "prepared": False,
        "metadata_path": to_project_relative(metadata_path, project_root),
        "images_dir": to_project_relative(images_dir, project_root),
        "processed_products_path": to_project_relative(products_path, project_root),
        "raw_row_count": raw_row_count,
        "product_count": 0,
        "output_product_count": 0,
        "valid_image_count": 0,
        "missing_image_count": 0,
        "skipped_missing_image_count": 0,
        "skipped_missing_id_count": 0,
        "skipped_invalid_id_count": 0,
        "skipped_invalid_row_count": 0,
        "missing_year_count": 0,
        "invalid_year_count": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "message": "数据集准备中。",
    }
