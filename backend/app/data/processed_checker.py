"""processed 商品数据检查。"""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import (
    default_data_check_report_path,
    default_products_path,
    load_products_with_stats,
)
from backend.app.data.validators import to_project_relative
from backend.app.schemas.product import ProductItem


MISSING_CHECK_FIELDS = (
    "title",
    "gender",
    "master_category",
    "sub_category",
    "article_type",
    "base_colour",
    "season",
    "year",
    "usage",
    "image_path",
)
TOP_VALUE_FIELDS = ("article_type", "sub_category", "base_colour", "gender")


def build_processed_data_report(
    products_path: Path | None = None,
    report_path: Path | None = None,
    limit: int | None = None,
    sample_size: int = 5,
    top_n: int = 10,
    image_check_size: int = 20,
    seed: int = 42,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """读取 products.jsonl 并生成轻量检查报告。"""
    settings = get_settings()
    project_root = project_root or settings.project_root
    products_path = products_path or default_products_path(settings)
    report_path = report_path or default_data_check_report_path(settings)

    load_result = load_products_with_stats(products_path=products_path, limit=limit)
    products = load_result.products
    rng = random.Random(seed)

    image_check_products = _sample_products(products, image_check_size, rng)
    image_path_check = _check_image_paths(image_check_products, project_root)

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "products_path": to_project_relative(products_path, project_root),
        "report_path": to_project_relative(report_path, project_root),
        "limit": limit,
        "product_count": len(products),
        "load_stats": load_result.to_stats(),
        "missing_field_counts": _missing_field_counts(products),
        "top_values": _top_values(products, top_n),
        "year_stats": _year_stats(products),
        "freshness_score_stats": _freshness_score_stats(products),
        "image_path_check": image_path_check,
        "sample_products": [
            product.model_dump()
            for product in _sample_products(products, sample_size, rng)
        ],
        "message": "processed 数据检查完成。" if load_result.file_exists else load_result.message,
    }


def format_processed_data_report(report: dict[str, Any]) -> str:
    """把结构化检查报告格式化成人类可读文本。"""
    load_stats = report.get("load_stats", {})
    year_stats = report.get("year_stats", {})
    freshness_stats = report.get("freshness_score_stats", {})
    image_check = report.get("image_path_check", {})
    top_values = report.get("top_values", {})

    lines = [
        "Processed Data Check",
        f"products_path: {report.get('products_path', '')}",
        f"product_count: {report.get('product_count', 0)}",
        f"total_lines_read: {load_stats.get('total_lines', 0)}",
        f"loaded_count: {load_stats.get('loaded_count', 0)}",
        f"skipped_invalid_json_count: {load_stats.get('skipped_invalid_json_count', 0)}",
        f"skipped_validation_error_count: {load_stats.get('skipped_validation_error_count', 0)}",
        f"year_range: {year_stats.get('min_year')} - {year_stats.get('max_year')}",
        f"freshness_score_range: {freshness_stats.get('min_score')} - {freshness_stats.get('max_score')}",
        f"image_sample_missing_count: {image_check.get('missing_count', 0)}",
        "",
        "Top Values:",
    ]

    for field_name, values in top_values.items():
        formatted = ", ".join(f"{item['value']}={item['count']}" for item in values)
        lines.append(f"- {field_name}: {formatted}")

    lines.extend(["", "Missing Field Counts:"])
    for field_name, count in report.get("missing_field_counts", {}).items():
        lines.append(f"- {field_name}: {count}")

    return "\n".join(lines)


def write_processed_data_report(report: dict[str, Any], report_path: Path | None = None) -> None:
    """写入 JSON 检查报告。"""
    path = report_path or default_data_check_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _sample_products(
    products: list[ProductItem],
    sample_size: int,
    rng: random.Random,
) -> list[ProductItem]:
    """固定随机种子抽样，保证报告和测试稳定。"""
    if sample_size <= 0 or not products:
        return []
    if len(products) <= sample_size:
        return list(products)
    return rng.sample(products, sample_size)


def _missing_field_counts(products: list[ProductItem]) -> dict[str, int]:
    """统计核心字段缺失数量。"""
    counts: dict[str, int] = {}
    for field_name in MISSING_CHECK_FIELDS:
        counts[field_name] = sum(
            1 for product in products if _is_missing(getattr(product, field_name))
        )
    return counts


def _top_values(products: list[ProductItem], top_n: int) -> dict[str, list[dict[str, Any]]]:
    """统计指定字段的 Top-N 值。"""
    result: dict[str, list[dict[str, Any]]] = {}
    for field_name in TOP_VALUE_FIELDS:
        counter = Counter(
            str(value)
            for product in products
            if not _is_missing(value := getattr(product, field_name))
        )
        result[field_name] = [
            {"value": value, "count": count}
            for value, count in counter.most_common(max(top_n, 0))
        ]
    return result


def _year_stats(products: list[ProductItem]) -> dict[str, int | None]:
    """统计年份范围。"""
    years = [product.year for product in products if isinstance(product.year, int)]
    return {
        "valid_count": len(years),
        "missing_count": sum(1 for product in products if product.year is None),
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
    }


def _freshness_score_stats(products: list[ProductItem]) -> dict[str, float | int | None]:
    """统计 freshness_score 范围和越界数量。"""
    scores = [product.freshness_score for product in products]
    return {
        "valid_count": len(scores),
        "missing_count": 0,
        "out_of_range_count": sum(1 for score in scores if score < 0.0 or score > 1.0),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }


def _check_image_paths(
    products: list[ProductItem],
    project_root: Path,
) -> dict[str, Any]:
    """抽样检查 image_path 是否能解析到存在的本地文件。"""
    missing_samples: list[dict[str, str]] = []
    existing_count = 0

    for product in products:
        if not product.image_path:
            missing_samples.append({"product_id": product.product_id, "image_path": ""})
            continue

        image_path = Path(product.image_path)
        if not image_path.is_absolute():
            image_path = project_root / image_path

        if image_path.is_file():
            existing_count += 1
        else:
            missing_samples.append(
                {
                    "product_id": product.product_id,
                    "image_path": product.image_path,
                }
            )

    checked_count = len(products)
    return {
        "checked_count": checked_count,
        "existing_count": existing_count,
        "missing_count": len(missing_samples),
        "missing_samples": missing_samples,
    }


def _is_missing(value: object) -> bool:
    """判断 ProductItem 字段是否缺失。"""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False
