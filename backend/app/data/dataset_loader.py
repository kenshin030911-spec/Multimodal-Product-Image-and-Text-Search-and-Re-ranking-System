"""标准化商品数据读写。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.product import ProductItem


@dataclass(frozen=True)
class ProductLoadResult:
    """products.jsonl 加载结果和逐行错误统计。"""

    products: list[ProductItem] = field(default_factory=list)
    file_exists: bool = False
    total_lines: int = 0
    loaded_count: int = 0
    skipped_empty_line_count: int = 0
    skipped_invalid_json_count: int = 0
    skipped_validation_error_count: int = 0
    message: str = ""

    def to_stats(self) -> dict[str, Any]:
        """转成可写入检查报告的普通字典。"""
        return {
            "file_exists": self.file_exists,
            "total_lines": self.total_lines,
            "loaded_count": self.loaded_count,
            "skipped_empty_line_count": self.skipped_empty_line_count,
            "skipped_invalid_json_count": self.skipped_invalid_json_count,
            "skipped_validation_error_count": self.skipped_validation_error_count,
            "message": self.message,
        }


def default_metadata_path(settings: Settings | None = None) -> Path:
    """返回默认 metadata 文件路径。"""
    settings = settings or get_settings()
    return settings.raw_metadata_dir / "styles.csv"


def default_products_path(settings: Settings | None = None) -> Path:
    """返回默认标准化商品 JSONL 输出路径。"""
    settings = settings or get_settings()
    return settings.processed_data_dir / "products.jsonl"


def default_dataset_stats_path(settings: Settings | None = None) -> Path:
    """返回默认数据统计 JSON 输出路径。"""
    settings = settings or get_settings()
    return settings.processed_data_dir / "dataset_stats.json"


def default_data_check_report_path(settings: Settings | None = None) -> Path:
    """返回默认 processed 数据检查报告路径。"""
    settings = settings or get_settings()
    return settings.output_dir / "data_checks" / "data_check_report.json"


def load_products(
    products_path: Path | None = None,
    limit: int | None = None,
) -> list[ProductItem]:
    """从 products.jsonl 读取标准化商品；保持第一轮调用方式兼容。"""
    return load_products_with_stats(products_path=products_path, limit=limit).products


def load_products_with_stats(
    products_path: Path | None = None,
    limit: int | None = None,
    skip_invalid: bool = True,
) -> ProductLoadResult:
    """逐行读取 products.jsonl，跳过坏 JSON 行和 schema 错误行。"""
    path = products_path or default_products_path()
    if not path.is_file():
        return ProductLoadResult(
            file_exists=False,
            message=f"products.jsonl 不存在: {path}",
        )

    if limit is not None and limit < 0:
        raise ValueError("limit 必须大于或等于 0。")

    products: list[ProductItem] = []
    total_lines = 0
    skipped_empty_line_count = 0
    skipped_invalid_json_count = 0
    skipped_validation_error_count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(products) >= limit:
                break

            total_lines += 1
            stripped = line.strip()
            if not stripped:
                skipped_empty_line_count += 1
                continue

            try:
                raw_product = json.loads(stripped)
            except JSONDecodeError:
                skipped_invalid_json_count += 1
                if skip_invalid:
                    continue
                raise

            try:
                product = ProductItem.model_validate(raw_product)
            except ValidationError:
                skipped_validation_error_count += 1
                if skip_invalid:
                    continue
                raise

            products.append(product)

    return ProductLoadResult(
        products=products,
        file_exists=True,
        total_lines=total_lines,
        loaded_count=len(products),
        skipped_empty_line_count=skipped_empty_line_count,
        skipped_invalid_json_count=skipped_invalid_json_count,
        skipped_validation_error_count=skipped_validation_error_count,
        message="products.jsonl 加载完成。",
    )


def write_products(products: list[ProductItem], products_path: Path | None = None) -> None:
    """逐行写入标准化商品，避免一次性拼接大文件。"""
    path = products_path or default_products_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for product in products:
            file.write(product.model_dump_json())
            file.write("\n")


def load_dataset_stats(stats_path: Path | None = None) -> dict[str, Any] | None:
    """读取 dataset_stats.json；不存在时返回 None。"""
    path = stats_path or default_dataset_stats_path()
    return load_json_file(path)


def load_json_file(path: Path) -> dict[str, Any] | None:
    """读取 JSON 文件；不存在或格式错误时返回 None。"""
    if not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (JSONDecodeError, OSError):
        return None

    if not isinstance(data, dict):
        return None
    return data


def write_dataset_stats(stats: dict[str, Any], stats_path: Path | None = None) -> None:
    """写入数据集统计文件，供 API 快速读取。"""
    path = stats_path or default_dataset_stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")
