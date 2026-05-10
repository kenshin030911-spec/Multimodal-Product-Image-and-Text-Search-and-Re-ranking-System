"""商品元数据读取和字段标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.data.validators import normalize_product_id


FIELD_ALIASES = {
    "id": ("id", "product_id", "productId"),
    "title": ("productDisplayName", "title", "product_display_name"),
    "gender": ("gender",),
    "master_category": ("masterCategory", "master_category"),
    "sub_category": ("subCategory", "sub_category"),
    "article_type": ("articleType", "article_type"),
    "base_colour": ("baseColour", "base_colour", "baseColor"),
    "season": ("season",),
    "year": ("year",),
    "usage": ("usage",),
}


@dataclass(frozen=True)
class StandardizedRow:
    """单行清洗结果和统计标记。"""

    product_data: dict[str, Any] | None
    skipped_reason: str | None = None
    missing_year: bool = False
    invalid_year: bool = False


def load_raw_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """稳健读取原始 metadata CSV，少量坏行由 pandas 跳过。"""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata 文件不存在: {metadata_path}")

    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(
                metadata_path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="skip",
            )
            return frame.to_dict(orient="records")
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except pd.errors.ParserError as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(f"无法读取 metadata CSV，已尝试多种编码: {'; '.join(errors)}")


def parse_year(value: object, current_year: int | None = None) -> tuple[int | None, str | None]:
    """解析年份，异常年份不会跳过商品，只返回错误标记。"""
    current_year = current_year or datetime.now().year
    if _is_missing(value):
        return None, "missing"

    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None, "invalid"

    if number != number.to_integral_value():
        return None, "invalid"

    year = int(number)
    if year < 1900 or year > current_year:
        return None, "invalid"

    return year, None


def standardize_metadata_row(row: dict[str, Any]) -> StandardizedRow:
    """把一行原始 metadata 转成 ProductItem 可用的字段。"""
    product_id, id_error = normalize_product_id(_get_value(row, FIELD_ALIASES["id"]))
    if product_id is None:
        return StandardizedRow(product_data=None, skipped_reason=f"{id_error}_id")

    article_type = _clean_text(_get_value(row, FIELD_ALIASES["article_type"]))
    sub_category = _clean_text(_get_value(row, FIELD_ALIASES["sub_category"]))
    title = _clean_text(_get_value(row, FIELD_ALIASES["title"])) or article_type or sub_category or ""
    year, year_error = parse_year(_get_value(row, FIELD_ALIASES["year"]))

    product_data = {
        "product_id": product_id,
        "title": title,
        "gender": _clean_text(_get_value(row, FIELD_ALIASES["gender"])),
        "master_category": _clean_text(_get_value(row, FIELD_ALIASES["master_category"])),
        "sub_category": sub_category,
        "article_type": article_type,
        "base_colour": _clean_text(_get_value(row, FIELD_ALIASES["base_colour"])),
        "season": _clean_text(_get_value(row, FIELD_ALIASES["season"])),
        "year": year,
        "usage": _clean_text(_get_value(row, FIELD_ALIASES["usage"])),
        "image_path": None,
        "freshness_score": 0.5,
    }

    return StandardizedRow(
        product_data=product_data,
        missing_year=year_error == "missing",
        invalid_year=year_error == "invalid",
    )


def _get_value(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """按多个可能字段名读取值，兼容不同数据来源。"""
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in aliases:
        value = normalized.get(alias.strip().lower())
        if value is not None:
            return value
    return None


def _clean_text(value: object) -> str | None:
    """清洗文本字段，空值统一为 None。"""
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: object) -> bool:
    """兼容 pandas 的 NaN/NA 和常见空字符串。"""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}
