"""数据集内部相对新鲜度计算。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def add_freshness_scores(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为商品写入 freshness_score，不参与本轮搜索或排序。"""
    global_years = _valid_years(products)
    article_years = _group_years(products, "article_type")
    sub_category_years = _group_years(products, "sub_category")

    output: list[dict[str, Any]] = []
    for product in products:
        item = dict(product)
        year = item.get("year")
        if not isinstance(year, int):
            item["freshness_score"] = 0.5
            output.append(item)
            continue

        group_years = _choose_group_years(item, article_years, sub_category_years, global_years)
        item["freshness_score"] = _normalize_year(year, group_years, global_years)
        output.append(item)

    return output


def _choose_group_years(
    product: dict[str, Any],
    article_years: dict[str, list[int]],
    sub_category_years: dict[str, list[int]],
    global_years: list[int],
) -> list[int]:
    """按 README 规则选择 article_type、sub_category 或全局年份组。"""
    article_type = product.get("article_type")
    if article_type:
        return article_years.get(str(article_type), [])

    sub_category = product.get("sub_category")
    if sub_category:
        return sub_category_years.get(str(sub_category), [])

    return global_years


def _group_years(products: list[dict[str, Any]], key: str) -> dict[str, list[int]]:
    """收集每个分组里的有效年份。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for product in products:
        group_value = product.get(key)
        year = product.get("year")
        if group_value and isinstance(year, int):
            groups[str(group_value)].append(year)
    return groups


def _valid_years(products: list[dict[str, Any]]) -> list[int]:
    """提取所有有效年份。"""
    return [product["year"] for product in products if isinstance(product.get("year"), int)]


def _normalize_year(year: int, group_years: list[int], global_years: list[int]) -> float:
    """按组内年份归一化计算相对新鲜度。"""
    years = group_years
    if len(years) < 2:
        years = global_years
    if len(years) < 2:
        return 0.5

    min_year = min(years)
    max_year = max(years)
    if max_year == min_year:
        return 0.5

    score = (year - min_year) / (max_year - min_year)
    return max(0.0, min(1.0, float(score)))
