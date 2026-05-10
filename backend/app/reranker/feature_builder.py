"""规则重排特征构造。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from backend.app.retrieval.candidate_builder import RetrievalResult
from backend.app.schemas.rerank import RerankFeature as SchemaRerankFeature


TOKEN_SPLIT_RE = re.compile(r"[\s\-_/,.;:()\\[\]{}]+")

ARTICLE_TYPE_MATCH_WEIGHT = 0.35
COLOR_MATCH_WEIGHT = 0.25
TITLE_MATCH_WEIGHT = 0.15
GENDER_MATCH_WEIGHT = 0.10
USAGE_MATCH_WEIGHT = 0.10
SUB_CATEGORY_MATCH_WEIGHT = 0.05


@dataclass(frozen=True)
class RerankFeature:
    """规则 reranker 的内部特征，不限制 recall_score 范围。"""

    product_id: str
    recall_score: float
    freshness_score: float
    title_match: float
    article_type_match: float
    color_match: float
    gender_match: float
    usage_match: float
    sub_category_match: float
    text_match_score: float
    metadata_match_score: float


def build_rerank_feature(result: RetrievalResult, query_text: str | None = None) -> RerankFeature:
    """从单条召回结果构造规则重排特征。"""
    recall_score = float(result.score)
    freshness_score = float(result.freshness_score)
    if query_text is None or not query_text.strip():
        return RerankFeature(
            product_id=result.product_id,
            recall_score=recall_score,
            freshness_score=freshness_score,
            title_match=0.0,
            article_type_match=0.0,
            color_match=0.0,
            gender_match=0.0,
            usage_match=0.0,
            sub_category_match=0.0,
            text_match_score=0.0,
            metadata_match_score=0.0,
        )

    query_tokens = _tokenize(query_text)
    title_match = _overlap_ratio(query_tokens, _tokenize(result.title))
    article_type_match = _field_match(query_text, query_tokens, result.article_type)
    color_match = _field_match(query_text, query_tokens, result.base_colour)
    gender_match = _field_match(query_text, query_tokens, result.gender)
    usage_match = _field_match(query_text, query_tokens, result.usage)
    sub_category_match = _field_match(query_text, query_tokens, result.sub_category)
    metadata_match_score = calculate_metadata_match_score(
        title_match=title_match,
        article_type_match=article_type_match,
        color_match=color_match,
        gender_match=gender_match,
        usage_match=usage_match,
        sub_category_match=sub_category_match,
    )

    return RerankFeature(
        product_id=result.product_id,
        recall_score=recall_score,
        freshness_score=freshness_score,
        title_match=title_match,
        article_type_match=article_type_match,
        color_match=color_match,
        gender_match=gender_match,
        usage_match=usage_match,
        sub_category_match=sub_category_match,
        text_match_score=title_match,
        metadata_match_score=metadata_match_score,
    )


def build_rerank_features(
    results: Sequence[RetrievalResult],
    query_text: str | None = None,
) -> list[RerankFeature]:
    """批量构造规则重排特征。"""
    return [build_rerank_feature(result, query_text=query_text) for result in results]


def calculate_metadata_match_score(
    *,
    title_match: float,
    article_type_match: float,
    color_match: float,
    gender_match: float,
    usage_match: float,
    sub_category_match: float,
) -> float:
    """按第一版规则计算 metadata match 分。"""
    return (
        ARTICLE_TYPE_MATCH_WEIGHT * article_type_match
        + COLOR_MATCH_WEIGHT * color_match
        + TITLE_MATCH_WEIGHT * title_match
        + GENDER_MATCH_WEIGHT * gender_match
        + USAGE_MATCH_WEIGHT * usage_match
        + SUB_CATEGORY_MATCH_WEIGHT * sub_category_match
    )


def build_rerank_features_placeholder() -> list[SchemaRerankFeature]:
    """兼容旧调用：真实规则特征请使用 build_rerank_features。"""
    return []


def _tokenize(text: str | None) -> set[str]:
    """lowercase 后按空格和常见符号简单切词。"""
    if text is None:
        return set()
    return {token for token in TOKEN_SPLIT_RE.split(text.lower()) if token}


def _normalize_text(text: str | None) -> str:
    """把文本规整成用于短语包含判断的形式。"""
    if text is None:
        return ""
    return " ".join(TOKEN_SPLIT_RE.split(text.lower())).strip()


def _overlap_ratio(query_tokens: set[str], field_tokens: set[str]) -> float:
    """计算 query token 被字段覆盖的比例，缺失字段按 0 处理。"""
    if not query_tokens or not field_tokens:
        return 0.0
    return len(query_tokens & field_tokens) / len(query_tokens)


def _field_match(query_text: str, query_tokens: set[str], field_value: str | None) -> float:
    """字段和 query 任意 token 或短语命中则记 1。"""
    field_tokens = _tokenize(field_value)
    if not query_tokens or not field_tokens:
        return 0.0
    if query_tokens & field_tokens:
        return 1.0

    normalized_field = _normalize_text(field_value)
    normalized_query = _normalize_text(query_text)
    if normalized_field and normalized_field in normalized_query:
        return 1.0
    return 0.0
