"""召回候选和商品元数据合并。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative
from backend.app.index.vector_searcher import VectorSearchResult
from backend.app.schemas.product import ProductItem
from backend.app.schemas.search import RecallCandidate


@dataclass(frozen=True)
class RetrievalResult:
    """离线检索结果，不限制 cosine score 范围。

    rank 是兼容字段，表示当前排序位置：初始召回阶段等于 recall_rank，
    规则重排后等于 final_rank。新代码应优先使用 recall_rank/final_rank。
    """

    product_id: str
    title: str
    image_path: str | None
    article_type: str | None
    base_colour: str | None
    gender: str | None
    usage: str | None
    sub_category: str | None
    freshness_score: float
    score: float
    # 兼容旧调用的当前排序位置；不要用它表示固定的原始召回排名。
    rank: int
    embedding_index: int
    recall_rank: int
    rerank_score: float
    final_rank: int


@dataclass(frozen=True)
class RetrievalResponse:
    """离线检索响应，供 service 和命令行 demo 使用。"""

    query_type: str
    query: str
    top_k: int
    results: list[RetrievalResult]
    missing_product_ids: list[str]


def build_product_lookup(products: Sequence[ProductItem]) -> dict[str, ProductItem]:
    """按 product_id 构造商品回查表。"""
    return {product.product_id: product for product in products}


def build_retrieval_response(
    query_type: str,
    query: str,
    top_k: int,
    vector_results: Sequence[VectorSearchResult],
    products: Sequence[ProductItem],
    project_root: Path | None = None,
) -> RetrievalResponse:
    """合并 VectorSearchResult 和 ProductItem，缺失商品会记录并跳过。"""
    settings = get_settings()
    project_root = project_root or settings.project_root
    product_lookup = build_product_lookup(products)
    results: list[RetrievalResult] = []
    missing_product_ids: list[str] = []

    for vector_result in vector_results:
        product = product_lookup.get(vector_result.product_id)
        if product is None:
            missing_product_ids.append(vector_result.product_id)
            continue

        results.append(
            RetrievalResult(
                product_id=product.product_id,
                title=product.title,
                image_path=_safe_relative_image_path(product.image_path, project_root),
                article_type=product.article_type,
                base_colour=product.base_colour,
                gender=product.gender,
                usage=product.usage,
                sub_category=product.sub_category,
                freshness_score=product.freshness_score,
                score=vector_result.score,
                rank=vector_result.rank,
                embedding_index=vector_result.embedding_index,
                recall_rank=vector_result.rank,
                rerank_score=vector_result.score,
                final_rank=vector_result.rank,
            )
        )

    return RetrievalResponse(
        query_type=query_type,
        query=query,
        top_k=top_k,
        results=results,
        missing_product_ids=missing_product_ids,
    )


def build_candidates_from_recall() -> list[RecallCandidate]:
    """兼容旧调用：真实候选构造请使用 build_retrieval_response。"""
    return []


def _safe_relative_image_path(image_path: str | None, project_root: Path) -> str | None:
    """返回相对图片路径，避免暴露本机绝对路径。"""
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute():
        return to_project_relative(path, project_root)
    return path.as_posix()
