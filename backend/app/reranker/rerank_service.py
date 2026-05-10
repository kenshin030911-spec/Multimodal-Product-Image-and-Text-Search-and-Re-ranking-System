"""规则重排服务。"""

from __future__ import annotations

from dataclasses import replace

from backend.app.reranker.feature_builder import build_rerank_features
from backend.app.reranker.rerank_model import RuleBasedReranker
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.schemas.search import SearchResult


def rerank_retrieval_response(
    response: RetrievalResponse,
    query_text: str | None = None,
) -> RetrievalResponse:
    """对基础召回结果做规则重排，保留原始召回 rank/score。"""
    if not response.results:
        return response

    features = build_rerank_features(response.results, query_text=query_text)
    score_by_product_id = RuleBasedReranker().score(features)
    ranked_results = sorted(
        response.results,
        key=lambda result: (
            -score_by_product_id.get(result.product_id, result.score),
            result.recall_rank,
        ),
    )

    reranked_results: list[RetrievalResult] = []
    for final_rank, result in enumerate(ranked_results, start=1):
        reranked_results.append(
            replace(
                result,
                # rank 是兼容字段，始终跟随当前最终排序位置。
                rank=final_rank,
                rerank_score=score_by_product_id.get(result.product_id, result.score),
                final_rank=final_rank,
            )
        )

    return replace(response, results=reranked_results)


def rerank_results_placeholder(results: list[SearchResult]) -> list[SearchResult]:
    """兼容旧调用：不改变 SearchResult 排序。"""
    return results
