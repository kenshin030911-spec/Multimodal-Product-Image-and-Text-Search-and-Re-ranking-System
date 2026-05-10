"""规则 reranker baseline 测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.app.reranker.feature_builder import build_rerank_feature
from backend.app.reranker.rerank_model import PlaceholderReranker, RuleBasedReranker
from backend.app.reranker.rerank_service import rerank_retrieval_response
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.schemas.rerank import RerankFeature as SchemaRerankFeature


def test_rerank_schema_and_placeholders() -> None:
    """旧 schema 和 placeholder 仍保持兼容。"""
    feature = SchemaRerankFeature(product_id="15970", recall_score=0.78)

    assert feature.freshness_score == 0.5
    assert PlaceholderReranker().score([feature]) == []


def test_feature_builder_matches_color_and_article_type() -> None:
    """简单文本规则能识别颜色、类别和其他元数据命中。"""
    result = _result(
        "p1",
        title="Black Cotton Shirts",
        article_type="Shirts",
        base_colour="Black",
        gender="Men",
        usage="Casual",
        sub_category="Topwear",
    )

    feature = build_rerank_feature(result, query_text="black casual shirts men topwear")

    assert feature.color_match == 1.0
    assert feature.article_type_match == 1.0
    assert feature.gender_match == 1.0
    assert feature.usage_match == 1.0
    assert feature.sub_category_match == 1.0
    assert feature.title_match > 0.0
    assert feature.metadata_match_score > 0.0


def test_feature_builder_handles_missing_fields_and_image_query() -> None:
    """缺失字段和 image-to-image 场景不会产生匹配特征或报错。"""
    result = _result(
        "p1",
        title="",
        article_type=None,
        base_colour=None,
        gender=None,
        usage=None,
        sub_category=None,
    )

    feature = build_rerank_feature(result, query_text=None)

    assert feature.title_match == 0.0
    assert feature.article_type_match == 0.0
    assert feature.color_match == 0.0
    assert feature.metadata_match_score == 0.0


def test_rule_based_reranker_uses_expected_formula() -> None:
    """规则分数等于 recall + metadata 小权重 + freshness 小权重。"""
    result = _result("p1", score=0.4, freshness_score=0.6)
    feature = build_rerank_feature(result, query_text=None)
    feature = replace(feature, metadata_match_score=1.0)

    score = RuleBasedReranker().score_feature(feature)

    assert score == pytest.approx(0.4 + 0.20 * 1.0 + 0.05 * 0.6)


def test_rule_rerank_can_change_order_with_metadata_match() -> None:
    """metadata 命中足够强时，规则重排可以调整基础召回顺序。"""
    response = _response(
        [
            _result(
                "p1",
                score=0.70,
                rank=1,
                title="Red Shoes",
                article_type="Shoes",
                base_colour="Red",
                gender="Women",
                usage="Sports",
                sub_category="Footwear",
            ),
            _result(
                "p2",
                score=0.64,
                rank=2,
                title="Black Casual Shirts",
                article_type="Shirts",
                base_colour="Black",
                gender="Men",
                usage="Casual",
                sub_category="Topwear",
            ),
        ]
    )

    reranked = rerank_retrieval_response(
        response,
        query_text="black casual shirts men topwear",
    )

    assert [result.product_id for result in reranked.results] == ["p2", "p1"]
    assert reranked.results[0].recall_rank == 2
    assert reranked.results[0].final_rank == 1
    assert reranked.results[0].rerank_score > reranked.results[0].score


def test_freshness_does_not_override_much_higher_recall() -> None:
    """freshness 权重较小，不会压倒明显更高的 recall_score。"""
    response = _response(
        [
            _result("high", score=0.80, rank=1, freshness_score=0.0),
            _result("fresh", score=0.70, rank=2, freshness_score=1.0),
        ]
    )

    reranked = rerank_retrieval_response(response, query_text=None)

    assert [result.product_id for result in reranked.results] == ["high", "fresh"]
    assert reranked.results[0].rerank_score == pytest.approx(0.80)
    assert reranked.results[1].rerank_score == pytest.approx(0.75)


def test_rerank_sort_is_stable_on_ties() -> None:
    """分数相同时按原始 recall_rank 升序稳定排序。"""
    response = _response(
        [
            _result("p1", score=0.5, rank=1, freshness_score=0.5),
            _result("p2", score=0.5, rank=2, freshness_score=0.5),
        ]
    )

    reranked = rerank_retrieval_response(response, query_text=None)

    assert [result.product_id for result in reranked.results] == ["p1", "p2"]
    assert [result.final_rank for result in reranked.results] == [1, 2]


def _response(results: list[RetrievalResult]) -> RetrievalResponse:
    return RetrievalResponse(
        query_type="text",
        query="black shirt",
        top_k=len(results),
        results=results,
        missing_product_ids=[],
    )


def _result(
    product_id: str,
    *,
    score: float = 0.5,
    rank: int = 1,
    title: str = "Product",
    article_type: str | None = "Shirts",
    base_colour: str | None = "Black",
    gender: str | None = "Men",
    usage: str | None = "Casual",
    sub_category: str | None = "Topwear",
    freshness_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        product_id=product_id,
        title=title,
        image_path=f"data/raw/images/{product_id}.jpg",
        article_type=article_type,
        base_colour=base_colour,
        gender=gender,
        usage=usage,
        sub_category=sub_category,
        freshness_score=freshness_score,
        score=score,
        rank=rank,
        embedding_index=rank - 1,
        recall_rank=rank,
        rerank_score=score,
        final_rank=rank,
    )
