"""核心 schema smoke test。"""

from backend.app.schemas.product import ProductItem
from backend.app.schemas.search import RecallCandidate, SearchRequest, SearchResult


def test_core_schemas_can_be_instantiated() -> None:
    """核心 Pydantic schema 可以正常实例化。"""
    product = ProductItem(
        product_id="15970",
        title="Turtle Check Men Navy Blue Shirt",
        article_type="Shirts",
        base_colour="Navy Blue",
    )
    request = SearchRequest(query_type="text", query="men navy blue shirt")
    candidate = RecallCandidate(product_id="15970", recall_rank=5, recall_score=0.78)
    result = SearchResult(
        product_id="15970",
        title=product.title,
        image_url="/static/images/15970.jpg",
        article_type=product.article_type,
        base_colour=product.base_colour,
        recall_rank=candidate.recall_rank,
        recall_score=candidate.recall_score,
        rerank_score=0.86,
        freshness_score=product.freshness_score,
        final_rank=1,
    )

    assert request.top_k == 20
    assert product.freshness_score == 0.5
    assert result.final_rank == 1
