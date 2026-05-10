"""搜索相关数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QueryType = Literal["text", "image"]
RerankerType = Literal["none", "rule", "trained", "pairwise"]


class SearchRequest(BaseModel):
    """通用搜索请求，供服务层复用。"""

    query_type: QueryType = Field(default="text", description="搜索类型")
    query: str | None = Field(default=None, description="文本查询或图片文件名")
    top_k: int = Field(default=20, ge=1, le=100, description="返回结果数量")
    use_rerank: bool = Field(default=True, description="是否启用重排")
    reranker_type: RerankerType | None = Field(
        default=None,
        description="none/rule/trained/pairwise；未传时由 use_rerank 兼容推导。",
    )


class TextSearchRequest(BaseModel):
    """文本搜图接口请求。"""

    query: str = Field(..., min_length=1, description="文本查询")
    top_k: int = Field(default=20, ge=1, le=100, description="返回结果数量")
    use_rerank: bool = Field(default=True, description="是否启用重排")
    reranker_type: RerankerType | None = Field(
        default=None,
        description="none/rule/trained/pairwise；未传时由 use_rerank 兼容推导。",
    )


class RecallCandidate(BaseModel):
    """向量召回阶段的候选商品。"""

    product_id: str
    recall_rank: int = Field(..., ge=1)
    recall_score: float = Field(..., ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """最终返回给前端的搜索结果，不暴露本地绝对路径。"""

    product_id: str
    title: str
    image_url: str | None = Field(default=None, description="前端可访问的图片 URL")
    image_path: str | None = Field(default=None, description="项目内相对图片路径")
    article_type: str | None = None
    base_colour: str | None = None
    recall_rank: int = Field(..., ge=1)
    recall_score: float
    rerank_score: float
    freshness_score: float = Field(..., ge=0.0, le=1.0)
    final_rank: int = Field(..., ge=1)


class SearchResponse(BaseModel):
    """搜索接口响应。"""

    query_type: QueryType
    query: str | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    use_rerank: bool = True
    reranker_type: RerankerType = "rule"
    reranker_message: str | None = None
    results: list[SearchResult] = Field(default_factory=list)
    placeholder: bool = Field(default=True, description="第一阶段占位标记")
    message: str = Field(default="第一阶段占位响应，尚未执行真实检索。")
