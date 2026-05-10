"""重排相关数据结构。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RerankFeature(BaseModel):
    """reranker 输入特征，后续规则重排和轻量模型会复用。"""

    product_id: str
    recall_score: float = Field(..., ge=0.0, le=1.0)
    article_type_match: int = Field(default=0, ge=0, le=1)
    sub_category_match: int = Field(default=0, ge=0, le=1)
    color_match: int = Field(default=0, ge=0, le=1)
    gender_match: int = Field(default=0, ge=0, le=1)
    season_match: int = Field(default=0, ge=0, le=1)
    text_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.5, ge=0.0, le=1.0)


class RerankScore(BaseModel):
    """重排后的分数结果。"""

    product_id: str
    rerank_score: float = Field(..., ge=0.0, le=1.0)
