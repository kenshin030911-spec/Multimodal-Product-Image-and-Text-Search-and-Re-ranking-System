"""商品数据结构。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProductItem(BaseModel):
    """标准化后的服装商品元数据。"""

    product_id: str = Field(..., description="商品 ID")
    title: str = Field(default="", description="商品标题")
    gender: str | None = Field(default=None, description="性别")
    master_category: str | None = Field(default=None, description="一级大类")
    sub_category: str | None = Field(default=None, description="二级类别")
    article_type: str | None = Field(default=None, description="商品类型")
    base_colour: str | None = Field(default=None, description="基础颜色")
    season: str | None = Field(default=None, description="季节")
    year: int | None = Field(default=None, description="年份")
    usage: str | None = Field(default=None, description="使用场景")
    image_path: str | None = Field(default=None, description="内部图片相对路径")
    freshness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="数据集内部相对新鲜度分数",
    )
