"""健康检查接口。"""

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """服务健康状态。"""

    status: str
    index_loaded: bool
    reranker_loaded: bool
    product_count: int
    placeholder: bool


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """返回服务可访问状态，不代表真实索引或模型已经加载。"""
    return HealthResponse(
        status="ok",
        index_loaded=False,
        reranker_loaded=False,
        product_count=0,
        placeholder=True,
    )
