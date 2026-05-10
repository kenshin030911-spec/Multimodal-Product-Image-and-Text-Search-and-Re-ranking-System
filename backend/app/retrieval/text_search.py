"""文本检索服务。"""

from __future__ import annotations

from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import load_products
from backend.app.index.index_store import load_index_bundle
from backend.app.index.vector_searcher import VectorSearcher
from backend.app.retrieval.candidate_builder import RetrievalResponse, build_retrieval_response
from backend.app.retrieval.encoder_factory import create_text_encoder_from_index_meta
from backend.app.schemas.search import SearchResponse, TextSearchRequest


def search_text_to_image(
    query_text: str,
    top_k: int = 20,
    index_dir: Path | None = None,
    products_path: Path | None = None,
    device: str = "auto",
    project_root: Path | None = None,
) -> RetrievalResponse:
    """离线 text-to-image 检索，不接 API 和 reranker。"""
    if not query_text or not query_text.strip():
        raise ValueError("query_text 不能为空。")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    settings = get_settings()
    project_root = project_root or settings.project_root
    index_bundle = load_index_bundle(index_dir)
    text_encoder = create_text_encoder_from_index_meta(
        index_bundle.meta,
        device=device,
        project_root=project_root,
    )
    query_vector = text_encoder.encode_batch([query_text.strip()])[0]
    vector_results = VectorSearcher(index_bundle).search(query_vector=query_vector, top_k=top_k)
    products = load_products(products_path)
    return build_retrieval_response(
        query_type="text",
        query=query_text.strip(),
        top_k=top_k,
        vector_results=vector_results,
        products=products,
        project_root=project_root,
    )


def placeholder_text_search(request: TextSearchRequest) -> SearchResponse:
    """返回文本检索占位响应，后续会接入文本编码和向量索引。"""
    return SearchResponse(
        query_type="text",
        query=request.query,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
        results=[],
        placeholder=True,
        message="第一阶段占位响应：文本编码、向量召回和 reranker 尚未实现。",
    )
