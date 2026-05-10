"""图片检索服务。"""

from __future__ import annotations

from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import load_products
from backend.app.data.validators import to_project_relative
from backend.app.index.index_store import load_index_bundle
from backend.app.index.vector_searcher import VectorSearcher
from backend.app.retrieval.candidate_builder import RetrievalResponse, build_retrieval_response
from backend.app.retrieval.encoder_factory import create_image_encoder_from_index_meta
from backend.app.schemas.search import SearchResponse


def search_image_to_image(
    query_image_path: Path | str,
    top_k: int = 20,
    index_dir: Path | None = None,
    products_path: Path | None = None,
    exclude_product_id: str | None = None,
    exclude_embedding_index: int | None = None,
    device: str = "auto",
    project_root: Path | None = None,
) -> RetrievalResponse:
    """离线 image-to-image 检索，不接 API 和 reranker。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0。")

    settings = get_settings()
    project_root = project_root or settings.project_root
    resolved_query_path = _resolve_query_image_path(query_image_path, project_root)
    if not resolved_query_path.is_file():
        raise FileNotFoundError(f"query image 不存在: {resolved_query_path}")

    index_bundle = load_index_bundle(index_dir)
    image_encoder = create_image_encoder_from_index_meta(
        index_bundle.meta,
        device=device,
        project_root=project_root,
    )
    query_vector = image_encoder.encode_batch([resolved_query_path])[0]
    vector_results = VectorSearcher(index_bundle).search(
        query_vector=query_vector,
        top_k=top_k,
        exclude_product_id=exclude_product_id,
        exclude_embedding_index=exclude_embedding_index,
    )
    products = load_products(products_path)
    return build_retrieval_response(
        query_type="image",
        query=to_project_relative(resolved_query_path, project_root),
        top_k=top_k,
        vector_results=vector_results,
        products=products,
        project_root=project_root,
    )


def placeholder_image_search(
    file_name: str,
    top_k: int,
    use_rerank: bool,
) -> SearchResponse:
    """返回图片检索占位响应，不暴露本地上传路径。"""
    return SearchResponse(
        query_type="image",
        query=file_name,
        top_k=top_k,
        use_rerank=use_rerank,
        reranker_type="rule" if use_rerank else "none",
        reranker_message=(
            "占位响应：已请求规则 reranker。"
            if use_rerank
            else "占位响应：reranker 未启用。"
        ),
        results=[],
        placeholder=True,
        message="第一阶段占位响应：图片编码、向量召回和 reranker 尚未实现。",
    )


def _resolve_query_image_path(query_image_path: Path | str, project_root: Path) -> Path:
    """把命令行传入的 query image 路径解析到项目根目录。"""
    path = Path(query_image_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
