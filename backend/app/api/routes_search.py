"""搜索接口。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, unquote

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import default_products_path
from backend.app.index.index_store import IMAGE_INDEX_FILE, INDEX_META_FILE
from backend.app.reranker.pairwise_rerank_service import (
    rerank_retrieval_response_with_pairwise_model,
)
from backend.app.reranker.rerank_service import rerank_retrieval_response
from backend.app.reranker.trained_rerank_service import (
    rerank_retrieval_response_with_trained_model,
)
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.retrieval.image_search import search_image_to_image
from backend.app.retrieval.text_search import search_text_to_image
from backend.app.schemas.search import (
    RerankerType,
    SearchResponse,
    SearchResult,
    TextSearchRequest,
)

router = APIRouter(prefix="/search")
SEARCH_COMPLETE_MESSAGE = "基础向量检索完成，reranker 未启用。"
RERANK_COMPLETE_MESSAGE = "基础向量检索完成，已启用规则 reranker baseline。"
TRAINED_RERANK_COMPLETE_MESSAGE = "基础向量检索完成，已启用 experimental trained reranker。"
PAIRWISE_RERANK_COMPLETE_MESSAGE = (
    "基础向量检索完成，已启用 experimental pairwise reranker。"
    "pairwise score is an ordering score, not calibrated probability."
)


@router.post("/text", response_model=SearchResponse)
def search_text(request: TextSearchRequest) -> SearchResponse:
    """文本搜图接口，执行基础向量召回，可选规则 reranker。"""
    settings = get_settings()
    _ensure_search_assets(settings)
    reranker_type = resolve_reranker_type(
        use_rerank=request.use_rerank,
        reranker_type=request.reranker_type,
    )
    try:
        retrieval_response = search_text_to_image(
            query_text=request.query,
            top_k=request.top_k,
            index_dir=settings.index_dir,
            products_path=default_products_path(settings),
            device="auto",
            project_root=settings.project_root,
        )
    except Exception as exc:
        raise _to_search_http_exception(exc) from exc

    message = SEARCH_COMPLETE_MESSAGE
    if reranker_type == "rule":
        retrieval_response = rerank_retrieval_response(
            retrieval_response,
            query_text=request.query,
        )
        message = RERANK_COMPLETE_MESSAGE
    elif reranker_type == "trained":
        try:
            retrieval_response = rerank_retrieval_response_with_trained_model(
                retrieval_response,
                query_text=request.query,
            )
        except Exception as exc:
            raise _to_search_http_exception(exc) from exc
        message = TRAINED_RERANK_COMPLETE_MESSAGE
    elif reranker_type == "pairwise":
        try:
            retrieval_response = rerank_retrieval_response_with_pairwise_model(
                retrieval_response,
                query_text=request.query,
            )
        except Exception as exc:
            raise _to_search_http_exception(exc) from exc
        message = PAIRWISE_RERANK_COMPLETE_MESSAGE

    return _to_search_response(
        retrieval_response=retrieval_response,
        reranker_type=reranker_type,
        message=message,
    )


@router.post("/image", response_model=SearchResponse)
async def search_image(
    file: Annotated[UploadFile, File(description="查询图片")],
    top_k: Annotated[int, Form(ge=1, le=100)] = 20,
    use_rerank: Annotated[bool, Form()] = True,
    reranker_type: Annotated[RerankerType | None, Form()] = None,
    exclude_product_id: Annotated[str | None, Form()] = None,
) -> SearchResponse:
    """图片搜图接口，保存受控临时文件后执行基础向量召回。"""
    settings = get_settings()
    resolved_reranker_type = resolve_reranker_type(
        use_rerank=use_rerank,
        reranker_type=reranker_type,
    )
    if resolved_reranker_type == "trained":
        raise HTTPException(
            status_code=400,
            detail="trained reranker currently supports text search only.",
        )
    if resolved_reranker_type == "pairwise":
        raise HTTPException(
            status_code=400,
            detail="pairwise reranker currently supports text search only.",
        )

    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片类型: {file.content_type}",
        )

    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="上传图片超过大小限制。")

    _ensure_search_assets(settings)
    temp_path = _write_upload_to_temp_file(file, content, settings.uploads_tmp_dir)
    try:
        retrieval_response = search_image_to_image(
            query_image_path=temp_path,
            top_k=top_k,
            index_dir=settings.index_dir,
            products_path=default_products_path(settings),
            exclude_product_id=exclude_product_id,
            device="auto",
            project_root=settings.project_root,
        )
    except Exception as exc:
        raise _to_search_http_exception(exc) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    message = SEARCH_COMPLETE_MESSAGE
    if resolved_reranker_type == "rule":
        retrieval_response = rerank_retrieval_response(
            retrieval_response,
            query_text=None,
        )
        message = RERANK_COMPLETE_MESSAGE

    return _to_search_response(
        retrieval_response=retrieval_response,
        reranker_type=resolved_reranker_type,
        query_override=file.filename or "uploaded-image",
        message=message,
    )


def resolve_reranker_type(
    use_rerank: bool,
    reranker_type: RerankerType | None,
) -> RerankerType:
    """Resolve new reranker_type while preserving old use_rerank behavior."""
    if reranker_type is not None:
        return reranker_type
    return "rule" if use_rerank else "none"


def _to_search_response(
    retrieval_response: RetrievalResponse,
    reranker_type: RerankerType,
    query_override: str | None = None,
    message: str = SEARCH_COMPLETE_MESSAGE,
) -> SearchResponse:
    """把离线检索响应映射成 API schema。"""
    settings = get_settings()
    return SearchResponse(
        query_type=retrieval_response.query_type,  # type: ignore[arg-type]
        query=query_override or retrieval_response.query,
        top_k=retrieval_response.top_k,
        use_rerank=reranker_type != "none",
        reranker_type=reranker_type,
        reranker_message=message,
        results=[
            _to_search_result(result, settings=settings)
            for result in retrieval_response.results
        ],
        placeholder=False,
        message=message,
    )


def _to_search_result(result: RetrievalResult, settings) -> SearchResult:
    """把 RetrievalResult 映射为 API SearchResult。"""
    return SearchResult(
        product_id=result.product_id,
        title=result.title,
        image_url=_to_image_url(result.image_path, settings),
        image_path=result.image_path,
        article_type=result.article_type,
        base_colour=result.base_colour,
        recall_rank=result.recall_rank,
        recall_score=result.score,
        rerank_score=result.rerank_score,
        freshness_score=result.freshness_score,
        final_rank=result.final_rank,
    )


def _to_image_url(image_path: str | None, settings) -> str | None:
    """把项目内图片路径转换成前端可访问 URL，路径不合法时返回 None。"""
    if not image_path:
        return None

    try:
        decoded_image_path = unquote(image_path)
        if _contains_path_traversal(decoded_image_path):
            return None

        raw_images_dir = settings.raw_images_dir.resolve()
        path = Path(decoded_image_path)
        if path.is_absolute():
            resolved_image_path = path.resolve()
        else:
            resolved_image_path = _resolve_relative_image_path(path, raw_images_dir, settings)

        if not resolved_image_path.is_relative_to(raw_images_dir):
            return None

        relative_path = resolved_image_path.relative_to(raw_images_dir)
        if _contains_path_traversal(relative_path.as_posix()):
            return None

        url_path = "/".join(quote(part) for part in relative_path.parts)
        if not url_path:
            return None
        return f"{settings.static_images_url_prefix.rstrip('/')}/{url_path}"
    except (OSError, RuntimeError, ValueError):
        return None


def _contains_path_traversal(path_value: str) -> bool:
    """检测原始路径中是否包含 .. 段。"""
    normalized = path_value.replace("\\", "/")
    return any(part == ".." for part in normalized.split("/"))


def _resolve_relative_image_path(path: Path, raw_images_dir: Path, settings) -> Path:
    """把相对 image_path 解析到当前 raw_images_dir 内。"""
    candidates = [
        (settings.project_root / path).resolve(),
        (settings.data_dir.parent / path).resolve(),
    ]
    if not _starts_with_data_segment(path):
        candidates.append((raw_images_dir / path).resolve())
    for candidate in candidates:
        if candidate.is_relative_to(raw_images_dir):
            return candidate
    return candidates[0]


def _starts_with_data_segment(path: Path) -> bool:
    """判断相对路径是否显式以 data 开头。"""
    return bool(path.parts) and path.parts[0].lower() == "data"


def _ensure_search_assets(settings) -> None:
    """提前检查搜索依赖文件，返回受控错误。"""
    missing: list[str] = []
    if not (settings.index_dir / INDEX_META_FILE).is_file():
        missing.append("index_meta.json")
    if not (settings.index_dir / IMAGE_INDEX_FILE).is_file():
        missing.append("image_index.npy")
    if not default_products_path(settings).is_file():
        missing.append("products.jsonl")
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"搜索依赖文件不存在: {', '.join(missing)}",
        )


def _write_upload_to_temp_file(file: UploadFile, content: bytes, upload_dir: Path) -> Path:
    """把上传图片写入 uploads/tmp 下的安全临时文件。"""
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = _safe_upload_suffix(file.filename)
    temp_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(content)
    return temp_path


def _safe_upload_suffix(file_name: str | None) -> str:
    """仅保留图片扩展名，避免使用用户传入文件名。"""
    suffix = Path(file_name or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".img"


def _to_search_http_exception(exc: Exception) -> HTTPException:
    """把内部异常转换成不暴露堆栈的 HTTPException。"""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ImportError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="搜索服务内部错误。")
