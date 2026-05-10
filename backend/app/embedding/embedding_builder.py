"""embedding 小批量生成编排。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import default_products_path, load_products
from backend.app.data.validators import to_project_relative
from backend.app.embedding.embedding_store import EmbeddingBundle, save_embedding_bundle
from backend.app.embedding.image_encoder import (
    DummyImageEncoder,
    ImageEncoder,
    TransformersCLIPImageEncoder,
)
from backend.app.embedding.text_encoder import (
    DummyTextEncoder,
    TextEncoder,
    TransformersCLIPTextEncoder,
    build_product_text,
)
from backend.app.embedding.transformers_clip import ImageEncodingError, TransformersCLIPRuntime
from backend.app.embedding.versioning import (
    DEFAULT_DUMMY_ENCODER_NAME,
    DEFAULT_REAL_MODEL_NAME,
    SUPPORTED_REAL_ENCODER_NAMES,
    build_embedding_version_info,
)
from backend.app.schemas.product import ProductItem


@dataclass(frozen=True)
class EmbeddingBuildResult:
    """embedding 生成结果。"""

    bundle: EmbeddingBundle
    product_count: int
    failed_images: list[dict[str, str]]
    failed_texts: list[dict[str, str]]
    skipped_products: list[dict[str, str]]


@dataclass(frozen=True)
class _EncoderBundle:
    """image/text encoder 和需要写入 meta 的运行时信息。"""

    image_encoder: ImageEncoder
    text_encoder: TextEncoder
    framework: str | None
    model_name: str | None
    device: str | None
    torch_dtype: str | None
    normalize_embeddings: bool
    model_revision: str | None


@dataclass(frozen=True)
class _EmbeddingCandidate:
    """一条待编码或已部分编码的商品行。"""

    product: ProductItem
    image_path: Path | None
    text: str | None
    image_embedding: np.ndarray | None = None
    text_embedding: np.ndarray | None = None


def build_embeddings(
    products_path: Path | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    batch_size: int = 16,
    encoder_name: str = DEFAULT_DUMMY_ENCODER_NAME,
    model_name: str | None = None,
    device: str = "auto",
    normalize: bool = True,
    image_only: bool = False,
    text_only: bool = False,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> EmbeddingBuildResult:
    """生成小批量 image/text embeddings；默认仍使用 dummy encoder。"""
    settings = get_settings()
    project_root = project_root or settings.project_root
    products_path = products_path or default_products_path(settings)
    output_dir = output_dir or settings.embeddings_dir

    if image_only and text_only:
        raise ValueError("--image-only 和 --text-only 不能同时使用。")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0。")

    generate_image = not text_only
    generate_text = not image_only
    encoder_bundle = _create_encoder_bundle(
        encoder_name=encoder_name,
        model_name=model_name,
        device=device,
        normalize=normalize,
    )
    image_encoder = encoder_bundle.image_encoder
    text_encoder = encoder_bundle.text_encoder
    if image_encoder.embedding_dim != text_encoder.embedding_dim:
        raise ValueError("image/text encoder 维度不一致。")

    products = load_products(products_path, limit=limit)
    prepared = _prepare_embedding_inputs(
        products=products,
        project_root=project_root,
        generate_image=generate_image,
        generate_text=generate_text,
    )
    candidates = prepared["candidates"]
    failed_images = prepared["failed_images"]
    failed_texts = prepared["failed_texts"]
    skipped_products = prepared["skipped_products"]

    if generate_image:
        candidates = _encode_candidate_images(
            encoder=image_encoder,
            candidates=candidates,
            batch_size=batch_size,
            failed_images=failed_images,
            skipped_products=skipped_products,
        )
    if generate_text:
        candidates = _encode_candidate_texts(
            encoder=text_encoder,
            candidates=candidates,
            batch_size=batch_size,
            failed_texts=failed_texts,
            skipped_products=skipped_products,
        )

    image_embeddings = (
        _stack_candidate_embeddings(candidates, "image", image_encoder.embedding_dim)
        if generate_image
        else None
    )
    text_embeddings = (
        _stack_candidate_embeddings(candidates, "text", text_encoder.embedding_dim)
        if generate_text
        else None
    )

    product_ids = [candidate.product.product_id for candidate in candidates]
    version_info = build_embedding_version_info(
        encoder_name=encoder_name,
        embedding_dim=image_encoder.embedding_dim,
        framework=encoder_bundle.framework,
        model_name=encoder_bundle.model_name,
        device=encoder_bundle.device,
        torch_dtype=encoder_bundle.torch_dtype,
        normalize_embeddings=encoder_bundle.normalize_embeddings,
        model_revision=encoder_bundle.model_revision,
    ).model_dump()
    meta: dict[str, Any] = {
        **version_info,
        "products_path": to_project_relative(products_path, project_root),
        "product_count": len(product_ids),
        "product_ids": product_ids,
        "product_id_to_index": {
            product_id: index for index, product_id in enumerate(product_ids)
        },
        "generated": {
            "image": generate_image,
            "text": generate_text,
        },
        "failed_images": failed_images,
        "failed_texts": failed_texts,
        "skipped_products": skipped_products,
        "batch_size": batch_size,
        "limit": limit,
    }

    bundle = save_embedding_bundle(
        output_dir=output_dir,
        meta=meta,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        overwrite=overwrite,
        project_root=project_root,
    )
    return EmbeddingBuildResult(
        bundle=bundle,
        product_count=len(product_ids),
        failed_images=failed_images,
        failed_texts=failed_texts,
        skipped_products=skipped_products,
    )


def _create_encoder_bundle(
    encoder_name: str,
    model_name: str | None,
    device: str,
    normalize: bool,
) -> _EncoderBundle:
    """根据 CLI 参数创建 dummy 或真实 Transformers encoder。"""
    if encoder_name == DEFAULT_DUMMY_ENCODER_NAME:
        return _EncoderBundle(
            image_encoder=DummyImageEncoder(),
            text_encoder=DummyTextEncoder(),
            framework="dummy",
            model_name=None,
            device=None,
            torch_dtype=None,
            normalize_embeddings=False,
            model_revision=None,
        )

    if encoder_name not in SUPPORTED_REAL_ENCODER_NAMES:
        raise ValueError(f"不支持的 encoder_name: {encoder_name}。")

    resolved_model_name = model_name or DEFAULT_REAL_MODEL_NAME
    runtime = TransformersCLIPRuntime(
        model_name=resolved_model_name,
        device=device,
        normalize_embeddings=normalize,
    )
    return _EncoderBundle(
        image_encoder=TransformersCLIPImageEncoder(runtime, encoder_name=encoder_name),
        text_encoder=TransformersCLIPTextEncoder(runtime, encoder_name=encoder_name),
        framework="transformers",
        model_name=resolved_model_name,
        device=runtime.device,
        torch_dtype=runtime.torch_dtype,
        normalize_embeddings=runtime.normalize_embeddings,
        model_revision=runtime.model_revision,
    )


def _prepare_embedding_inputs(
    products: list[ProductItem],
    project_root: Path,
    generate_image: bool,
    generate_text: bool,
) -> dict[str, Any]:
    """筛选本轮可成功生成所需模态的商品，保证行顺序对齐。"""
    candidates: list[_EmbeddingCandidate] = []
    failed_images: list[dict[str, str]] = []
    failed_texts: list[dict[str, str]] = []
    skipped_products: list[dict[str, str]] = []

    for product in products:
        product_image_path: Path | None = None
        product_text = build_product_text(product)
        reasons: list[str] = []

        if generate_image:
            product_image_path = _resolve_image_path(product, project_root)
            if product_image_path is None or not product_image_path.is_file():
                failed_images.append(
                    {
                        "product_id": product.product_id,
                        "reason": "image_path_not_found",
                    }
                )
                reasons.append("image_failed")

        if generate_text and not product_text:
            failed_texts.append(
                {
                    "product_id": product.product_id,
                    "reason": "empty_product_text",
                }
            )
            reasons.append("text_failed")

        if reasons:
            skipped_products.append(
                {
                    "product_id": product.product_id,
                    "reason": ",".join(reasons),
                }
            )
            continue

        candidates.append(
            _EmbeddingCandidate(
                product=product,
                image_path=product_image_path if generate_image else None,
                text=product_text if generate_text else None,
            )
        )

    return {
        "candidates": candidates,
        "failed_images": failed_images,
        "failed_texts": failed_texts,
        "skipped_products": skipped_products,
    }


def _resolve_image_path(product: ProductItem, project_root: Path) -> Path | None:
    """把 ProductItem.image_path 解析为本地 Path，仅供离线生成使用。"""
    if not product.image_path:
        return None
    image_path = Path(product.image_path)
    if not image_path.is_absolute():
        image_path = project_root / image_path
    return image_path


def _encode_candidate_images(
    encoder: ImageEncoder,
    candidates: list[_EmbeddingCandidate],
    batch_size: int,
    failed_images: list[dict[str, str]],
    skipped_products: list[dict[str, str]],
) -> list[_EmbeddingCandidate]:
    """按 batch 编码图片；单张坏图会被记录并跳过。"""
    encoded: list[_EmbeddingCandidate] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        encoded.extend(
            _encode_image_batch_with_fallback(
                encoder=encoder,
                batch=batch,
                failed_images=failed_images,
                skipped_products=skipped_products,
            )
        )
    return encoded


def _encode_candidate_texts(
    encoder: TextEncoder,
    candidates: list[_EmbeddingCandidate],
    batch_size: int,
    failed_texts: list[dict[str, str]],
    skipped_products: list[dict[str, str]],
) -> list[_EmbeddingCandidate]:
    """按 batch 编码文本；单条异常文本会被记录并跳过。"""
    encoded: list[_EmbeddingCandidate] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        encoded.extend(
            _encode_text_batch_with_fallback(
                encoder=encoder,
                batch=batch,
                failed_texts=failed_texts,
                skipped_products=skipped_products,
            )
        )
    return encoded


def _encode_image_batch_with_fallback(
    encoder: ImageEncoder,
    batch: list[_EmbeddingCandidate],
    failed_images: list[dict[str, str]],
    skipped_products: list[dict[str, str]],
) -> list[_EmbeddingCandidate]:
    """批量失败时回退到逐张编码，避免坏图中断全流程。"""
    paths = [_require_image_path(candidate) for candidate in batch]
    try:
        vectors = encoder.encode_batch(paths)
        _validate_batch_output(vectors, len(batch), encoder.embedding_dim, "image_embeddings")
        return [
            replace(candidate, image_embedding=vectors[index].astype(np.float32, copy=False))
            for index, candidate in enumerate(batch)
        ]
    except ImageEncodingError as exc:
        if len(batch) == 1:
            candidate = batch[0]
            failed_images.append(
                {
                    "product_id": candidate.product.product_id,
                    "reason": _failure_reason(exc),
                }
            )
            skipped_products.append(
                {
                    "product_id": candidate.product.product_id,
                    "reason": "image_failed",
                }
            )
            return []

    encoded: list[_EmbeddingCandidate] = []
    for candidate in batch:
        encoded.extend(
            _encode_image_batch_with_fallback(
                encoder=encoder,
                batch=[candidate],
                failed_images=failed_images,
                skipped_products=skipped_products,
            )
        )
    return encoded


def _encode_text_batch_with_fallback(
    encoder: TextEncoder,
    batch: list[_EmbeddingCandidate],
    failed_texts: list[dict[str, str]],
    skipped_products: list[dict[str, str]],
) -> list[_EmbeddingCandidate]:
    """按 batch 编码文本；文本缺失已在准备阶段过滤。"""
    texts = [_require_text(candidate) for candidate in batch]
    vectors = encoder.encode_batch(texts)
    _validate_batch_output(vectors, len(batch), encoder.embedding_dim, "text_embeddings")
    return [
        replace(candidate, text_embedding=vectors[index].astype(np.float32, copy=False))
        for index, candidate in enumerate(batch)
    ]


def _stack_candidate_embeddings(
    candidates: list[_EmbeddingCandidate],
    modality: str,
    embedding_dim: int,
) -> np.ndarray:
    """按最终商品顺序合并单行 embedding。"""
    vectors: list[np.ndarray] = []
    for candidate in candidates:
        vector = candidate.image_embedding if modality == "image" else candidate.text_embedding
        if vector is None:
            raise ValueError(f"{modality}_embedding 缺失，无法保存对齐结果。")
        vectors.append(vector.astype(np.float32, copy=False))
    if not vectors:
        return np.empty((0, embedding_dim), dtype=np.float32)
    return np.vstack(vectors).astype(np.float32, copy=False)


def _require_image_path(candidate: _EmbeddingCandidate) -> Path:
    """取出候选图片路径，缺失视为编码错误。"""
    if candidate.image_path is None:
        raise ValueError("image_path_missing")
    return candidate.image_path


def _require_text(candidate: _EmbeddingCandidate) -> str:
    """取出候选文本，缺失视为编码错误。"""
    if candidate.text is None:
        raise ValueError("empty_product_text")
    return candidate.text


def _validate_batch_output(
    vectors: np.ndarray,
    expected_rows: int,
    embedding_dim: int,
    name: str,
) -> None:
    """校验 encoder batch 输出，避免写入错位 embedding。"""
    if vectors.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组。")
    if vectors.shape != (expected_rows, embedding_dim):
        raise ValueError(f"{name} shape 应为 {(expected_rows, embedding_dim)}，实际为 {vectors.shape}。")
    if vectors.dtype != np.float32:
        raise ValueError(f"{name} dtype 必须是 float32。")


def _failure_reason(exc: Exception) -> str:
    """生成不包含本机绝对路径的失败原因。"""
    if isinstance(exc, ImageEncodingError):
        return exc.reason
    if isinstance(exc, ValueError) and str(exc):
        return str(exc)
    return exc.__class__.__name__


def _stack_batches(batches: Sequence[np.ndarray] | Any, embedding_dim: int) -> np.ndarray:
    """合并 batch 结果，空输入返回固定二维 float32 数组。"""
    arrays = [array for array in batches if array.size > 0]
    if not arrays:
        return np.empty((0, embedding_dim), dtype=np.float32)
    return np.vstack(arrays).astype(np.float32, copy=False)
