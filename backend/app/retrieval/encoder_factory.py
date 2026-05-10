"""根据 index_meta 创建离线检索 query encoder。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.embedding.image_encoder import DummyImageEncoder, ImageEncoder, TransformersCLIPImageEncoder
from backend.app.embedding.text_encoder import DummyTextEncoder, TextEncoder, TransformersCLIPTextEncoder
from backend.app.embedding.transformers_clip import TransformersCLIPRuntime


def create_text_encoder_from_index_meta(
    index_meta: dict[str, Any],
    device: str = "auto",
    project_root: Path | None = None,
) -> TextEncoder:
    """按 index_meta 创建文本 query encoder。"""
    if _is_dummy_source(index_meta):
        return DummyTextEncoder(embedding_dim=int(index_meta["embedding_dim"]))

    encoder_name = _source_encoder_name(index_meta)
    runtime = _create_transformers_runtime(index_meta, device=device, project_root=project_root)
    return TransformersCLIPTextEncoder(runtime=runtime, encoder_name=encoder_name)


def create_image_encoder_from_index_meta(
    index_meta: dict[str, Any],
    device: str = "auto",
    project_root: Path | None = None,
) -> ImageEncoder:
    """按 index_meta 创建图片 query encoder。"""
    if _is_dummy_source(index_meta):
        return DummyImageEncoder(embedding_dim=int(index_meta["embedding_dim"]))

    encoder_name = _source_encoder_name(index_meta)
    runtime = _create_transformers_runtime(index_meta, device=device, project_root=project_root)
    return TransformersCLIPImageEncoder(runtime=runtime, encoder_name=encoder_name)


def resolve_model_name_for_runtime(source_model_name: object, project_root: Path | None = None) -> str:
    """把本地相对模型路径解析成绝对路径，Hugging Face repo id 原样返回。"""
    if source_model_name is None or not str(source_model_name).strip():
        raise ValueError("真实 encoder 的 source_model_name 不能为空。")

    settings = get_settings()
    project_root = project_root or settings.project_root
    model_name = str(source_model_name).strip()
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path.resolve())

    normalized = model_name.replace("\\", "/")
    if _looks_like_local_model_path(normalized, project_root):
        return str((project_root / normalized).resolve())
    return model_name


def _create_transformers_runtime(
    index_meta: dict[str, Any],
    device: str,
    project_root: Path | None,
) -> TransformersCLIPRuntime:
    """创建共享 Transformers CLIP runtime。"""
    encoder_name = _source_encoder_name(index_meta)
    if encoder_name not in {"fashion-clip", "clip"}:
        raise ValueError(f"不支持的 source_encoder_name: {encoder_name}")

    return _cached_transformers_runtime(
        model_name=resolve_model_name_for_runtime(index_meta.get("source_model_name"), project_root),
        device=device,
        normalize_embeddings=bool(index_meta.get("source_normalize_embeddings", True)),
    )


@lru_cache(maxsize=4)
def _cached_transformers_runtime(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
) -> TransformersCLIPRuntime:
    """缓存真实模型 runtime，避免每次 API 请求重新加载模型。"""
    return TransformersCLIPRuntime(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize_embeddings,
    )


def _is_dummy_source(index_meta: dict[str, Any]) -> bool:
    """判断 index 是否由 dummy encoder 生成。"""
    return bool(index_meta.get("source_is_dummy", False))


def _source_encoder_name(index_meta: dict[str, Any]) -> str:
    """读取并校验 source_encoder_name。"""
    encoder_name = str(index_meta.get("source_encoder_name") or "").strip()
    if not encoder_name:
        raise ValueError("index_meta 缺少 source_encoder_name。")
    return encoder_name


def _looks_like_local_model_path(normalized_model_name: str, project_root: Path) -> bool:
    """区分本地相对模型路径和 Hugging Face repo id。"""
    if normalized_model_name.startswith(("./", "../", "models/")):
        return True
    return (project_root / normalized_model_name).exists()
