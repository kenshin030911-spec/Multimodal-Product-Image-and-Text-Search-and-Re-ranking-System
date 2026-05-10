"""文本编码器接口和 dummy 实现。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from backend.app.embedding.versioning import (
    DEFAULT_DUMMY_EMBEDDING_DIM,
    DEFAULT_DUMMY_ENCODER_NAME,
    DEFAULT_DUMMY_ENCODER_VERSION,
    REAL_ENCODER_VERSION,
)
from backend.app.embedding.transformers_clip import TransformersCLIPRuntime
from backend.app.schemas.product import ProductItem


class TextEncoder(Protocol):
    """文本编码器接口，后续真实 CLIP 会实现相同方法。"""

    encoder_name: str
    encoder_version: str
    embedding_dim: int

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码文本。"""


@dataclass(frozen=True)
class DummyTextEncoder:
    """稳定 dummy 文本编码器，不加载真实模型。"""

    embedding_dim: int = DEFAULT_DUMMY_EMBEDDING_DIM
    encoder_name: str = DEFAULT_DUMMY_ENCODER_NAME
    encoder_version: str = DEFAULT_DUMMY_ENCODER_VERSION

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """根据文本内容生成稳定 float32 向量。"""
        vectors = [_stable_vector(text, self.embedding_dim) for text in texts]
        if not vectors:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TransformersCLIPTextEncoder:
    """基于 Transformers CLIPModel 的真实文本编码器。"""

    runtime: TransformersCLIPRuntime
    encoder_name: str
    encoder_version: str = REAL_ENCODER_VERSION

    @property
    def embedding_dim(self) -> int:
        """返回真实模型 projection 维度。"""
        return self.runtime.embedding_dim

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """调用 CLIP text encoder，截断交给 CLIPProcessor tokenizer 处理。"""
        return self.runtime.encode_texts(texts)


def build_product_text(product: ProductItem) -> str:
    """把商品结构化字段拼成后续文本 encoder 的输入。"""
    parts = [
        product.title,
        _format_part("article type", product.article_type),
        _format_part("color", product.base_colour),
        _format_part("gender", product.gender),
        _format_part("usage", product.usage),
        _format_part("category", product.sub_category),
    ]
    return ". ".join(part for part in parts if part)


def encode_text_placeholder(text: str) -> list[float]:
    """兼容旧调用：使用 dummy encoder 生成单条文本向量。"""
    return DummyTextEncoder().encode_batch([text])[0].tolist()


def _format_part(label: str, value: str | None) -> str | None:
    """格式化非空商品字段，避免把 None 写进文本。"""
    if value is None or not value.strip():
        return None
    return f"{label}: {value.strip()}"


def _stable_vector(seed_text: str, embedding_dim: int) -> np.ndarray:
    """使用 md5 生成稳定向量，避免 Python hash 的随机化。"""
    digest = hashlib.md5(seed_text.encode("utf-8")).digest()
    repeats = (embedding_dim // len(digest)) + 1
    raw = (digest * repeats)[:embedding_dim]
    vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    return vector
