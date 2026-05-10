"""图片编码器接口和 dummy 实现。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from backend.app.embedding.versioning import (
    DEFAULT_DUMMY_EMBEDDING_DIM,
    DEFAULT_DUMMY_ENCODER_NAME,
    DEFAULT_DUMMY_ENCODER_VERSION,
    REAL_ENCODER_VERSION,
)
from backend.app.embedding.transformers_clip import TransformersCLIPRuntime


class ImageEncoder(Protocol):
    """图片编码器接口，后续真实 CLIP 会实现相同方法。"""

    encoder_name: str
    encoder_version: str
    embedding_dim: int

    def encode_batch(self, image_paths: Sequence[Path]) -> np.ndarray:
        """批量编码图片路径。"""


@dataclass(frozen=True)
class DummyImageEncoder:
    """稳定 dummy 图片编码器，不加载真实模型。"""

    embedding_dim: int = DEFAULT_DUMMY_EMBEDDING_DIM
    encoder_name: str = DEFAULT_DUMMY_ENCODER_NAME
    encoder_version: str = DEFAULT_DUMMY_ENCODER_VERSION

    def encode_batch(self, image_paths: Sequence[Path]) -> np.ndarray:
        """根据 image_path 字符串生成稳定 float32 向量。"""
        vectors: list[np.ndarray] = []
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"图片不存在: {image_path}")
            vectors.append(_stable_vector(image_path.as_posix(), self.embedding_dim))
        if not vectors:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(vectors).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TransformersCLIPImageEncoder:
    """基于 Transformers CLIPModel 的真实图片编码器。"""

    runtime: TransformersCLIPRuntime
    encoder_name: str
    encoder_version: str = REAL_ENCODER_VERSION

    @property
    def embedding_dim(self) -> int:
        """返回真实模型 projection 维度。"""
        return self.runtime.embedding_dim

    def encode_batch(self, image_paths: Sequence[Path]) -> np.ndarray:
        """使用 Pillow 读取 RGB 图片，再调用 CLIP image encoder。"""
        return self.runtime.encode_images(image_paths)


def encode_image_placeholder(image_path: Path) -> list[float]:
    """兼容旧调用：使用 dummy encoder 生成单张图片向量。"""
    return DummyImageEncoder().encode_batch([image_path])[0].tolist()


def _stable_vector(seed_text: str, embedding_dim: int) -> np.ndarray:
    """使用 md5 生成稳定向量，避免 Python hash 的随机化。"""
    digest = hashlib.md5(seed_text.encode("utf-8")).digest()
    repeats = (embedding_dim // len(digest)) + 1
    raw = (digest * repeats)[:embedding_dim]
    vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    return vector
