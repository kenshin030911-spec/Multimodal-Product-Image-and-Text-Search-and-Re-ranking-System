"""向量版本记录。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_DUMMY_EMBEDDING_DIM = 8
DEFAULT_DUMMY_ENCODER_NAME = "dummy"
DEFAULT_DUMMY_ENCODER_VERSION = "dummy-v1"
DEFAULT_REAL_MODEL_NAME = "patrickjohncyh/fashion-clip"
REAL_ENCODER_VERSION = "transformers-clip-v1"
SUPPORTED_REAL_ENCODER_NAMES = ("fashion-clip", "clip")
SUPPORTED_ENCODER_NAMES = (DEFAULT_DUMMY_ENCODER_NAME, *SUPPORTED_REAL_ENCODER_NAMES)


@dataclass(frozen=True)
class EmbeddingVersionInfo:
    """记录当前 embedding 编码器版本。"""

    encoder_name: str
    encoder_version: str
    embedding_dim: int
    is_dummy: bool
    created_at: str
    framework: str | None = None
    model_name: str | None = None
    device: str | None = None
    torch_dtype: str | None = None
    normalize_embeddings: bool = False
    model_revision: str | None = None

    def model_dump(self) -> dict[str, Any]:
        """转成可写入 JSON 的字典。"""
        return {
            "encoder_name": self.encoder_name,
            "encoder_version": self.encoder_version,
            "embedding_dim": self.embedding_dim,
            "is_dummy": self.is_dummy,
            "created_at": self.created_at,
            "framework": self.framework,
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "normalize_embeddings": self.normalize_embeddings,
            "model_revision": self.model_revision,
        }


def get_embedding_version(
    encoder_name: str = DEFAULT_DUMMY_ENCODER_NAME,
    model_name: str | None = None,
) -> str:
    """返回编码器版本号。"""
    if encoder_name == DEFAULT_DUMMY_ENCODER_NAME:
        return DEFAULT_DUMMY_ENCODER_VERSION
    if encoder_name in SUPPORTED_REAL_ENCODER_NAMES:
        return f"{REAL_ENCODER_VERSION}:{model_name or DEFAULT_REAL_MODEL_NAME}"
    raise ValueError(f"不支持的 encoder_name: {encoder_name}")


def build_embedding_version_info(
    encoder_name: str = DEFAULT_DUMMY_ENCODER_NAME,
    embedding_dim: int = DEFAULT_DUMMY_EMBEDDING_DIM,
    framework: str | None = None,
    model_name: str | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
    normalize_embeddings: bool = False,
    model_revision: str | None = None,
) -> EmbeddingVersionInfo:
    """创建 embedding 版本信息。"""
    return EmbeddingVersionInfo(
        encoder_name=encoder_name,
        encoder_version=get_embedding_version(encoder_name, model_name=model_name),
        embedding_dim=embedding_dim,
        is_dummy=encoder_name == DEFAULT_DUMMY_ENCODER_NAME,
        created_at=datetime.now(timezone.utc).isoformat(),
        framework=framework,
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        normalize_embeddings=normalize_embeddings,
        model_revision=model_revision,
    )
