"""应用配置。

第一阶段只管理路径和基础开关，不连接真实数据、模型或索引。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# config.py 位于 backend/app/core/ 下，向上 3 层是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _env_list(name: str, default: str) -> list[str]:
    """把逗号分隔的环境变量解析成字符串列表。"""
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _env_path(name: str, default: str) -> Path:
    """把环境变量路径解析到项目根目录下，避免硬编码本机绝对路径。"""
    raw_value = os.getenv(name, default)
    path = Path(raw_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    """把常见布尔环境变量写法解析成 bool。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_url_prefix(name: str, default: str) -> str:
    """读取 URL 前缀，确保以 / 开头且不以 / 结尾。"""
    raw_value = os.getenv(name, default).strip() or default
    if not raw_value.startswith("/"):
        raw_value = f"/{raw_value}"
    return raw_value.rstrip("/") or "/"


@dataclass(frozen=True)
class Settings:
    """集中保存应用配置，后续模块只依赖这里的路径。"""

    app_name: str
    app_version: str
    project_root: Path
    data_dir: Path
    raw_images_dir: Path
    raw_metadata_dir: Path
    processed_data_dir: Path
    embeddings_dir: Path
    index_dir: Path
    model_dir: Path
    base_encoder_dir: Path
    reranker_dir: Path
    output_dir: Path
    eval_reports_dir: Path
    search_cases_dir: Path
    failure_cases_dir: Path
    uploads_tmp_dir: Path
    static_images_url_prefix: str
    top_k: int
    use_rerank: bool
    max_upload_size_mb: int
    allowed_image_types: list[str]
    cors_origins: list[str]

    @property
    def max_upload_size_bytes(self) -> int:
        """把 MB 配置换算成字节，供上传接口校验。"""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """读取配置并缓存，避免各模块重复解析环境变量。"""
    data_dir = _env_path("DATA_DIR", "data")
    model_dir = _env_path("MODEL_DIR", "models")
    output_dir = _env_path("OUTPUT_DIR", "outputs")

    return Settings(
        app_name=os.getenv("APP_NAME", "Fashion Multimodal Search"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        raw_images_dir=data_dir / "raw" / "images",
        raw_metadata_dir=data_dir / "raw" / "metadata",
        processed_data_dir=data_dir / "processed",
        embeddings_dir=data_dir / "embeddings",
        index_dir=data_dir / "index",
        model_dir=model_dir,
        base_encoder_dir=model_dir / "base_encoder",
        reranker_dir=model_dir / "reranker",
        output_dir=output_dir,
        eval_reports_dir=output_dir / "eval_reports",
        search_cases_dir=output_dir / "search_cases",
        failure_cases_dir=output_dir / "failure_cases",
        uploads_tmp_dir=_env_path("UPLOADS_TMP_DIR", "uploads/tmp"),
        static_images_url_prefix=_env_url_prefix(
            "STATIC_IMAGES_URL_PREFIX",
            "/static/images",
        ),
        top_k=int(os.getenv("TOP_K", "20")),
        use_rerank=_env_bool("USE_RERANK", True),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "5")),
        allowed_image_types=_env_list(
            "ALLOWED_IMAGE_TYPES",
            "image/jpeg,image/png,image/webp",
        ),
        cors_origins=_env_list(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ),
    )
