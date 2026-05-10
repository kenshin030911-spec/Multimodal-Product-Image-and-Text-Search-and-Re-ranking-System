"""FastAPI 应用入口。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes_dataset import router as dataset_router
from backend.app.api.routes_eval import router as eval_router
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_search import router as search_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging


def create_app() -> FastAPI:
    """创建 FastAPI 应用并注册第一阶段占位路由。"""
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Fashion Multimodal Search 第一阶段工程骨架。",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(search_router, prefix="/api", tags=["search"])
    app.include_router(dataset_router, prefix="/api", tags=["dataset"])
    app.include_router(eval_router, prefix="/api", tags=["evaluation"])

    settings.raw_images_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        settings.static_images_url_prefix,
        StaticFiles(directory=settings.raw_images_dir),
        name="static_images",
    )

    return app


app = create_app()
