"""数据集状态接口。"""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import (
    default_data_check_report_path,
    default_dataset_stats_path,
    default_metadata_path,
    default_products_path,
    load_json_file,
    load_dataset_stats,
)
from backend.app.data.validators import to_project_relative


class DatasetStatusResponse(BaseModel):
    """数据集准备状态。"""

    prepared: bool
    product_count: int
    valid_image_count: int = 0
    image_count: int = 0
    missing_image_count: int = 0
    metadata_path: str
    processed_products_path: str
    data_check_available: bool = False
    data_check_report_path: str | None = None
    checked_product_count: int | None = None
    invalid_json_line_count: int | None = None
    validation_error_count: int | None = None
    sample_checked_image_count: int | None = None
    sample_missing_image_count: int | None = None
    placeholder: bool
    message: str


router = APIRouter(prefix="/dataset")


@router.get("/status", response_model=DatasetStatusResponse)
def dataset_status() -> DatasetStatusResponse:
    """读取数据准备统计，不扫描大量图片文件。"""
    settings = get_settings()
    metadata_path = default_metadata_path(settings)
    products_path = default_products_path(settings)
    stats_path = default_dataset_stats_path(settings)
    data_check_report_path = default_data_check_report_path(settings)
    metadata_path_text = to_project_relative(metadata_path, settings.project_root)
    products_path_text = to_project_relative(products_path, settings.project_root)
    data_check_report_path_text = to_project_relative(
        data_check_report_path,
        settings.project_root,
    )
    data_check_report = load_json_file(data_check_report_path)
    data_check_fields = _extract_data_check_fields(data_check_report)

    if not products_path.is_file() or not stats_path.is_file():
        return DatasetStatusResponse(
            prepared=False,
            product_count=0,
            valid_image_count=0,
            image_count=0,
            missing_image_count=0,
            metadata_path=metadata_path_text,
            processed_products_path=products_path_text,
            data_check_available=data_check_report is not None,
            data_check_report_path=data_check_report_path_text,
            **data_check_fields,
            placeholder=False,
            message="数据集尚未准备，请先运行 python backend/scripts/prepare_dataset.py。",
        )

    stats = load_dataset_stats(stats_path) or {}
    valid_image_count = int(stats.get("valid_image_count", 0))
    return DatasetStatusResponse(
        prepared=bool(stats.get("prepared", False)),
        product_count=int(stats.get("product_count", 0)),
        valid_image_count=valid_image_count,
        image_count=valid_image_count,
        missing_image_count=int(stats.get("missing_image_count", 0)),
        metadata_path=str(stats.get("metadata_path", metadata_path_text)),
        processed_products_path=str(stats.get("processed_products_path", products_path_text)),
        data_check_available=data_check_report is not None,
        data_check_report_path=data_check_report_path_text,
        **data_check_fields,
        placeholder=False,
        message=str(stats.get("message", "已读取数据集基础统计。")),
    )


def _extract_data_check_fields(report: dict | None) -> dict:
    """从已有 data_check_report.json 中提取轻量状态字段。"""
    if not report:
        return {
            "checked_product_count": None,
            "invalid_json_line_count": None,
            "validation_error_count": None,
            "sample_checked_image_count": None,
            "sample_missing_image_count": None,
        }

    load_stats = report.get("load_stats", {})
    image_path_check = report.get("image_path_check", {})
    return {
        "checked_product_count": int(report.get("product_count", 0)),
        "invalid_json_line_count": int(load_stats.get("skipped_invalid_json_count", 0)),
        "validation_error_count": int(load_stats.get("skipped_validation_error_count", 0)),
        "sample_checked_image_count": int(image_path_check.get("checked_count", 0)),
        "sample_missing_image_count": int(image_path_check.get("missing_count", 0)),
    }
