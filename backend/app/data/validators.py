"""数据校验工具。"""

from decimal import Decimal, InvalidOperation
from pathlib import Path


def validate_image_path(image_path: Path) -> bool:
    """检查图片路径是否存在；后续会扩展损坏图片校验。"""
    return image_path.is_file()


def normalize_product_id(value: object) -> tuple[str | None, str | None]:
    """把原始商品 ID 统一成字符串，返回值和错误原因。"""
    if value is None:
        return None, "missing"

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return None, "missing"

    try:
        number = Decimal(text)
    except InvalidOperation:
        return None, "invalid"

    if number < 0 or number != number.to_integral_value():
        return None, "invalid"

    return str(int(number)), None


def build_image_path(images_dir: Path, product_id: str) -> Path:
    """按默认规则拼接商品图片路径。"""
    return images_dir / f"{product_id}.jpg"


def to_project_relative(path: Path, project_root: Path) -> str:
    """把路径转成相对项目根目录的字符串，避免对外暴露绝对路径。"""
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return path.name if path.is_absolute() else path.as_posix()
