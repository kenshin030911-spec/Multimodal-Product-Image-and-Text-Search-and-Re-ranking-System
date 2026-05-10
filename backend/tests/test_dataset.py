"""数据准备模块 smoke test。"""

import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import (
    load_products,
    load_products_with_stats,
    write_dataset_stats,
    write_products,
)
from backend.app.data.processed_checker import (
    build_processed_data_report,
    format_processed_data_report,
    write_processed_data_report,
)
from backend.app.data.processor import prepare_dataset
from backend.app.data.split_manager import get_dataset_splits
from backend.app.data.validators import normalize_product_id, validate_image_path
from backend.app.main import app
from backend.app.schemas.product import ProductItem


def test_prepare_dataset_outputs_products_and_stats(tmp_path: Path) -> None:
    """临时小数据可以完成标准化、缺图跳过和 freshness 计算。"""
    paths = _build_tiny_dataset(tmp_path)

    stats = prepare_dataset(
        metadata_path=paths["metadata_path"],
        images_dir=paths["images_dir"],
        products_path=paths["products_path"],
        stats_path=paths["stats_path"],
        project_root=tmp_path,
    )
    products = load_products(paths["products_path"])

    assert stats["prepared"] is True
    assert stats["raw_row_count"] == 7
    assert stats["output_product_count"] == 5
    assert stats["missing_image_count"] == 1
    assert stats["skipped_missing_image_count"] == 1
    assert stats["skipped_missing_id_count"] == 1
    assert stats["missing_year_count"] == 1
    assert stats["invalid_year_count"] == 1

    assert len(products) == 5
    assert all(isinstance(product, ProductItem) for product in products)
    assert all(0.0 <= product.freshness_score <= 1.0 for product in products)

    product_by_id = {product.product_id: product for product in products}
    assert "15970" in product_by_id
    assert "15970.0" not in product_by_id
    assert product_by_id["15970"].image_path == "data/raw/images/15970.jpg"
    assert product_by_id["15970"].article_type == "Shirts"
    assert product_by_id["15972"].year is None
    assert product_by_id["15972"].freshness_score == 0.5
    assert product_by_id["15973"].article_type is None
    assert product_by_id["15973"].base_colour is None
    assert product_by_id["15975"].year is None
    assert product_by_id["15975"].freshness_score == 0.5


def test_dataset_status_without_processed_files(tmp_path: Path, monkeypatch) -> None:
    """没有 processed 文件时，状态接口返回未准备。"""
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.get("/api/dataset/status")
    get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is False
    assert data["product_count"] == 0
    assert data["placeholder"] is False
    assert "prepare_dataset.py" in data["message"]


def test_dataset_status_reads_stats_file(tmp_path: Path, monkeypatch) -> None:
    """有 stats 文件时，状态接口直接返回基础统计。"""
    paths = _build_tiny_dataset(tmp_path)
    prepare_dataset(
        metadata_path=paths["metadata_path"],
        images_dir=paths["images_dir"],
        products_path=paths["products_path"],
        stats_path=paths["stats_path"],
        project_root=tmp_path,
    )

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.get("/api/dataset/status")
    get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is True
    assert data["product_count"] == 5
    assert data["valid_image_count"] == 5
    assert data["missing_image_count"] == 1
    assert data["placeholder"] is False


def test_load_products_with_stats_handles_limit_and_bad_lines(tmp_path: Path) -> None:
    """products.jsonl 加载支持 limit，并统计坏 JSON 和 schema 错误。"""
    products_path = tmp_path / "products.jsonl"
    valid_products = [
        _product("10001", "Shirts"),
        _product("10002", "Dresses"),
        _product("10003", "Sports Shoes"),
    ]
    write_products(valid_products, products_path)
    with products_path.open("a", encoding="utf-8") as file:
        file.write("\n")
        file.write("{bad json}\n")
        file.write(json.dumps({"title": "missing product id"}) + "\n")

    limited_products = load_products(products_path, limit=2)
    result = load_products_with_stats(products_path)

    assert len(limited_products) == 2
    assert result.file_exists is True
    assert result.total_lines == 6
    assert result.loaded_count == 3
    assert result.skipped_empty_line_count == 1
    assert result.skipped_invalid_json_count == 1
    assert result.skipped_validation_error_count == 1


def test_processed_checker_builds_report_and_checks_images(tmp_path: Path) -> None:
    """processed 检查报告包含统计、抽样和图片路径抽查结果。"""
    paths = _build_processed_dataset(tmp_path)
    report = build_processed_data_report(
        products_path=paths["products_path"],
        report_path=paths["report_path"],
        sample_size=2,
        top_n=2,
        image_check_size=3,
        seed=7,
        project_root=tmp_path,
    )
    write_processed_data_report(report, paths["report_path"])
    saved_report = json.loads(paths["report_path"].read_text(encoding="utf-8"))
    text_report = format_processed_data_report(report)

    assert report["product_count"] == 3
    assert report["load_stats"]["skipped_invalid_json_count"] == 1
    assert report["load_stats"]["skipped_validation_error_count"] == 1
    assert report["missing_field_counts"]["gender"] == 1
    assert report["top_values"]["article_type"][0]["value"] == "Shirts"
    assert report["year_stats"]["min_year"] == 2011
    assert report["year_stats"]["max_year"] == 2015
    assert report["freshness_score_stats"]["min_score"] == 0.2
    assert report["freshness_score_stats"]["max_score"] == 0.8
    assert report["image_path_check"]["checked_count"] == 3
    assert report["image_path_check"]["missing_count"] == 1
    assert saved_report["product_count"] == 3
    assert "Processed Data Check" in text_report


def test_dataset_status_reads_data_check_report(tmp_path: Path, monkeypatch) -> None:
    """dataset status 在报告存在时返回增强统计，但不扫描 products。"""
    paths = _build_processed_dataset(tmp_path)
    write_dataset_stats(
        {
            "prepared": True,
            "product_count": 3,
            "valid_image_count": 3,
            "missing_image_count": 0,
            "metadata_path": "data/raw/metadata/styles.csv",
            "processed_products_path": "data/processed/products.jsonl",
            "message": "数据集准备完成。",
        },
        tmp_path / "data" / "processed" / "dataset_stats.json",
    )
    report = build_processed_data_report(
        products_path=paths["products_path"],
        report_path=paths["report_path"],
        image_check_size=3,
        project_root=tmp_path,
    )
    write_processed_data_report(report, paths["report_path"])

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    get_settings.cache_clear()

    client = TestClient(app)
    response = client.get("/api/dataset/status")
    get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is True
    assert data["data_check_available"] is True
    assert data["checked_product_count"] == 3
    assert data["invalid_json_line_count"] == 1
    assert data["validation_error_count"] == 1
    assert data["sample_checked_image_count"] == 3
    assert data["sample_missing_image_count"] == 1
    assert not Path(data["data_check_report_path"]).is_absolute()


def test_dataset_helpers_do_not_need_real_data() -> None:
    """保留第一轮轻量 helper 测试，不依赖真实数据集。"""
    assert load_products(Path("not-exist.jsonl")) == []
    assert get_dataset_splits() == {"train": [], "val": [], "test": []}
    assert validate_image_path(Path("not-exist.jpg")) is False
    assert normalize_product_id(15970.0) == ("15970", None)
    assert normalize_product_id("") == (None, "missing")


def _build_tiny_dataset(tmp_path: Path) -> dict[str, Path]:
    """创建 7 行临时 metadata 和 5 张 1x1 jpg 图片。"""
    metadata_dir = tmp_path / "data" / "raw" / "metadata"
    images_dir = tmp_path / "data" / "raw" / "images"
    processed_dir = tmp_path / "data" / "processed"
    metadata_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    metadata_path = metadata_dir / "styles.csv"
    metadata_path.write_text(
        "\n".join(
            [
                "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName",
                "15970.0,Men,Apparel,Topwear,Shirts,Navy Blue,Fall,2011,Casual,Turtle Check Men Navy Blue Shirt",
                "15971,Men,Apparel,Topwear,Shirts,Black,Fall,2012,Casual,Missing Image Shirt",
                "15972,Women,Apparel,Dresses,Dresses,Red,Summer,,Casual,Red Dress",
                "15973,Men,Apparel,Topwear,, ,Winter,2010,Casual,",
                "15974,Women,Footwear,Shoes,Sports Shoes,White,Summer,2015,Sports,White Sneakers",
                ",Men,Apparel,Topwear,Shirts,Blue,Fall,2011,Casual,Missing Id Shirt",
                "15975,Men,Apparel,Topwear,Shirts,Blue,Fall,2099,Casual,Future Year Shirt",
            ]
        ),
        encoding="utf-8",
    )

    for product_id in ("15970", "15972", "15973", "15974", "15975"):
        _create_test_image(images_dir / f"{product_id}.jpg")

    return {
        "metadata_path": metadata_path,
        "images_dir": images_dir,
        "products_path": processed_dir / "products.jsonl",
        "stats_path": processed_dir / "dataset_stats.json",
    }


def _create_test_image(path: Path) -> None:
    """创建真实 1x1 jpg，方便后续扩展图片校验。"""
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(path, format="JPEG")


def _build_processed_dataset(tmp_path: Path) -> dict[str, Path]:
    """创建临时 processed 数据和部分图片。"""
    images_dir = tmp_path / "data" / "raw" / "images"
    processed_dir = tmp_path / "data" / "processed"
    report_dir = tmp_path / "outputs" / "data_checks"
    images_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    _create_test_image(images_dir / "10001.jpg")
    _create_test_image(images_dir / "10002.jpg")

    products_path = processed_dir / "products.jsonl"
    write_products(
        [
            _product("10001", "Shirts", gender="Men", year=2011, freshness_score=0.2),
            _product("10002", "Shirts", gender=None, year=2015, freshness_score=0.8),
            _product(
                "10003",
                "Dresses",
                gender="Women",
                year=None,
                freshness_score=0.5,
                image_path="data/raw/images/missing.jpg",
            ),
        ],
        products_path,
    )
    with products_path.open("a", encoding="utf-8") as file:
        file.write("{bad json}\n")
        file.write(json.dumps({"title": "missing product id"}) + "\n")

    return {
        "products_path": products_path,
        "report_path": report_dir / "data_check_report.json",
    }


def _product(
    product_id: str,
    article_type: str,
    gender: str | None = "Men",
    year: int | None = 2011,
    freshness_score: float = 0.5,
    image_path: str | None = None,
) -> ProductItem:
    """创建测试用 ProductItem。"""
    return ProductItem(
        product_id=product_id,
        title=f"Product {product_id}",
        gender=gender,
        master_category="Apparel",
        sub_category="Topwear",
        article_type=article_type,
        base_colour="Black",
        season="Fall",
        year=year,
        usage="Casual",
        image_path=image_path or f"data/raw/images/{product_id}.jpg",
        freshness_score=freshness_score,
    )
