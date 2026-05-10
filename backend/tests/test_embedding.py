"""embedding 模块 smoke test。"""

import os
from pathlib import Path

import numpy as np
import pytest

from backend.app.data.dataset_loader import write_products
from backend.app.embedding.embedding_builder import build_embeddings
from backend.app.embedding.embedding_store import load_embedding_bundle, load_embeddings_placeholder
from backend.app.embedding.image_encoder import DummyImageEncoder
from backend.app.embedding.text_encoder import DummyTextEncoder, build_product_text
from backend.app.embedding.versioning import DEFAULT_DUMMY_EMBEDDING_DIM, get_embedding_version
from backend.app.schemas.product import ProductItem


def test_dummy_encoders_are_stable_float32(tmp_path: Path) -> None:
    """dummy encoder 输出稳定 float32 向量。"""
    image_path = tmp_path / "query.jpg"
    image_path.write_bytes(b"fake-image")
    image_encoder = DummyImageEncoder()
    text_encoder = DummyTextEncoder()

    image_a = image_encoder.encode_batch([image_path])
    image_b = image_encoder.encode_batch([image_path])
    text_a = text_encoder.encode_batch(["black dress"])
    text_b = text_encoder.encode_batch(["black dress"])

    assert image_a.shape == (1, DEFAULT_DUMMY_EMBEDDING_DIM)
    assert text_a.shape == (1, DEFAULT_DUMMY_EMBEDDING_DIM)
    assert image_a.dtype == np.float32
    assert text_a.dtype == np.float32
    np.testing.assert_array_equal(image_a, image_b)
    np.testing.assert_array_equal(text_a, text_b)


def test_product_text_uses_fields_without_none() -> None:
    """商品文本会组合关键字段，并跳过缺失字段。"""
    product = _product("10001", gender=None, usage="Casual")

    text = build_product_text(product)

    assert "Product 10001" in text
    assert "article type: Shirts" in text
    assert "color: Black" in text
    assert "usage: Casual" in text
    assert "None" not in text


def test_build_embeddings_saves_and_loads_aligned_bundle(tmp_path: Path) -> None:
    """同时生成 image/text 时，只保留两个模态都成功的商品并保持顺序一致。"""
    paths = _build_embedding_inputs(tmp_path)

    result = build_embeddings(
        products_path=paths["products_path"],
        output_dir=paths["output_dir"],
        batch_size=2,
        encoder_name="dummy",
        overwrite=True,
        project_root=tmp_path,
    )
    bundle = load_embedding_bundle(paths["output_dir"])

    assert result.product_count == 2
    assert bundle.image_embeddings is not None
    assert bundle.text_embeddings is not None
    assert bundle.image_embeddings.shape == (2, DEFAULT_DUMMY_EMBEDDING_DIM)
    assert bundle.text_embeddings.shape == (2, DEFAULT_DUMMY_EMBEDDING_DIM)
    assert bundle.image_embeddings.dtype == np.float32
    assert bundle.text_embeddings.dtype == np.float32
    assert bundle.meta["product_ids"] == ["10001", "10002"]
    assert bundle.meta["product_id_to_index"] == {"10001": 0, "10002": 1}
    assert bundle.meta["generated"] == {"image": True, "text": True}
    assert bundle.meta["failed_images"] == [
        {"product_id": "10003", "reason": "image_path_not_found"}
    ]
    assert bundle.meta["skipped_products"] == [
        {"product_id": "10003", "reason": "image_failed"}
    ]
    assert not Path(bundle.meta["output_dir"]).is_absolute()
    assert not Path(bundle.meta["image_embedding_file"]).is_absolute()
    assert not Path(bundle.meta["text_embedding_file"]).is_absolute()


def test_build_embeddings_limit_and_text_only(tmp_path: Path) -> None:
    """limit 生效，text-only 不检查图片是否存在。"""
    paths = _build_embedding_inputs(tmp_path)

    result = build_embeddings(
        products_path=paths["products_path"],
        output_dir=paths["output_dir"],
        limit=2,
        text_only=True,
        overwrite=True,
        project_root=tmp_path,
    )

    assert result.product_count == 2
    assert result.bundle.image_embeddings is None
    assert result.bundle.text_embeddings is not None
    assert result.bundle.text_embeddings.shape == (2, DEFAULT_DUMMY_EMBEDDING_DIM)
    assert result.bundle.meta["generated"] == {"image": False, "text": True}
    assert result.bundle.meta["failed_images"] == []


def test_build_embeddings_image_only_and_overwrite_guard(tmp_path: Path) -> None:
    """image-only 只要求图片成功，默认不覆盖已有输出。"""
    paths = _build_embedding_inputs(tmp_path)

    result = build_embeddings(
        products_path=paths["products_path"],
        output_dir=paths["output_dir"],
        image_only=True,
        overwrite=True,
        project_root=tmp_path,
    )

    assert result.product_count == 2
    assert result.bundle.image_embeddings is not None
    assert result.bundle.text_embeddings is None
    assert result.bundle.meta["generated"] == {"image": True, "text": False}

    with pytest.raises(FileExistsError):
        build_embeddings(
            products_path=paths["products_path"],
            output_dir=paths["output_dir"],
            image_only=True,
            overwrite=False,
            project_root=tmp_path,
        )


def test_embedding_helpers_and_encoder_validation() -> None:
    """兼容旧 helper，并校验未知 encoder 会报清晰错误。"""
    assert load_embeddings_placeholder() == {}
    assert get_embedding_version("dummy") == "dummy-v1"
    assert get_embedding_version("clip").startswith("transformers-clip-v1:")
    with pytest.raises(ValueError):
        get_embedding_version("unknown")


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="真实模型 smoke test 默认跳过，避免 pytest 下载模型。",
)
def test_real_transformers_clip_encoder_smoke(tmp_path: Path) -> None:
    """显式启用时，验证 Transformers CLIP image/text encoder 基本输出。"""
    from PIL import Image

    from backend.app.embedding.image_encoder import TransformersCLIPImageEncoder
    from backend.app.embedding.text_encoder import TransformersCLIPTextEncoder
    from backend.app.embedding.transformers_clip import TransformersCLIPRuntime
    from backend.app.embedding.versioning import DEFAULT_REAL_MODEL_NAME

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(32, 64, 96)).save(image_path)
    runtime = TransformersCLIPRuntime(
        model_name=DEFAULT_REAL_MODEL_NAME,
        device="cpu",
        normalize_embeddings=True,
    )
    image_encoder = TransformersCLIPImageEncoder(runtime, encoder_name="fashion-clip")
    text_encoder = TransformersCLIPTextEncoder(runtime, encoder_name="fashion-clip")

    image_vectors = image_encoder.encode_batch([image_path])
    text_vectors = text_encoder.encode_batch(["black casual shirt"])

    assert image_vectors.shape == (1, runtime.embedding_dim)
    assert text_vectors.shape == (1, runtime.embedding_dim)
    assert image_vectors.dtype == np.float32
    assert text_vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(image_vectors, axis=1), [1.0], atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(text_vectors, axis=1), [1.0], atol=1e-4)


def _build_embedding_inputs(tmp_path: Path) -> dict[str, Path]:
    """创建临时 products.jsonl 和两张存在的图片。"""
    image_dir = tmp_path / "data" / "raw" / "images"
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "data" / "embeddings"
    image_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    (image_dir / "10001.jpg").write_bytes(b"fake-image-1")
    (image_dir / "10002.jpg").write_bytes(b"fake-image-2")

    products_path = processed_dir / "products.jsonl"
    write_products(
        [
            _product("10001"),
            _product("10002", article_type="Dresses", gender="Women"),
            _product("10003", image_path="data/raw/images/missing.jpg"),
        ],
        products_path,
    )

    return {
        "products_path": products_path,
        "output_dir": output_dir,
    }


def _product(
    product_id: str,
    article_type: str = "Shirts",
    gender: str | None = "Men",
    usage: str | None = "Casual",
    image_path: str | None = None,
) -> ProductItem:
    """创建测试用商品。"""
    return ProductItem(
        product_id=product_id,
        title=f"Product {product_id}",
        gender=gender,
        master_category="Apparel",
        sub_category="Topwear",
        article_type=article_type,
        base_colour="Black",
        season="Fall",
        year=2011,
        usage=usage,
        image_path=image_path or f"data/raw/images/{product_id}.jpg",
        freshness_score=0.5,
    )
