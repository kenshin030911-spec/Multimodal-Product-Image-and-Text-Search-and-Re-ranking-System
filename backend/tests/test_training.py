"""Weak-supervised reranker dataset builder tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.data.dataset_loader import write_products
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.schemas.evaluation import EvalQuery
from backend.app.schemas.product import ProductItem
from backend.app.training import sample_builder
from backend.app.training.dataset_splitter import split_query_ids
from backend.app.training.feature_exporter import FEATURE_NAMES, rerank_feature_to_dict
from backend.app.training.sample_builder import (
    LABEL_MAPPING,
    META_FILE,
    TRAIN_FILE,
    VALID_FILE,
    build_reranker_dataset,
    label_from_relevance,
)


def test_label_strategy_keeps_relevance_grade_policy() -> None:
    """Explicit positives override metadata, while weak grades 3/2 are positive."""
    explicit = EvalQuery(
        query_id="Q1",
        query_text="red shoes",
        positive_product_ids=["p1"],
        label_source="manual",
    )
    weak = EvalQuery(query_id="Q2", query_text="black shirt", label_source="weak_metadata")

    assert label_from_relevance(explicit, "p1", 0) == 1
    assert label_from_relevance(explicit, "p2", 3) == 0
    assert label_from_relevance(weak, "p3", 3) == 1
    assert label_from_relevance(weak, "p4", 2) == 1
    assert label_from_relevance(weak, "p5", 1) == 0
    assert label_from_relevance(weak, "p6", 0) == 0
    assert LABEL_MAPPING["weak_metadata"]["1"] == 0


def test_feature_exporter_has_stable_complete_float_fields() -> None:
    """Feature exporter preserves fixed names and JSON-safe float values."""
    feature = sample_builder.build_rerank_feature(
        _result(
            "p1",
            score=0.7,
            title="Black Casual Shirts",
            article_type="Shirts",
            base_colour="Black",
            gender="Men",
            usage="Casual",
            sub_category="Topwear",
        ),
        query_text="black casual shirts men topwear",
    )

    exported = rerank_feature_to_dict(feature)

    assert tuple(exported.keys()) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in exported.values())
    assert exported["article_type_match"] == 1.0
    assert exported["color_match"] == 1.0


def test_split_query_ids_is_query_level_and_handles_small_sets() -> None:
    """Split helper keeps each query in exactly one split."""
    one = split_query_ids(["Q1"], train_ratio=0.8, seed=42)
    many = split_query_ids(["Q1", "Q2", "Q3", "Q4"], train_ratio=0.5, seed=42)

    assert one == {"Q1": "train"}
    assert set(many) == {"Q1", "Q2", "Q3", "Q4"}
    assert set(many.values()) == {"train", "valid"}
    with pytest.raises(ValueError):
        split_query_ids(["Q1"], train_ratio=1.0)


def test_build_reranker_dataset_writes_files_and_samples(tmp_path: Path, monkeypatch) -> None:
    """Dataset builder writes train/valid JSONL and meta with sampled negatives."""
    products_path = _write_products(tmp_path)
    eval_queries_path = _write_eval_queries(tmp_path)

    monkeypatch.setattr(sample_builder, "search_text_to_image", _fake_search_text_to_image)

    result = build_reranker_dataset(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "data" / "processed",
        candidate_k=5,
        max_queries=3,
        max_positives_per_query=20,
        max_negatives_per_query=1,
        min_positives_per_query=1,
        train_ratio=0.5,
        seed=7,
        device="cpu",
        overwrite=True,
        project_root=tmp_path,
    )

    train_path = tmp_path / "data" / "processed" / TRAIN_FILE
    valid_path = tmp_path / "data" / "processed" / VALID_FILE
    meta_path = tmp_path / "data" / "processed" / META_FILE
    train_rows = _read_jsonl(train_path)
    valid_rows = _read_jsonl(valid_path)
    all_rows = train_rows + valid_rows
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert train_path.is_file()
    assert valid_path.is_file()
    assert meta_path.is_file()
    assert result.meta["query_count"] == 3
    assert meta["used_query_count"] == 2
    assert meta["skipped_no_positive_query_count"] == 1
    assert meta["positive_count"] == 3
    assert meta["negative_count"] == 2
    assert meta["max_positives_per_query"] == 20
    assert meta["max_negatives_per_query"] == 1
    assert meta["min_positives_per_query"] == 1
    assert meta["query_generation_mode"] == "eval_queries_jsonl"
    assert meta["query_templates"] == "basic"
    assert meta["queries_per_product"] == 2
    assert meta["max_query_variants"] == 12
    assert meta["query_template_names"] == []
    assert "not real user search logs" in meta["query_generation_note"]
    assert "relevance_grade descending" in meta["positive_sampling_policy"]
    assert "hard negatives" in meta["negative_sampling_policy"]
    assert meta["feature_names"] == list(FEATURE_NAMES)
    assert "not large-scale manual annotations" in meta["note"]

    query_ids_by_split: dict[str, set[str]] = {"train": set(), "valid": set()}
    for row in all_rows:
        query_ids_by_split[row["split"]].add(row["query_id"])
        assert set(row["features"]) == set(FEATURE_NAMES)
        assert row["split"] in {"train", "valid"}
        assert row["relevance_grade"] in {0, 1, 2, 3}
        assert isinstance(row["recall_score"], float)

    assert query_ids_by_split["train"].isdisjoint(query_ids_by_split["valid"])

    q2_rows = [row for row in all_rows if row["query_id"] == "Q2"]
    q2_by_product = {row["product_id"]: row for row in q2_rows}
    assert q2_by_product["p3"]["label"] == 1
    assert q2_by_product["p4"]["label"] == 1
    assert q2_by_product["p5"]["relevance_grade"] == 1
    assert q2_by_product["p5"]["label"] == 0
    assert "p6" not in q2_by_product

    q1_rows = [row for row in all_rows if row["query_id"] == "Q1"]
    assert {row["product_id"] for row in q1_rows} == {"p1", "p2"}
    assert {row["label_source"] for row in q1_rows} == {"manual"}


def test_build_reranker_dataset_truncates_positives_by_grade_then_rank(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Positive sampling keeps higher relevance grades before lower-grade positives."""
    products_path = _write_products(tmp_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps(
            {
                "query_id": "QPOS",
                "query_text": "positive sampling query",
                "expected_article_type": "Shirts",
                "expected_base_colour": "Black",
                "expected_gender": "Men",
                "expected_sub_category": "Topwear",
                "label_source": "weak_metadata",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sample_builder, "search_text_to_image", _fake_search_text_to_image)

    result = build_reranker_dataset(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "data" / "processed",
        candidate_k=5,
        max_queries=1,
        max_positives_per_query=2,
        max_negatives_per_query=0,
        min_positives_per_query=1,
        train_ratio=0.8,
        seed=42,
        device="cpu",
        overwrite=True,
        project_root=tmp_path,
    )

    rows = result.train_samples + result.valid_samples

    assert [row["product_id"] for row in rows] == ["p3", "p8"]
    assert [row["relevance_grade"] for row in rows] == [3, 3]
    assert result.meta["positive_count"] == 2
    assert result.meta["negative_count"] == 0
    assert result.meta["max_positives_per_query"] == 2


def test_build_reranker_dataset_accepts_augmented_generated_queries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """When eval_queries.jsonl is absent, builder can use augmented query generation."""
    products_path = _write_products(tmp_path)

    def fake_search_text_to_image(**kwargs) -> RetrievalResponse:
        query_text = kwargs["query_text"]
        return _response(
            query_text=query_text,
            results=[
                _result(
                    "p3",
                    score=0.90,
                    rank=1,
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Men",
                    sub_category="Topwear",
                ),
                _result(
                    "p6",
                    score=0.80,
                    rank=2,
                    article_type="Shoes",
                    base_colour="Red",
                    gender="Women",
                    sub_category="Footwear",
                ),
            ],
        )

    monkeypatch.setattr(sample_builder, "search_text_to_image", fake_search_text_to_image)

    result = build_reranker_dataset(
        eval_queries_path=tmp_path / "data" / "processed" / "missing_eval_queries.jsonl",
        products_path=products_path,
        output_dir=tmp_path / "data" / "processed" / "augmented",
        candidate_k=2,
        max_queries=2,
        max_positives_per_query=2,
        max_negatives_per_query=1,
        min_positives_per_query=1,
        train_ratio=0.8,
        seed=42,
        device="cpu",
        query_templates="augmented",
        queries_per_product=2,
        max_query_variants=12,
        overwrite=True,
        project_root=tmp_path,
    )

    rows = result.train_samples + result.valid_samples
    assert result.meta["query_generation_mode"] == "augmented"
    assert result.meta["query_templates"] == "augmented"
    assert result.meta["queries_per_product"] == 2
    assert "article_type_only" in result.meta["query_template_names"]
    assert result.meta["used_query_count"] == 2
    assert rows
    assert all(row["query_text"] for row in rows)


def test_build_reranker_dataset_rejects_existing_outputs(tmp_path: Path, monkeypatch) -> None:
    """Builder refuses to overwrite train/valid/meta without --overwrite."""
    products_path = _write_products(tmp_path)
    eval_queries_path = _write_eval_queries(tmp_path)
    output_dir = tmp_path / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / TRAIN_FILE).write_text("", encoding="utf-8")
    monkeypatch.setattr(sample_builder, "search_text_to_image", _fake_search_text_to_image)

    with pytest.raises(FileExistsError):
        build_reranker_dataset(
            eval_queries_path=eval_queries_path,
            products_path=products_path,
            output_dir=output_dir,
            candidate_k=5,
            max_queries=1,
            overwrite=False,
            project_root=tmp_path,
        )


def test_build_reranker_dataset_help() -> None:
    """build_reranker_dataset.py --help is available without loading models."""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/build_reranker_dataset.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--max-positives-per-query" in completed.stdout
    assert "--max-negatives-per-query" in completed.stdout
    assert "--query-templates" in completed.stdout
    assert "--queries-per-product" in completed.stdout
    assert "--max-query-variants" in completed.stdout
    assert "--overwrite" in completed.stdout


def _fake_search_text_to_image(**kwargs) -> RetrievalResponse:
    """Return deterministic retrieval results keyed by query text."""
    query_text = kwargs["query_text"]
    if query_text == "explicit positive query":
        results = [
            _result("p1", score=0.90, rank=1, article_type="Shoes", base_colour="Red"),
            _result("p2", score=0.80, rank=2, article_type="Shirts", base_colour="Black"),
            _result("p3", score=0.70, rank=3, article_type="Shirts", base_colour="Black"),
        ]
    elif query_text == "weak metadata query":
        results = [
            _result("p3", score=0.90, rank=1, article_type="Shirts", base_colour="Black", gender="Men"),
            _result("p4", score=0.80, rank=2, article_type="Shirts", base_colour="Black", gender="Women"),
            _result("p5", score=0.70, rank=3, article_type="Shirts", base_colour="Blue", gender="Men"),
            _result("p6", score=0.60, rank=4, article_type="Shoes", base_colour="Red", gender="Women"),
        ]
    elif query_text == "positive sampling query":
        results = [
            _result("p4", score=0.99, rank=1, article_type="Shirts", base_colour="Black", gender="Women"),
            _result("p5", score=0.90, rank=2, article_type="Shirts", base_colour="Blue", gender="Men"),
            _result("p3", score=0.80, rank=3, article_type="Shirts", base_colour="Black", gender="Men"),
            _result("p8", score=0.95, rank=4, article_type="Shirts", base_colour="Black", gender="Men"),
            _result("p6", score=0.60, rank=5, article_type="Shoes", base_colour="Red", gender="Women"),
        ]
    else:
        results = [
            _result("p6", score=0.90, rank=1, article_type="Shoes", base_colour="Red", gender="Women"),
            _result("p7", score=0.80, rank=2, article_type="Watches", base_colour="Silver", gender="Women"),
        ]
    return _response(query_text=query_text, results=results)


def _write_eval_queries(tmp_path: Path) -> Path:
    """Write three eval queries: explicit, weak metadata, and skipped no-positive."""
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query_id": "Q1",
            "query_text": "explicit positive query",
            "positive_product_ids": ["p1"],
            "label_source": "manual",
        },
        {
            "query_id": "Q2",
            "query_text": "weak metadata query",
            "expected_article_type": "Shirts",
            "expected_base_colour": "Black",
            "expected_gender": "Men",
            "expected_sub_category": "Topwear",
            "label_source": "weak_metadata",
        },
        {
            "query_id": "Q3",
            "query_text": "no positive query",
            "expected_article_type": "Bags",
            "expected_base_colour": "Green",
            "expected_gender": "Men",
            "expected_sub_category": "Bags",
            "label_source": "weak_metadata",
        },
    ]
    with eval_queries_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row))
            file.write("\n")
    return eval_queries_path


def _write_products(tmp_path: Path) -> Path:
    """Write products used by relevance lookup."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    write_products(
        [
            _product("p1", article_type="Shoes", base_colour="Red", gender="Women", sub_category="Footwear"),
            _product("p2", article_type="Shirts", base_colour="Black", gender="Men", sub_category="Topwear"),
            _product("p3", article_type="Shirts", base_colour="Black", gender="Men", sub_category="Topwear"),
            _product("p4", article_type="Shirts", base_colour="Black", gender="Women", sub_category="Topwear"),
            _product("p5", article_type="Shirts", base_colour="Blue", gender="Men", sub_category="Topwear"),
            _product("p6", article_type="Shoes", base_colour="Red", gender="Women", sub_category="Footwear"),
            _product("p7", article_type="Watches", base_colour="Silver", gender="Women", sub_category="Watches"),
            _product("p8", article_type="Shirts", base_colour="Black", gender="Men", sub_category="Topwear"),
        ],
        products_path,
    )
    return products_path


def _read_jsonl(path: Path) -> list[dict]:
    """Read all JSONL rows from a file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _response(query_text: str, results: list[RetrievalResult]) -> RetrievalResponse:
    return RetrievalResponse(
        query_type="text",
        query=query_text,
        top_k=len(results),
        results=results,
        missing_product_ids=[],
    )


def _result(
    product_id: str,
    *,
    score: float = 0.5,
    rank: int = 1,
    title: str = "Product",
    article_type: str | None = "Shirts",
    base_colour: str | None = "Black",
    gender: str | None = "Men",
    usage: str | None = "Casual",
    sub_category: str | None = "Topwear",
    freshness_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        product_id=product_id,
        title=title,
        image_path=f"data/raw/images/{product_id}.jpg",
        article_type=article_type,
        base_colour=base_colour,
        gender=gender,
        usage=usage,
        sub_category=sub_category,
        freshness_score=freshness_score,
        score=score,
        rank=rank,
        embedding_index=rank - 1,
        recall_rank=rank,
        rerank_score=score,
        final_rank=rank,
    )


def _product(
    product_id: str,
    *,
    article_type: str,
    base_colour: str,
    gender: str,
    sub_category: str,
) -> ProductItem:
    return ProductItem(
        product_id=product_id,
        title=f"Product {product_id}",
        gender=gender,
        master_category="Apparel",
        sub_category=sub_category,
        article_type=article_type,
        base_colour=base_colour,
        season="Fall",
        year=2011,
        usage="Casual",
        image_path=f"data/raw/images/{product_id}.jpg",
        freshness_score=0.5,
    )
