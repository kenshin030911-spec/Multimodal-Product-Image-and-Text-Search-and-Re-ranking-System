"""Offline evaluation tests."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import write_products
from backend.app.evaluation import eval_runner
from backend.app.evaluation.eval_runner import (
    evaluate_case,
    generate_weak_metadata_queries,
    relevance_grade,
    run_evaluation,
)
from backend.app.evaluation.metrics import (
    aggregate_metrics,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.app.evaluation.query_generation import normalize_query_text
from backend.app.evaluation.report_writer import (
    DETAILS_FILE,
    SUMMARY_FILE,
    TEXT_SUMMARY_FILE,
    write_evaluation_reports,
)
from backend.app.main import create_app
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.schemas.evaluation import EvalQuery
from backend.app.schemas.product import ProductItem
from backend.app.training.feature_exporter import FEATURE_NAMES
from backend.app.training.model_store import save_model_artifacts, save_trained_reranker
from backend.app.training.pairwise_ranker import PairwiseLogisticRanker
from backend.app.training.pairwise_trainer import MODEL_TYPE as PAIRWISE_MODEL_TYPE


class FakeTrainedModel:
    """Small serializable model that scores candidates by trained feature rows."""

    def predict_proba(self, x_rows):
        gender_index = list(FEATURE_NAMES).index("gender_match")
        color_index = list(FEATURE_NAMES).index("color_match")
        probabilities = []
        for row in x_rows:
            score = min(1.0, max(0.0, float(row[gender_index]) * 0.9 + float(row[color_index]) * 0.05))
            probabilities.append([1.0 - score, score])
        return probabilities


def test_evaluation_metrics_handle_core_cases() -> None:
    """Precision/Recall/Hit/MRR/NDCG use relevance grades safely."""
    grades = [0, 2, 1]
    ideal = [2, 1]

    assert precision_at_k(grades, 2) == pytest.approx(0.5)
    assert recall_at_k(grades, positive_count=2, k=2) == pytest.approx(0.5)
    assert hit_at_k(grades, 2) == 1.0
    assert mrr(grades, 3) == pytest.approx(0.5)

    expected_dcg = 3.0 / math.log2(3) + 1.0 / math.log2(4)
    expected_idcg = 3.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(grades, 3, ideal_relevance_grades=ideal) == pytest.approx(
        expected_dcg / expected_idcg
    )
    assert precision_at_k([], 3) == 0.0
    assert recall_at_k([1], positive_count=0, k=1) == 0.0
    with pytest.raises(ValueError):
        precision_at_k([1], 0)


def test_aggregate_metrics_averages_rows() -> None:
    """Metric aggregation returns per-key means."""
    aggregated = aggregate_metrics(
        [
            {"precision_at_k": 1.0, "mrr": 1.0},
            {"precision_at_k": 0.0, "mrr": 0.5},
        ]
    )

    assert aggregated["precision_at_k"] == pytest.approx(0.5)
    assert aggregated["mrr"] == pytest.approx(0.75)


def test_positive_product_ids_override_weak_metadata() -> None:
    """Explicit positive IDs take precedence over metadata matches."""
    case = EvalQuery(
        query_id="Q1",
        query_text="black shirt",
        expected_article_type="Shirts",
        expected_base_colour="Black",
        expected_gender="Men",
        expected_sub_category="Topwear",
        positive_product_ids=["p1"],
        label_source="manual",
    )

    assert relevance_grade(case, _result("p1", article_type="Shoes", base_colour="Red")) == 1
    assert relevance_grade(case, _result("p2", article_type="Shirts", base_colour="Black")) == 0


def test_weak_metadata_query_generation_is_deterministic() -> None:
    """Missing eval_queries.jsonl can be replaced by seeded weak metadata cases."""
    products = [
        _product("p1", article_type="Shirts", base_colour="Black", gender="Men"),
        _product("p2", article_type="Dresses", base_colour="Red", gender="Women"),
    ]

    first = generate_weak_metadata_queries(products, max_queries=2, seed=42)
    second = generate_weak_metadata_queries(products, max_queries=2, seed=42)

    assert [case.query_id for case in first] == ["Q000001", "Q000002"]
    assert [case.query_text for case in first] == [case.query_text for case in second]
    assert all(case.label_source == "weak_metadata" for case in first)
    assert all(case.query_template_name == "gender_color_article_type" for case in first)
    assert all(case.query_generation_mode == "basic" for case in first)


def test_basic_query_generation_keeps_legacy_query_text() -> None:
    """Basic mode keeps the original gender-color-article query format."""
    products = [_product("p1", article_type="Shirts", base_colour="Black", gender="Men")]

    cases = generate_weak_metadata_queries(
        products,
        max_queries=1,
        seed=42,
        query_templates="basic",
        queries_per_product=2,
    )

    assert [case.query_text for case in cases] == ["men black shirts"]


def test_augmented_query_generation_varies_templates_and_deduplicates() -> None:
    """Augmented mode emits multiple normalized template variants with unique texts."""
    products = [
        _product("p1", article_type="Shirts", base_colour="Navy Blue", gender="Men"),
        _product("p2", article_type="Shirts", base_colour="Navy Blue", gender="Men"),
    ]

    first = generate_weak_metadata_queries(
        products,
        max_queries=12,
        seed=7,
        query_templates="augmented",
        queries_per_product=12,
        max_query_variants=12,
    )
    second = generate_weak_metadata_queries(
        products,
        max_queries=12,
        seed=7,
        query_templates="augmented",
        queries_per_product=12,
        max_query_variants=12,
    )

    query_texts = [case.query_text for case in first]
    assert len(first) > 1
    assert [case.query_text for case in first] == [case.query_text for case in second]
    assert len(query_texts) == len(set(query_texts))
    assert [case.query_id for case in first] == [
        f"Q{index:06d}" for index in range(1, len(first) + 1)
    ]
    assert {case.query_template_name for case in first} - {"gender_color_article_type"}
    assert all(case.query_text == case.query_text.lower() for case in first)
    assert all(case.query_generation_mode == "augmented" for case in first)


def test_augmented_query_generation_skips_missing_template_fields() -> None:
    """Templates requiring missing usage or season fields are skipped safely."""
    product = _product("p1", article_type="Shirts", base_colour="Black", gender="Men")
    product = product.model_copy(update={"usage": None, "season": None})

    cases = generate_weak_metadata_queries(
        [product],
        max_queries=12,
        seed=42,
        query_templates="augmented",
        queries_per_product=12,
        max_query_variants=12,
    )

    template_names = {case.query_template_name for case in cases}
    assert "usage_article_type" not in template_names
    assert "season_article_type" not in template_names
    assert "season_color_article_type" not in template_names


def test_augmented_query_generation_respects_max_queries_and_normalizes() -> None:
    """Generated query count and normalization helpers are deterministic."""
    products = [
        _product("p1", article_type="Shirts", base_colour="Black", gender="Men"),
        _product("p2", article_type="Dresses", base_colour="Red", gender="Women"),
    ]

    cases = generate_weak_metadata_queries(
        products,
        max_queries=2,
        seed=42,
        query_templates="augmented",
        queries_per_product=3,
        max_query_variants=12,
    )

    assert len(cases) == 2
    assert normalize_query_text("  Men   Black Black   Shirts  ") == "men black shirts"


def test_eval_query_supports_old_and_new_generation_fields() -> None:
    """EvalQuery remains backward compatible while accepting template metadata."""
    old_case = EvalQuery.model_validate({"query_id": "Q1", "query_text": "black shirt"})
    new_case = EvalQuery.model_validate(
        {
            "query_id": "Q2",
            "query_text": "black shirts for men",
            "query_template_name": "color_article_type_for_gender",
            "query_template_fields": ["base_colour", "article_type", "gender"],
            "query_generation_mode": "augmented",
        }
    )

    assert old_case.query_template_name is None
    assert old_case.query_template_fields == []
    assert new_case.query_template_name == "color_article_type_for_gender"
    assert new_case.query_generation_mode == "augmented"


def test_report_writer_writes_summary_details_and_text(tmp_path: Path) -> None:
    """Report writer emits JSON, JSONL, and readable text files."""
    summary = {
        "prepared": True,
        "query_count": 1,
        "metric_k": 10,
        "candidate_k": 20,
        "max_queries": 1,
        "eval_source": "test",
        "label_policy": "weak-supervised test policy",
        "query_generation_mode": "basic",
        "query_templates": "basic",
        "queries_per_product": 2,
        "max_query_variants": 12,
        "query_template_names": ["gender_color_article_type"],
        "before_rerank": {"precision_at_k": 0.0},
        "after_rerank": {"precision_at_k": 1.0},
        "delta": {"precision_at_k": 1.0},
        "metrics": {
            "precision_at_10_before_rerank": 0.0,
            "precision_at_10_after_rerank": 1.0,
        },
    }
    details = [{"query_id": "Q1", "before_top_k": [], "after_top_k": []}]

    write_evaluation_reports(summary, details, output_dir=tmp_path, project_root=tmp_path)

    saved_summary = json.loads((tmp_path / SUMMARY_FILE).read_text(encoding="utf-8"))
    detail_lines = (tmp_path / DETAILS_FILE).read_text(encoding="utf-8").splitlines()
    text_summary = (tmp_path / TEXT_SUMMARY_FILE).read_text(encoding="utf-8")

    assert saved_summary["prepared"] is True
    assert saved_summary["summary_path"] == SUMMARY_FILE
    assert json.loads(detail_lines[0])["query_id"] == "Q1"
    assert "weak-supervised" in text_summary
    assert "Query Generation:" in text_summary
    assert "mode: basic" in text_summary


def test_report_writer_formats_three_way_summary(tmp_path: Path) -> None:
    """Text report shows trained reranker deltas when present."""
    write_evaluation_reports(
        summary={
            "prepared": True,
            "query_count": 1,
            "metric_k": 1,
            "candidate_k": 3,
            "max_queries": 1,
            "eval_source": "test",
            "label_policy": "weak-supervised test policy",
            "include_trained_reranker": True,
            "vector_recall": {"precision_at_k": 0.0},
            "rule_rerank": {"precision_at_k": 1.0},
            "trained_rerank": {"precision_at_k": 1.0},
            "delta_rule_vs_vector": {"precision_at_k": 1.0},
            "delta_trained_vs_vector": {"precision_at_k": 1.0},
            "delta_trained_vs_rule": {"precision_at_k": 0.0},
            "before_rerank": {"precision_at_k": 0.0},
            "after_rerank": {"precision_at_k": 1.0},
            "delta": {"precision_at_k": 1.0},
            "metrics": {
                "precision_at_1_vector_recall": 0.0,
                "precision_at_1_rule_rerank": 1.0,
                "precision_at_1_trained_rerank": 1.0,
            },
        },
        details=[],
        output_dir=tmp_path,
        project_root=tmp_path,
    )

    text_summary = (tmp_path / TEXT_SUMMARY_FILE).read_text(encoding="utf-8")

    assert "include_trained_reranker: True" in text_summary
    assert "trained=" in text_summary
    assert "trained_vs_rule" in text_summary


def test_report_writer_formats_four_way_summary(tmp_path: Path) -> None:
    """Text report shows pairwise metrics and score definitions when present."""
    write_evaluation_reports(
        summary={
            "prepared": True,
            "query_count": 1,
            "metric_k": 10,
            "candidate_k": 20,
            "max_queries": 1,
            "eval_source": "test",
            "label_policy": "weak-supervised test policy",
            "include_trained_reranker": True,
            "include_pairwise_reranker": True,
            "vector_recall": {"ndcg_at_k": 0.1, "mrr": 0.2},
            "rule_rerank": {"ndcg_at_k": 0.3, "mrr": 0.4},
            "binary_trained_rerank": {"ndcg_at_k": 0.5, "mrr": 0.6},
            "trained_rerank": {"ndcg_at_k": 0.5, "mrr": 0.6},
            "pairwise_rerank": {"ndcg_at_k": 0.7, "mrr": 0.8},
            "delta_rule_vs_vector": {"ndcg_at_k": 0.2, "mrr": 0.2},
            "delta_binary_trained_vs_vector": {"ndcg_at_k": 0.4, "mrr": 0.4},
            "delta_binary_trained_vs_rule": {"ndcg_at_k": 0.2, "mrr": 0.2},
            "delta_pairwise_vs_vector": {"ndcg_at_k": 0.6, "mrr": 0.6},
            "delta_pairwise_vs_rule": {"ndcg_at_k": 0.4, "mrr": 0.4},
            "delta_pairwise_vs_binary_trained": {"ndcg_at_k": 0.2, "mrr": 0.2},
            "before_rerank": {"ndcg_at_k": 0.1, "mrr": 0.2},
            "after_rerank": {"ndcg_at_k": 0.3, "mrr": 0.4},
            "delta": {"ndcg_at_k": 0.2, "mrr": 0.2},
            "trained_reranker_model": {
                "model_type": "fake_binary",
                "framework": "test",
                "trained_rerank_score_definition": "predict_proba(features)[1]",
            },
            "pairwise_reranker_model": {
                "model_type": "pairwise_logistic_ranker",
                "framework": "sklearn",
                "score_definition": "coef dot standardized features",
                "score_note": "pairwise_rerank_score is an ordering score, not calibrated probability",
            },
            "metrics": {},
        },
        details=[],
        output_dir=tmp_path,
        project_root=tmp_path,
    )

    text_summary = (tmp_path / TEXT_SUMMARY_FILE).read_text(encoding="utf-8")

    assert "binary_trained=" in text_summary
    assert "pairwise=" in text_summary
    assert "Score Definitions:" in text_summary
    assert "coef dot standardized features" in text_summary
    assert "not calibrated probabilities" in text_summary


def test_routes_eval_without_report_returns_placeholder(tmp_path: Path, monkeypatch) -> None:
    """Summary API does not run evaluation when report is absent."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    get_settings.cache_clear()
    client = TestClient(create_app())

    try:
        response = client.get("/api/evaluation/summary")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is False
    assert data["placeholder"] is True
    assert "run_evaluation.py" in data["message"]


def test_routes_eval_reads_existing_report(tmp_path: Path, monkeypatch) -> None:
    """Summary API reads evaluation_summary.json without loading models."""
    output_dir = tmp_path / "outputs" / "eval_reports"
    write_evaluation_reports(
        summary={
            "prepared": True,
            "query_count": 1,
            "metric_k": 10,
            "candidate_k": 20,
            "max_queries": 1,
            "eval_source": "test",
            "label_policy": "policy",
            "include_trained_reranker": True,
            "vector_recall": {"precision_at_k": 0.0},
            "rule_rerank": {"precision_at_k": 1.0},
            "trained_rerank": {"precision_at_k": 0.5},
            "delta_rule_vs_vector": {"precision_at_k": 1.0},
            "delta_trained_vs_vector": {"precision_at_k": 0.5},
            "delta_trained_vs_rule": {"precision_at_k": -0.5},
            "before_rerank": {"precision_at_k": 0.0},
            "after_rerank": {"precision_at_k": 1.0},
            "delta": {"precision_at_k": 1.0},
            "metrics": {
                "precision_at_10_before_rerank": 0.0,
                "precision_at_10_after_rerank": 1.0,
                "ndcg_at_10_after_rerank": 1.0,
            },
            "message": "done",
        },
        details=[],
        output_dir=output_dir,
        project_root=tmp_path,
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    get_settings.cache_clear()
    client = TestClient(create_app())

    try:
        response = client.get("/api/evaluation/summary")
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    data = response.json()
    assert data["prepared"] is True
    assert data["placeholder"] is False
    assert data["metrics"]["precision_at_10_after_rerank"] == 1.0
    assert data["before_rerank"]["precision_at_k"] == 0.0
    assert data["include_trained_reranker"] is True
    assert data["trained_rerank"]["precision_at_k"] == 0.5
    assert data["delta_trained_vs_rule"]["precision_at_k"] == -0.5


def test_runner_computes_before_after_and_rerank_changes_order(tmp_path: Path, monkeypatch) -> None:
    """Runner evaluates one mocked search response before and after reranking."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    products = [
        _product(
            "p1",
            title="Red Shoes",
            article_type="Shoes",
            base_colour="Red",
            gender="Women",
            sub_category="Footwear",
        ),
        _product(
            "p2",
            title="Black Casual Shirts",
            article_type="Shirts",
            base_colour="Black",
            gender="Men",
            sub_category="Topwear",
        ),
    ]
    write_products(products, products_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps(
            {
                "query_id": "Q1",
                "query_text": "black casual shirts men topwear",
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

    def fake_search_text_to_image(**kwargs) -> RetrievalResponse:
        _ = kwargs
        return _response(
            [
                _result(
                    "p1",
                    score=0.70,
                    rank=1,
                    title="Red Shoes",
                    article_type="Shoes",
                    base_colour="Red",
                    gender="Women",
                    sub_category="Footwear",
                ),
                _result(
                    "p2",
                    score=0.64,
                    rank=2,
                    title="Black Casual Shirts",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Men",
                    sub_category="Topwear",
                ),
            ]
        )

    monkeypatch.setattr(eval_runner, "search_text_to_image", fake_search_text_to_image)

    result = run_evaluation(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=2,
        max_queries=1,
        seed=42,
        device="cpu",
        project_root=tmp_path,
    )

    detail = result.details[0]
    assert detail["before_top_k"][0]["product_id"] == "p1"
    assert detail["after_top_k"][0]["product_id"] == "p2"
    assert detail["vector_top_k"][0]["product_id"] == "p1"
    assert detail["rule_top_k"][0]["product_id"] == "p2"
    assert detail["before_metrics"]["hit_at_k"] == 0.0
    assert detail["after_metrics"]["hit_at_k"] == 1.0
    assert result.summary["include_trained_reranker"] is False
    assert result.summary["metrics"]["hit_at_1_after_rerank"] == 1.0
    assert result.summary["query_generation_mode"] == "eval_queries_jsonl"
    assert result.summary["query_template_names"] == []
    assert (tmp_path / "outputs" / "eval_reports" / SUMMARY_FILE).is_file()


def test_runner_records_augmented_query_generation_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Generated augmented queries are recorded in summary and detail rows."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    products = [
        _product(
            "p2",
            title="Black Casual Shirts",
            article_type="Shirts",
            base_colour="Black",
            gender="Men",
            sub_category="Topwear",
        )
    ]
    write_products(products, products_path)

    def fake_search_text_to_image(**kwargs) -> RetrievalResponse:
        _ = kwargs
        return _response(
            [
                _result(
                    "p2",
                    score=0.80,
                    rank=1,
                    title="Black Casual Shirts",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Men",
                    sub_category="Topwear",
                )
            ]
        )

    monkeypatch.setattr(eval_runner, "search_text_to_image", fake_search_text_to_image)

    result = run_evaluation(
        eval_queries_path=tmp_path / "data" / "processed" / "missing_eval_queries.jsonl",
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=1,
        max_queries=2,
        seed=42,
        device="cpu",
        query_templates="augmented",
        queries_per_product=2,
        max_query_variants=12,
        project_root=tmp_path,
    )

    assert result.summary["query_generation_mode"] == "augmented"
    assert result.summary["query_templates"] == "augmented"
    assert result.summary["queries_per_product"] == 2
    assert "article_type_only" in result.summary["query_template_names"]
    assert "not real user search logs" in result.summary["query_generation_note"]
    assert result.details[0]["query_generation_mode"] == "augmented"
    assert result.details[0]["query_template_name"]


def test_runner_three_way_evaluation_uses_same_candidate_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Vector, rule, and trained ranking are all evaluated from one search call."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    products = [
        _product(
            "p1",
            title="Red Shoes",
            article_type="Shoes",
            base_colour="Red",
            gender="Women",
            sub_category="Footwear",
        ),
        _product(
            "p2",
            title="Black Shirts",
            article_type="Shirts",
            base_colour="Black",
            gender="Women",
            sub_category="Topwear",
        ),
        _product(
            "p3",
            title="Black Casual Shirts Men",
            article_type="Shirts",
            base_colour="Black",
            gender="Men",
            sub_category="Topwear",
        ),
    ]
    write_products(products, products_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps(
            {
                "query_id": "Q1",
                "query_text": "black casual shirts men topwear",
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
    model_path, meta_path = _write_fake_trained_model(tmp_path)
    search_call_count = 0

    def fake_search_text_to_image(**kwargs) -> RetrievalResponse:
        nonlocal search_call_count
        search_call_count += 1
        _ = kwargs
        return _response(
            [
                _result(
                    "p1",
                    score=0.70,
                    rank=1,
                    title="Red Shoes",
                    article_type="Shoes",
                    base_colour="Red",
                    gender="Women",
                    sub_category="Footwear",
                ),
                _result(
                    "p2",
                    score=0.68,
                    rank=2,
                    title="Black Shirts",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Women",
                    sub_category="Topwear",
                ),
                _result(
                    "p3",
                    score=0.66,
                    rank=3,
                    title="Black Casual Shirts Men",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Men",
                    sub_category="Topwear",
                ),
            ]
        )

    monkeypatch.setattr(eval_runner, "search_text_to_image", fake_search_text_to_image)

    result = run_evaluation(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=3,
        max_queries=1,
        seed=42,
        device="cpu",
        include_trained_reranker=True,
        trained_model_path=model_path,
        trained_meta_path=meta_path,
        project_root=tmp_path,
    )

    detail = result.details[0]
    summary = result.summary

    assert search_call_count == 1
    assert detail["vector_top_k"][0]["product_id"] == "p1"
    assert detail["rule_top_k"][0]["product_id"] == "p3"
    assert detail["trained_top_k"][0]["product_id"] == "p3"
    assert detail["trained_top_k"][0]["score_name"] == "trained_rerank_score"
    assert detail["trained_top_k"][0]["recall_rank"] == 3
    assert detail["trained_top_k"][0]["recall_score"] == pytest.approx(0.66)
    assert detail["before_top_k"] == detail["vector_top_k"]
    assert detail["after_top_k"] == detail["rule_top_k"]
    assert "trained_metrics" in detail
    assert summary["include_trained_reranker"] is True
    assert summary["vector_recall"]["hit_at_k"] == 0.0
    assert summary["rule_rerank"]["hit_at_k"] == 1.0
    assert summary["trained_rerank"]["hit_at_k"] == 1.0
    assert summary["delta_trained_vs_vector"]["hit_at_k"] == 1.0
    assert summary["delta_trained_vs_rule"]["hit_at_k"] == 0.0
    assert summary["before_rerank"] == summary["vector_recall"]
    assert summary["after_rerank"] == summary["rule_rerank"]
    assert "precision_at_1_trained_rerank" in summary["metrics"]

    detail_rows = (tmp_path / "outputs" / "eval_reports" / DETAILS_FILE).read_text(
        encoding="utf-8",
    ).splitlines()
    saved_detail = json.loads(detail_rows[0])
    assert "trained_top_k" in saved_detail
    assert saved_detail["trained_top_k"][0]["score_name"] == "trained_rerank_score"


def test_runner_four_way_evaluation_uses_same_candidate_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Vector, rule, binary trained, and pairwise rankings use one candidate pool."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    products = [
        _product(
            "p1",
            title="Red Shoes",
            article_type="Shoes",
            base_colour="Red",
            gender="Women",
            sub_category="Footwear",
        ),
        _product(
            "p2",
            title="Black Shirts",
            article_type="Shirts",
            base_colour="Black",
            gender="Women",
            sub_category="Topwear",
        ),
        _product(
            "p3",
            title="Black Casual Shirts Men",
            article_type="Shirts",
            base_colour="Black",
            gender="Men",
            sub_category="Topwear",
        ),
    ]
    write_products(products, products_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps(
            {
                "query_id": "Q1",
                "query_text": "black casual shirts men topwear",
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
    binary_model_path, binary_meta_path = _write_fake_trained_model(tmp_path)
    pairwise_model_path, pairwise_meta_path = _write_fake_pairwise_model(tmp_path)
    search_call_count = 0

    def fake_search_text_to_image(**kwargs) -> RetrievalResponse:
        nonlocal search_call_count
        search_call_count += 1
        _ = kwargs
        return _response(
            [
                _result(
                    "p1",
                    score=0.70,
                    rank=1,
                    title="Red Shoes",
                    article_type="Shoes",
                    base_colour="Red",
                    gender="Women",
                    sub_category="Footwear",
                ),
                _result(
                    "p2",
                    score=0.68,
                    rank=2,
                    title="Black Shirts",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Women",
                    sub_category="Topwear",
                ),
                _result(
                    "p3",
                    score=0.66,
                    rank=3,
                    title="Black Casual Shirts Men",
                    article_type="Shirts",
                    base_colour="Black",
                    gender="Men",
                    sub_category="Topwear",
                ),
            ]
        )

    monkeypatch.setattr(eval_runner, "search_text_to_image", fake_search_text_to_image)

    result = run_evaluation(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=3,
        max_queries=1,
        seed=42,
        device="cpu",
        include_trained_reranker=True,
        trained_model_path=binary_model_path,
        trained_meta_path=binary_meta_path,
        include_pairwise_reranker=True,
        pairwise_model_path=pairwise_model_path,
        pairwise_meta_path=pairwise_meta_path,
        project_root=tmp_path,
    )

    detail = result.details[0]
    summary = result.summary

    assert search_call_count == 1
    assert detail["vector_top_k"][0]["product_id"] == "p1"
    assert detail["rule_top_k"][0]["product_id"] == "p3"
    assert detail["binary_trained_top_k"][0]["product_id"] == "p3"
    assert detail["trained_top_k"][0]["product_id"] == "p3"
    assert detail["pairwise_top_k"][0]["product_id"] == "p2"
    assert detail["pairwise_top_k"][0]["score_name"] == "pairwise_rerank_score"
    assert detail["pairwise_top_k"][0]["recall_rank"] == 2
    assert detail["pairwise_top_k"][0]["recall_score"] == pytest.approx(0.68)
    assert "pairwise_metrics" in detail
    assert "binary_trained_metrics" in detail
    assert summary["include_pairwise_reranker"] is True
    assert summary["binary_trained_rerank"] == summary["trained_rerank"]
    assert "pairwise_rerank" in summary
    assert "delta_pairwise_vs_vector" in summary
    assert "delta_pairwise_vs_rule" in summary
    assert "delta_pairwise_vs_binary_trained" in summary
    assert "precision_at_1_pairwise_rerank" in summary["metrics"]
    assert "precision_at_1_binary_trained_rerank" in summary["metrics"]

    detail_rows = (tmp_path / "outputs" / "eval_reports" / DETAILS_FILE).read_text(
        encoding="utf-8",
    ).splitlines()
    saved_detail = json.loads(detail_rows[0])
    assert "pairwise_top_k" in saved_detail
    assert saved_detail["pairwise_top_k"][0]["score_name"] == "pairwise_rerank_score"


def test_runner_does_not_require_trained_model_when_not_requested(tmp_path: Path, monkeypatch) -> None:
    """Missing trained artifacts are ignored unless include_trained_reranker is true."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    write_products([_product("p1"), _product("p2")], products_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps({"query_id": "Q1", "query_text": "black shirt"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        eval_runner,
        "search_text_to_image",
        lambda **kwargs: _response([_result("p1", rank=1), _result("p2", rank=2)]),
    )

    result = run_evaluation(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=2,
        max_queries=1,
        include_trained_reranker=False,
        trained_model_path=tmp_path / "missing.joblib",
        trained_meta_path=tmp_path / "missing.json",
        project_root=tmp_path,
    )

    assert result.summary["include_trained_reranker"] is False
    assert "trained_rerank" not in result.summary
    assert result.summary["include_pairwise_reranker"] is False
    assert "pairwise_rerank" not in result.summary


def test_runner_fails_when_requested_trained_model_is_missing(tmp_path: Path) -> None:
    """Explicit trained evaluation fails clearly when artifacts are absent."""
    with pytest.raises(FileNotFoundError, match="trained reranker evaluation requested"):
        run_evaluation(
            include_trained_reranker=True,
            trained_model_path=tmp_path / "missing.joblib",
            trained_meta_path=tmp_path / "missing.json",
            project_root=tmp_path,
        )


def test_runner_fails_on_trained_feature_name_mismatch(tmp_path: Path) -> None:
    """Trained reranker feature_names must match FEATURE_NAMES."""
    model_path = tmp_path / "models" / "reranker" / "trained_reranker.joblib"
    meta_path = tmp_path / "models" / "reranker" / "trained_reranker_meta.json"
    save_trained_reranker(
        model=FakeTrainedModel(),
        model_path=model_path,
        meta={"feature_names": ["wrong_feature"]},
        meta_path=meta_path,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="feature_names mismatch"):
        run_evaluation(
            include_trained_reranker=True,
            trained_model_path=model_path,
            trained_meta_path=meta_path,
            project_root=tmp_path,
        )


def test_runner_does_not_require_pairwise_model_when_not_requested(tmp_path: Path, monkeypatch) -> None:
    """Missing pairwise artifacts are ignored unless include_pairwise_reranker is true."""
    products_path = tmp_path / "data" / "processed" / "products.jsonl"
    write_products([_product("p1"), _product("p2")], products_path)
    eval_queries_path = tmp_path / "data" / "processed" / "eval_queries.jsonl"
    eval_queries_path.write_text(
        json.dumps({"query_id": "Q1", "query_text": "black shirt"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        eval_runner,
        "search_text_to_image",
        lambda **kwargs: _response([_result("p1", rank=1), _result("p2", rank=2)]),
    )

    result = run_evaluation(
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_dir=tmp_path / "outputs" / "eval_reports",
        metric_k=1,
        candidate_k=2,
        max_queries=1,
        include_pairwise_reranker=False,
        pairwise_model_path=tmp_path / "missing.joblib",
        pairwise_meta_path=tmp_path / "missing.json",
        project_root=tmp_path,
    )

    assert result.summary["include_pairwise_reranker"] is False
    assert "pairwise_rerank" not in result.summary


def test_runner_fails_when_requested_pairwise_model_is_missing(tmp_path: Path) -> None:
    """Explicit pairwise evaluation fails clearly when artifacts are absent."""
    with pytest.raises(FileNotFoundError, match="pairwise reranker evaluation requested"):
        run_evaluation(
            include_pairwise_reranker=True,
            pairwise_model_path=tmp_path / "missing.joblib",
            pairwise_meta_path=tmp_path / "missing.json",
            project_root=tmp_path,
        )


def test_runner_fails_on_pairwise_feature_name_mismatch(tmp_path: Path) -> None:
    """Pairwise reranker feature_names must match FEATURE_NAMES."""
    model_path = tmp_path / "models" / "reranker_pairwise" / "pairwise_reranker.joblib"
    meta_path = tmp_path / "models" / "reranker_pairwise" / "pairwise_reranker_meta.json"
    ranker = _fake_pairwise_ranker()
    save_model_artifacts(
        model=ranker,
        model_path=model_path,
        meta={
            "model_type": PAIRWISE_MODEL_TYPE,
            "framework": "test",
            "feature_names": ["wrong_feature"],
        },
        meta_path=meta_path,
        overwrite=True,
    )

    with pytest.raises(ValueError, match="feature_names mismatch"):
        run_evaluation(
            include_pairwise_reranker=True,
            pairwise_model_path=model_path,
            pairwise_meta_path=meta_path,
            project_root=tmp_path,
        )


def test_evaluate_case_uses_positive_count_for_recall() -> None:
    """Single-case evaluator reports positive count and metric dictionaries."""
    case = EvalQuery(query_id="Q1", query_text="black shirt", positive_product_ids=["p2"])
    products = [_product("p1"), _product("p2")]
    product_lookup = {product.product_id: product for product in products}
    before = _response([_result("p1", rank=1), _result("p2", rank=2)])
    after = _response([_result("p2", rank=1), _result("p1", rank=2)])

    detail = evaluate_case(case, before, after, products, product_lookup, metric_k=1)

    assert detail["positive_count"] == 1
    assert detail["before_metrics"]["recall_at_k"] == 0.0
    assert detail["after_metrics"]["recall_at_k"] == 1.0


def test_run_evaluation_help() -> None:
    """run_evaluation.py --help is available and does not load FashionCLIP."""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/run_evaluation.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--eval-queries-path" in completed.stdout
    assert "--candidate-k" in completed.stdout
    assert "--include-trained-reranker" in completed.stdout
    assert "--trained-model-path" in completed.stdout
    assert "--query-templates" in completed.stdout
    assert "--queries-per-product" in completed.stdout
    assert "--max-query-variants" in completed.stdout
    assert "--include-pairwise-reranker" in completed.stdout
    assert "--pairwise-model-path" in completed.stdout
    assert "--pairwise-meta-path" in completed.stdout


def _response(results: list[RetrievalResult]) -> RetrievalResponse:
    return RetrievalResponse(
        query_type="text",
        query="black shirt",
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
    title: str | None = None,
    article_type: str = "Shirts",
    base_colour: str = "Black",
    gender: str = "Men",
    sub_category: str = "Topwear",
) -> ProductItem:
    return ProductItem(
        product_id=product_id,
        title=title or f"Product {product_id}",
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


def _write_fake_trained_model(tmp_path: Path) -> tuple[Path, Path]:
    """Persist fake trained model artifacts for offline evaluation tests."""
    model_path = tmp_path / "models" / "reranker" / "trained_reranker.joblib"
    meta_path = tmp_path / "models" / "reranker" / "trained_reranker_meta.json"
    save_trained_reranker(
        model=FakeTrainedModel(),
        model_path=model_path,
        meta={
            "model_type": "fake",
            "framework": "test",
            "feature_names": list(FEATURE_NAMES),
            "trained_rerank_score_definition": "fake predict_proba(features)[1]",
        },
        meta_path=meta_path,
        overwrite=True,
    )
    return model_path, meta_path


def _write_fake_pairwise_model(tmp_path: Path) -> tuple[Path, Path]:
    """Persist fake pairwise model artifacts for offline evaluation tests."""
    model_path = tmp_path / "models" / "reranker_pairwise" / "pairwise_reranker.joblib"
    meta_path = tmp_path / "models" / "reranker_pairwise" / "pairwise_reranker_meta.json"
    save_model_artifacts(
        model=_fake_pairwise_ranker(),
        model_path=model_path,
        meta={
            "model_type": PAIRWISE_MODEL_TYPE,
            "framework": "test",
            "feature_names": list(FEATURE_NAMES),
            "score_definition": "coef dot standardized features",
            "score_note": (
                "pairwise_rerank_score is an ordering score, "
                "not calibrated probability"
            ),
        },
        meta_path=meta_path,
        overwrite=True,
    )
    return model_path, meta_path


def _fake_pairwise_ranker() -> PairwiseLogisticRanker:
    """Create a deterministic pairwise scorer that prefers p2-like features."""
    ranker = PairwiseLogisticRanker(feature_names=list(FEATURE_NAMES))
    ranker.fit_item_scaler(np.zeros((1, len(FEATURE_NAMES))))
    coefficients = np.zeros(len(FEATURE_NAMES), dtype=float)
    coefficients[list(FEATURE_NAMES).index("article_type_match")] = 1.0
    coefficients[list(FEATURE_NAMES).index("color_match")] = 1.0
    coefficients[list(FEATURE_NAMES).index("sub_category_match")] = 1.0
    coefficients[list(FEATURE_NAMES).index("gender_match")] = -2.0
    ranker.coef_ = coefficients
    return ranker
