"""Offline text retrieval evaluation runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import default_products_path, load_products
from backend.app.data.validators import to_project_relative
from backend.app.evaluation.metrics import (
    aggregate_metrics,
    calculate_ranking_metrics,
    empty_metrics,
)
from backend.app.evaluation.query_generation import (
    DEFAULT_MAX_QUERY_VARIANTS,
    DEFAULT_QUERIES_PER_PRODUCT,
    DEFAULT_QUERY_TEMPLATES,
    QUERY_GENERATION_NOTE,
    QueryTemplateMode,
    generate_weak_metadata_queries as generate_weak_metadata_queries_impl,
    query_template_names_for_mode,
)
from backend.app.evaluation.report_writer import write_evaluation_reports
from backend.app.reranker.feature_builder import build_rerank_feature
from backend.app.reranker.rerank_service import rerank_retrieval_response
from backend.app.retrieval.candidate_builder import RetrievalResponse, RetrievalResult
from backend.app.retrieval.text_search import search_text_to_image
from backend.app.schemas.evaluation import EvaluationSummaryResponse, EvalQuery
from backend.app.schemas.product import ProductItem
from backend.app.training.feature_exporter import FEATURE_NAMES, rerank_feature_to_dict
from backend.app.training.model_store import load_trained_reranker
from backend.app.training.pairwise_trainer import load_pairwise_reranker


DEFAULT_METRIC_K = 10
DEFAULT_CANDIDATE_K = 50
DEFAULT_MAX_QUERIES = 50
DEFAULT_SEED = 42
TRAINED_RERANKER_MODEL_FILE = "trained_reranker.joblib"
TRAINED_RERANKER_META_FILE = "trained_reranker_meta.json"
PAIRWISE_RERANKER_MODEL_FILE = "pairwise_reranker.joblib"
PAIRWISE_RERANKER_META_FILE = "pairwise_reranker_meta.json"
LABEL_POLICY = (
    "Weak-supervised evaluation. positive_product_ids override metadata labels; "
    "explicit positives use relevance grade 1. Without explicit positives, "
    "metadata grades are: 3 = article_type + base_colour + gender match, "
    "2 = article_type + base_colour match, 1 = article_type or sub_category match, "
    "0 = not relevant. Labels are not large-scale manual annotations; metrics are "
    "for relative comparison between vector recall, the rule-based reranker, "
    "the binary trained reranker, and the pairwise reranker when enabled. "
    "Binary trained scores are predict_proba(features)[1]. Pairwise scores are "
    "coef dot standardized features and are not calibrated probabilities. "
    "Classification metrics do not guarantee ranking improvements."
)


@dataclass(frozen=True)
class EvaluationRunResult:
    """In-memory result returned by the offline runner."""

    summary: dict[str, Any]
    details: list[dict[str, Any]]
    report_paths: dict[str, Path]


@dataclass(frozen=True)
class TrainedRerankerBundle:
    """Loaded trained reranker artifacts for offline evaluation only."""

    model: Any
    meta: dict[str, Any]
    feature_names: list[str]
    model_path: Path
    meta_path: Path


@dataclass(frozen=True)
class PairwiseRerankerBundle:
    """Loaded pairwise reranker artifacts for offline evaluation only."""

    model: Any
    meta: dict[str, Any]
    feature_names: list[str]
    model_path: Path
    meta_path: Path


def default_eval_queries_path() -> Path:
    """Return the default eval_queries.jsonl path."""
    return get_settings().processed_data_dir / "eval_queries.jsonl"


def default_trained_model_path() -> Path:
    """Return the default trained reranker model path."""
    return get_settings().reranker_dir / TRAINED_RERANKER_MODEL_FILE


def default_trained_meta_path() -> Path:
    """Return the default trained reranker metadata path."""
    return get_settings().reranker_dir / TRAINED_RERANKER_META_FILE


def default_pairwise_model_path() -> Path:
    """Return the default pairwise reranker model path."""
    return get_settings().model_dir / "reranker_pairwise" / PAIRWISE_RERANKER_MODEL_FILE


def default_pairwise_meta_path() -> Path:
    """Return the default pairwise reranker metadata path."""
    return get_settings().model_dir / "reranker_pairwise" / PAIRWISE_RERANKER_META_FILE


def run_evaluation(
    eval_queries_path: Path | None = None,
    products_path: Path | None = None,
    output_dir: Path | None = None,
    metric_k: int = DEFAULT_METRIC_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    max_queries: int = DEFAULT_MAX_QUERIES,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    query_templates: QueryTemplateMode = DEFAULT_QUERY_TEMPLATES,
    queries_per_product: int = DEFAULT_QUERIES_PER_PRODUCT,
    max_query_variants: int = DEFAULT_MAX_QUERY_VARIANTS,
    include_trained_reranker: bool = False,
    trained_model_path: Path | None = None,
    trained_meta_path: Path | None = None,
    include_pairwise_reranker: bool = False,
    pairwise_model_path: Path | None = None,
    pairwise_meta_path: Path | None = None,
    project_root: Path | None = None,
) -> EvaluationRunResult:
    """Run offline evaluation and write reports."""
    _validate_runner_args(metric_k=metric_k, candidate_k=candidate_k, max_queries=max_queries)
    settings = get_settings()
    project_root = project_root or settings.project_root
    eval_queries_path = eval_queries_path or default_eval_queries_path()
    products_path = products_path or default_products_path(settings)
    output_dir = output_dir or settings.eval_reports_dir
    trained_bundle = (
        load_trained_reranker_for_evaluation(
            model_path=trained_model_path or default_trained_model_path(),
            meta_path=trained_meta_path or default_trained_meta_path(),
        )
        if include_trained_reranker
        else None
    )
    pairwise_bundle = (
        load_pairwise_reranker_for_evaluation(
            model_path=pairwise_model_path or default_pairwise_model_path(),
            meta_path=pairwise_meta_path or default_pairwise_meta_path(),
        )
        if include_pairwise_reranker
        else None
    )

    products = load_products(products_path)
    if not products:
        raise ValueError(f"products.jsonl 为空或不存在: {products_path}")

    eval_cases, eval_source = load_or_generate_eval_cases(
        eval_queries_path=eval_queries_path,
        products=products,
        max_queries=max_queries,
        seed=seed,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    )
    if not eval_cases:
        raise ValueError("没有可用评估 query。")

    product_lookup = {product.product_id: product for product in products}
    details: list[dict[str, Any]] = []
    vector_rows: list[dict[str, float]] = []
    rule_rows: list[dict[str, float]] = []
    binary_trained_rows: list[dict[str, float]] = []
    pairwise_rows: list[dict[str, float]] = []

    for case in eval_cases:
        query_text = _case_query_text(case)
        vector_response = search_text_to_image(
            query_text=query_text,
            top_k=candidate_k,
            products_path=products_path,
            device=device,
            project_root=project_root,
        )
        rule_response = rerank_retrieval_response(vector_response, query_text=query_text)
        trained_response = (
            rerank_with_trained_model(
                response=vector_response,
                query_text=query_text,
                trained_bundle=trained_bundle,
            )
            if trained_bundle is not None
            else None
        )
        pairwise_response = (
            rerank_with_pairwise_model(
                response=vector_response,
                query_text=query_text,
                pairwise_bundle=pairwise_bundle,
            )
            if pairwise_bundle is not None
            else None
        )
        detail = evaluate_case(
            case=case,
            before_response=vector_response,
            after_response=rule_response,
            products=products,
            product_lookup=product_lookup,
            metric_k=metric_k,
            trained_response=trained_response,
            pairwise_response=pairwise_response,
        )
        details.append(detail)
        vector_rows.append(detail["vector_metrics"])
        rule_rows.append(detail["rule_metrics"])
        if trained_response is not None:
            binary_trained_rows.append(detail["binary_trained_metrics"])
        if pairwise_response is not None:
            pairwise_rows.append(detail["pairwise_metrics"])

    vector_metrics = aggregate_metrics(vector_rows)
    rule_metrics = aggregate_metrics(rule_rows)
    binary_trained_metrics = (
        aggregate_metrics(binary_trained_rows) if trained_bundle is not None else None
    )
    pairwise_metrics = aggregate_metrics(pairwise_rows) if pairwise_bundle is not None else None
    delta_rule_vs_vector = _metric_delta(rule_metrics, vector_metrics)
    delta_binary_trained_vs_vector = (
        _metric_delta(binary_trained_metrics, vector_metrics)
        if binary_trained_metrics is not None
        else None
    )
    delta_binary_trained_vs_rule = (
        _metric_delta(binary_trained_metrics, rule_metrics)
        if binary_trained_metrics is not None
        else None
    )
    delta_pairwise_vs_vector = (
        _metric_delta(pairwise_metrics, vector_metrics)
        if pairwise_metrics is not None
        else None
    )
    delta_pairwise_vs_rule = (
        _metric_delta(pairwise_metrics, rule_metrics)
        if pairwise_metrics is not None
        else None
    )
    delta_pairwise_vs_binary_trained = (
        _metric_delta(pairwise_metrics, binary_trained_metrics)
        if pairwise_metrics is not None and binary_trained_metrics is not None
        else None
    )
    flat_metrics = flatten_comparison_metrics(
        before_metrics=vector_metrics,
        after_metrics=rule_metrics,
        metric_k=metric_k,
        binary_trained_metrics=binary_trained_metrics,
        pairwise_metrics=pairwise_metrics,
    )
    generated_queries = eval_source != "eval_queries_jsonl"
    query_generation_mode = query_templates if generated_queries else "eval_queries_jsonl"
    query_template_names = (
        query_template_names_for_mode(query_templates) if generated_queries else []
    )
    summary = {
        "prepared": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_count": len(details),
        "metric_k": metric_k,
        "candidate_k": candidate_k,
        "max_queries": max_queries,
        "seed": seed,
        "eval_source": eval_source,
        "label_policy": LABEL_POLICY,
        "query_generation_mode": query_generation_mode,
        "query_templates": query_templates,
        "queries_per_product": queries_per_product,
        "max_query_variants": max_query_variants,
        "query_template_names": query_template_names,
        "query_generation_note": QUERY_GENERATION_NOTE,
        "include_trained_reranker": include_trained_reranker,
        "include_pairwise_reranker": include_pairwise_reranker,
        "vector_recall": vector_metrics,
        "rule_rerank": rule_metrics,
        "delta_rule_vs_vector": delta_rule_vs_vector,
        "before_rerank": vector_metrics,
        "after_rerank": rule_metrics,
        "delta": delta_rule_vs_vector,
        "metrics": flat_metrics,
        "message": _summary_message(
            include_trained_reranker=include_trained_reranker,
            include_pairwise_reranker=include_pairwise_reranker,
        ),
    }
    if trained_bundle is not None and binary_trained_metrics is not None:
        summary.update(
            {
                "binary_trained_rerank": binary_trained_metrics,
                "trained_rerank": binary_trained_metrics,
                "delta_binary_trained_vs_vector": delta_binary_trained_vs_vector,
                "delta_binary_trained_vs_rule": delta_binary_trained_vs_rule,
                "delta_trained_vs_vector": delta_binary_trained_vs_vector,
                "delta_trained_vs_rule": delta_binary_trained_vs_rule,
                "trained_reranker_model": {
                    "model_path": to_project_relative(
                        trained_bundle.model_path,
                        project_root,
                    ),
                    "meta_path": to_project_relative(
                        trained_bundle.meta_path,
                        project_root,
                    ),
                    "model_type": trained_bundle.meta.get("model_type"),
                    "framework": trained_bundle.meta.get("framework"),
                    "feature_names": trained_bundle.feature_names,
                    "trained_rerank_score_definition": trained_bundle.meta.get(
                        "trained_rerank_score_definition",
                        "predict_proba(features)[1]",
                    ),
                },
            }
        )
    if pairwise_bundle is not None and pairwise_metrics is not None:
        summary.update(
            {
                "pairwise_rerank": pairwise_metrics,
                "delta_pairwise_vs_vector": delta_pairwise_vs_vector,
                "delta_pairwise_vs_rule": delta_pairwise_vs_rule,
                "delta_pairwise_vs_binary_trained": (
                    delta_pairwise_vs_binary_trained or {}
                ),
                "pairwise_reranker_model": {
                    "model_path": to_project_relative(
                        pairwise_bundle.model_path,
                        project_root,
                    ),
                    "meta_path": to_project_relative(
                        pairwise_bundle.meta_path,
                        project_root,
                    ),
                    "model_type": pairwise_bundle.meta.get("model_type"),
                    "framework": pairwise_bundle.meta.get("framework"),
                    "feature_names": pairwise_bundle.feature_names,
                    "score_definition": pairwise_bundle.meta.get(
                        "score_definition",
                        "coef dot standardized features",
                    ),
                    "score_note": pairwise_bundle.meta.get(
                        "score_note",
                        "pairwise_rerank_score is an ordering score, not calibrated probability",
                    ),
                },
            }
        )
    report_paths = write_evaluation_reports(
        summary=summary,
        details=details,
        output_dir=output_dir,
        project_root=project_root,
    )
    written_summary = dict(summary)
    written_summary.update(
        {
            key: _path.as_posix()
            for key, _path in report_paths.items()
        }
    )
    return EvaluationRunResult(summary=written_summary, details=details, report_paths=report_paths)


def load_trained_reranker_for_evaluation(
    model_path: Path,
    meta_path: Path,
) -> TrainedRerankerBundle:
    """Load trained reranker artifacts for an explicit offline evaluation run."""
    try:
        model, meta = load_trained_reranker(
            model_path=model_path,
            meta_path=meta_path,
            expected_feature_names=FEATURE_NAMES,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "trained reranker evaluation requested but model artifacts are missing: "
            f"{exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"trained reranker cannot be used for evaluation: {exc}") from exc

    feature_names = list(meta.get("feature_names") or FEATURE_NAMES)
    return TrainedRerankerBundle(
        model=model,
        meta=meta,
        feature_names=feature_names,
        model_path=model_path,
        meta_path=meta_path,
    )


def load_pairwise_reranker_for_evaluation(
    model_path: Path,
    meta_path: Path,
) -> PairwiseRerankerBundle:
    """Load pairwise reranker artifacts for an explicit offline evaluation run."""
    try:
        model, meta = load_pairwise_reranker(
            model_path=model_path,
            meta_path=meta_path,
            expected_feature_names=FEATURE_NAMES,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "pairwise reranker evaluation requested but model artifacts are missing: "
            f"{exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"pairwise reranker cannot be used for evaluation: {exc}") from exc

    feature_names = list(meta.get("feature_names") or FEATURE_NAMES)
    return PairwiseRerankerBundle(
        model=model,
        meta=meta,
        feature_names=feature_names,
        model_path=model_path,
        meta_path=meta_path,
    )


def load_or_generate_eval_cases(
    eval_queries_path: Path,
    products: Sequence[ProductItem],
    max_queries: int = DEFAULT_MAX_QUERIES,
    seed: int = DEFAULT_SEED,
    query_templates: QueryTemplateMode = DEFAULT_QUERY_TEMPLATES,
    queries_per_product: int = DEFAULT_QUERIES_PER_PRODUCT,
    max_query_variants: int = DEFAULT_MAX_QUERY_VARIANTS,
) -> tuple[list[EvalQuery], str]:
    """Load eval_queries.jsonl, or generate deterministic weak metadata queries."""
    if eval_queries_path.is_file():
        return load_eval_queries(eval_queries_path, max_queries=max_queries), "eval_queries_jsonl"
    return generate_weak_metadata_queries(
        products=products,
        max_queries=max_queries,
        seed=seed,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    ), "generated_weak_metadata"


def load_eval_queries(eval_queries_path: Path, max_queries: int) -> list[EvalQuery]:
    """Read eval query JSONL and skip empty lines."""
    if max_queries <= 0:
        raise ValueError("max_queries 必须大于 0。")

    cases: list[EvalQuery] = []
    with eval_queries_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if len(cases) >= max_queries:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_case = json.loads(stripped)
                case = EvalQuery.model_validate(raw_case)
            except (JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"eval query 第 {line_number} 行无效: {exc}") from exc
            if case.query_type != "text":
                raise ValueError("第一版评估只支持 text query。")
            cases.append(case)
    return cases


def generate_weak_metadata_queries(
    products: Sequence[ProductItem],
    max_queries: int = DEFAULT_MAX_QUERIES,
    seed: int = DEFAULT_SEED,
    query_templates: QueryTemplateMode = DEFAULT_QUERY_TEMPLATES,
    queries_per_product: int = DEFAULT_QUERIES_PER_PRODUCT,
    max_query_variants: int = DEFAULT_MAX_QUERY_VARIANTS,
) -> list[EvalQuery]:
    """Compatibility wrapper for weak metadata query generation."""
    return generate_weak_metadata_queries_impl(
        products=products,
        max_queries=max_queries,
        seed=seed,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    )


def evaluate_case(
    case: EvalQuery,
    before_response: RetrievalResponse,
    after_response: RetrievalResponse,
    products: Sequence[ProductItem],
    product_lookup: dict[str, ProductItem],
    metric_k: int,
    trained_response: RetrievalResponse | None = None,
    pairwise_response: RetrievalResponse | None = None,
) -> dict[str, Any]:
    """Evaluate one query for vector, rule, and optional trained rerankers."""
    positive_count = _positive_count(case, products)
    ideal_grades = _ideal_relevance_grades(case, products)
    vector_grades = [
        relevance_grade(case, result, product_lookup=product_lookup)
        for result in before_response.results
    ]
    rule_grades = [
        relevance_grade(case, result, product_lookup=product_lookup)
        for result in after_response.results
    ]
    vector_metrics = calculate_ranking_metrics(
        vector_grades,
        positive_count=positive_count,
        k=metric_k,
        ideal_relevance_grades=ideal_grades,
    )
    rule_metrics = calculate_ranking_metrics(
        rule_grades,
        positive_count=positive_count,
        k=metric_k,
        ideal_relevance_grades=ideal_grades,
    )

    detail = {
        "query_id": case.query_id,
        "query_text": _case_query_text(case),
        "label_source": case.label_source,
        "query_template_name": case.query_template_name,
        "query_template_fields": list(case.query_template_fields),
        "query_generation_mode": case.query_generation_mode,
        "positive_count": positive_count,
        "vector_metrics": vector_metrics,
        "rule_metrics": rule_metrics,
        "before_metrics": vector_metrics,
        "after_metrics": rule_metrics,
        "vector_top_k": _top_k_details(
            before_response.results,
            vector_grades,
            metric_k=metric_k,
            score_name="vector_score",
        ),
        "rule_top_k": _top_k_details(
            after_response.results,
            rule_grades,
            metric_k=metric_k,
            score_name="rule_rerank_score",
        ),
    }
    detail["before_top_k"] = detail["vector_top_k"]
    detail["after_top_k"] = detail["rule_top_k"]

    if trained_response is not None:
        trained_grades = [
            relevance_grade(case, result, product_lookup=product_lookup)
            for result in trained_response.results
        ]
        trained_metrics = calculate_ranking_metrics(
            trained_grades,
            positive_count=positive_count,
            k=metric_k,
            ideal_relevance_grades=ideal_grades,
        )
        detail["binary_trained_metrics"] = trained_metrics
        detail["trained_metrics"] = trained_metrics
        detail["binary_trained_top_k"] = _top_k_details(
            trained_response.results,
            trained_grades,
            metric_k=metric_k,
            score_name="binary_trained_rerank_score",
        )
        detail["trained_top_k"] = _top_k_details(
            trained_response.results,
            trained_grades,
            metric_k=metric_k,
            score_name="trained_rerank_score",
        )
    if pairwise_response is not None:
        pairwise_grades = [
            relevance_grade(case, result, product_lookup=product_lookup)
            for result in pairwise_response.results
        ]
        pairwise_metrics = calculate_ranking_metrics(
            pairwise_grades,
            positive_count=positive_count,
            k=metric_k,
            ideal_relevance_grades=ideal_grades,
        )
        detail["pairwise_metrics"] = pairwise_metrics
        detail["pairwise_top_k"] = _top_k_details(
            pairwise_response.results,
            pairwise_grades,
            metric_k=metric_k,
            score_name="pairwise_rerank_score",
        )
    return detail


def rerank_with_trained_model(
    response: RetrievalResponse,
    query_text: str,
    trained_bundle: TrainedRerankerBundle,
) -> RetrievalResponse:
    """Rerank one candidate pool using the loaded sklearn reranker."""
    if not response.results:
        return response
    if not hasattr(trained_bundle.model, "predict_proba"):
        raise ValueError("trained reranker model must provide predict_proba().")

    x_rows = _trained_feature_matrix(
        results=response.results,
        query_text=query_text,
        feature_names=trained_bundle.feature_names,
    )
    probabilities = trained_bundle.model.predict_proba(x_rows)
    trained_scores = [float(row[1]) for row in probabilities]
    scored_results = list(zip(response.results, trained_scores))
    scored_results.sort(key=lambda item: (-item[1], item[0].recall_rank))

    reranked_results: list[RetrievalResult] = []
    for final_rank, (result, score) in enumerate(scored_results, start=1):
        reranked_results.append(
            replace(
                result,
                rank=final_rank,
                rerank_score=score,
                final_rank=final_rank,
            )
        )
    return replace(response, results=reranked_results)


def rerank_with_pairwise_model(
    response: RetrievalResponse,
    query_text: str,
    pairwise_bundle: PairwiseRerankerBundle,
) -> RetrievalResponse:
    """Rerank one candidate pool using the loaded pairwise ranker."""
    if not response.results:
        return response
    if not hasattr(pairwise_bundle.model, "score_items"):
        raise ValueError("pairwise reranker model must provide score_items().")

    x_rows = _trained_feature_matrix(
        results=response.results,
        query_text=query_text,
        feature_names=pairwise_bundle.feature_names,
    )
    pairwise_scores = [float(score) for score in pairwise_bundle.model.score_items(x_rows)]
    scored_results = list(zip(response.results, pairwise_scores))
    scored_results.sort(key=lambda item: (-item[1], item[0].recall_rank))

    reranked_results: list[RetrievalResult] = []
    for final_rank, (result, score) in enumerate(scored_results, start=1):
        reranked_results.append(
            replace(
                result,
                rank=final_rank,
                rerank_score=score,
                final_rank=final_rank,
            )
        )
    return replace(response, results=reranked_results)


def relevance_grade(
    case: EvalQuery,
    result: RetrievalResult | ProductItem,
    product_lookup: dict[str, ProductItem] | None = None,
) -> int:
    """Return explicit-positive or weak-metadata relevance grade."""
    positive_ids = set(case.positive_product_ids)
    product_id = result.product_id
    if positive_ids:
        return 1 if product_id in positive_ids else 0

    product = result
    if not isinstance(result, ProductItem) and product_lookup is not None:
        product = product_lookup.get(product_id, result)
    return _weak_metadata_relevance(case, product)


def flatten_comparison_metrics(
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    metric_k: int,
    binary_trained_metrics: dict[str, float] | None = None,
    pairwise_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    """Flatten metric sets for API compatibility and four-way comparison."""
    metric_name_map = {
        "precision_at_k": f"precision_at_{metric_k}",
        "recall_at_k": f"recall_at_{metric_k}",
        "hit_at_k": f"hit_at_{metric_k}",
        "ndcg_at_k": f"ndcg_at_{metric_k}",
        "mrr": "mrr",
    }
    output: dict[str, float] = {}
    for key, flat_name in metric_name_map.items():
        output[f"{flat_name}_before_rerank"] = float(before_metrics.get(key, 0.0))
        output[f"{flat_name}_after_rerank"] = float(after_metrics.get(key, 0.0))
        output[f"{flat_name}_vector_recall"] = float(before_metrics.get(key, 0.0))
        output[f"{flat_name}_rule_rerank"] = float(after_metrics.get(key, 0.0))
        if binary_trained_metrics is not None:
            output[f"{flat_name}_binary_trained_rerank"] = float(
                binary_trained_metrics.get(key, 0.0)
            )
            output[f"{flat_name}_trained_rerank"] = float(
                binary_trained_metrics.get(key, 0.0)
            )
        if pairwise_metrics is not None:
            output[f"{flat_name}_pairwise_rerank"] = float(
                pairwise_metrics.get(key, 0.0)
            )
    return output


def get_placeholder_summary() -> EvaluationSummaryResponse:
    """Return an explicit no-report placeholder response."""
    return EvaluationSummaryResponse(
        metrics=empty_metrics(),
        prepared=False,
        placeholder=True,
        message="尚未生成评估报告，请运行 python backend/scripts/run_evaluation.py。",
    )


def _validate_runner_args(metric_k: int, candidate_k: int, max_queries: int) -> None:
    """Validate runner sizes before loading models or data."""
    if metric_k <= 0:
        raise ValueError("metric_k 必须大于 0。")
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0。")
    if max_queries <= 0:
        raise ValueError("max_queries 必须大于 0。")
    if metric_k > candidate_k:
        raise ValueError("metric_k 不能大于 candidate_k。")


def _summary_message(
    include_trained_reranker: bool,
    include_pairwise_reranker: bool,
) -> str:
    """Return a concise summary message for the enabled comparison modes."""
    if include_trained_reranker and include_pairwise_reranker:
        return "离线弱监督四方排序评估完成。"
    if include_trained_reranker:
        return "离线弱监督三方排序评估完成。"
    if include_pairwise_reranker:
        return "离线弱监督 vector/rule/pairwise 排序评估完成。"
    return "离线弱监督评估完成。"


def _metric_delta(
    target_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    """Return target minus baseline for every observed metric key."""
    return {
        key: target_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
        for key in sorted(set(target_metrics) | set(baseline_metrics))
    }


def _trained_feature_matrix(
    results: Sequence[RetrievalResult],
    query_text: str,
    feature_names: Sequence[str],
) -> list[list[float]]:
    """Build a feature matrix for trained reranker inference."""
    rows: list[list[float]] = []
    for result in results:
        feature = build_rerank_feature(result, query_text=query_text)
        feature_dict = rerank_feature_to_dict(feature)
        rows.append([float(feature_dict[name]) for name in feature_names])
    return rows


def _case_query_text(case: EvalQuery) -> str:
    """Return the normalized query text from the eval case."""
    return str(case.query_text or case.query or "").strip()


def _build_weak_query_text(product: ProductItem) -> str:
    """Build a simple deterministic text query from metadata."""
    parts = [
        product.gender,
        product.base_colour,
        product.article_type,
    ]
    return " ".join(str(part).strip().lower() for part in parts if part)


def _positive_count(case: EvalQuery, products: Sequence[ProductItem]) -> int:
    """Count explicit positives or weak metadata positives in the product corpus."""
    if case.positive_product_ids:
        return len(set(case.positive_product_ids))
    return sum(1 for product in products if relevance_grade(case, product) > 0)


def _ideal_relevance_grades(case: EvalQuery, products: Sequence[ProductItem]) -> list[int]:
    """Build ideal relevance grades from explicit positives or all products."""
    if case.positive_product_ids:
        return [1 for _ in set(case.positive_product_ids)]
    return [relevance_grade(case, product) for product in products]


def _weak_metadata_relevance(case: EvalQuery, result: RetrievalResult | ProductItem) -> int:
    """Apply first-version weak metadata relevance grades."""
    article_match = _field_equal(case.expected_article_type, result.article_type)
    color_match = _field_equal(case.expected_base_colour, result.base_colour)
    gender = getattr(result, "gender", None)
    gender_match = _field_equal(case.expected_gender, gender)
    sub_category = getattr(result, "sub_category", None)
    sub_category_match = _field_equal(case.expected_sub_category, sub_category)

    if article_match and color_match and gender_match:
        return 3
    if article_match and color_match:
        return 2
    if article_match or sub_category_match:
        return 1
    return 0


def _field_equal(expected: str | None, actual: str | None) -> bool:
    """Case-insensitive string equality for metadata fields."""
    if expected is None or actual is None:
        return False
    return expected.strip().lower() == actual.strip().lower()


def _top_k_details(
    results: Sequence[RetrievalResult],
    relevance_grades: Sequence[int],
    metric_k: int,
    score_name: str,
) -> list[dict[str, Any]]:
    """Format top-k results for JSONL details."""
    rows: list[dict[str, Any]] = []
    for rank, (result, grade) in enumerate(
        zip(results[:metric_k], relevance_grades[:metric_k]),
        start=1,
    ):
        score = result.score if score_name == "vector_score" else result.rerank_score
        rows.append(
            {
                "product_id": result.product_id,
                "rank": rank,
                "recall_rank": result.recall_rank,
                "final_rank": result.final_rank,
                "score": float(score),
                "recall_score": float(result.score),
                "score_name": score_name,
                "relevance_grade": int(grade),
            }
        )
    return rows
