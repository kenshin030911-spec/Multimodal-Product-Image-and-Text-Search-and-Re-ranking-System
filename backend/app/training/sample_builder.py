"""Weak-supervised reranker training sample builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.data.dataset_loader import default_products_path, load_json_file, load_products
from backend.app.data.validators import to_project_relative
from backend.app.evaluation.eval_runner import (
    default_eval_queries_path,
    load_or_generate_eval_cases,
    relevance_grade,
)
from backend.app.evaluation.query_generation import (
    DEFAULT_MAX_QUERY_VARIANTS,
    DEFAULT_QUERIES_PER_PRODUCT,
    DEFAULT_QUERY_TEMPLATES,
    QUERY_GENERATION_NOTE,
    QueryTemplateMode,
    query_template_names_for_mode,
)
from backend.app.index.index_store import INDEX_META_FILE
from backend.app.reranker.feature_builder import build_rerank_feature
from backend.app.retrieval.candidate_builder import RetrievalResult
from backend.app.retrieval.text_search import search_text_to_image
from backend.app.schemas.evaluation import EvalQuery
from backend.app.schemas.product import ProductItem
from backend.app.training.dataset_splitter import split_query_ids
from backend.app.training.feature_exporter import FEATURE_NAMES, rerank_feature_to_dict


DEFAULT_CANDIDATE_K = 50
DEFAULT_MAX_QUERIES = 200
DEFAULT_MAX_POSITIVES_PER_QUERY = 20
DEFAULT_MAX_NEGATIVES_PER_QUERY = 10
DEFAULT_MIN_POSITIVES_PER_QUERY = 1
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 42

TRAIN_FILE = "reranker_train.jsonl"
VALID_FILE = "reranker_valid.jsonl"
META_FILE = "reranker_dataset_meta.json"

LABEL_POLICY = (
    "Weak-supervised binary labels for first-version reranker training. "
    "If positive_product_ids are provided, hits are label=1 and misses are label=0. "
    "Otherwise metadata relevance grade 3 and 2 are label=1, while grade 1 and 0 "
    "are label=0. relevance_grade is kept for future ranking or weighting. "
    "These labels are not large-scale manual annotations."
)
LABEL_MAPPING = {
    "explicit_positive_ids": {
        "hit": 1,
        "miss": 0,
    },
    "weak_metadata": {
        "3": 1,
        "2": 1,
        "1": 0,
        "0": 0,
    },
}


@dataclass(frozen=True)
class RerankerDatasetBuildResult:
    """Result returned by the dataset builder."""

    train_samples: list[dict[str, Any]]
    valid_samples: list[dict[str, Any]]
    meta: dict[str, Any]
    output_paths: dict[str, Path]


def build_reranker_dataset(
    eval_queries_path: Path | None = None,
    products_path: Path | None = None,
    output_dir: Path | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    max_queries: int = DEFAULT_MAX_QUERIES,
    max_positives_per_query: int = DEFAULT_MAX_POSITIVES_PER_QUERY,
    max_negatives_per_query: int = DEFAULT_MAX_NEGATIVES_PER_QUERY,
    min_positives_per_query: int = DEFAULT_MIN_POSITIVES_PER_QUERY,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    query_templates: QueryTemplateMode = DEFAULT_QUERY_TEMPLATES,
    queries_per_product: int = DEFAULT_QUERIES_PER_PRODUCT,
    max_query_variants: int = DEFAULT_MAX_QUERY_VARIANTS,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> RerankerDatasetBuildResult:
    """Build weak-supervised query-product samples for later reranker training."""
    _validate_args(
        candidate_k=candidate_k,
        max_queries=max_queries,
        max_positives_per_query=max_positives_per_query,
        max_negatives_per_query=max_negatives_per_query,
        min_positives_per_query=min_positives_per_query,
        train_ratio=train_ratio,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    )
    settings = get_settings()
    project_root = project_root or settings.project_root
    eval_queries_path = eval_queries_path or default_eval_queries_path()
    products_path = products_path or default_products_path(settings)
    output_dir = output_dir or settings.processed_data_dir
    output_paths = _output_paths(output_dir)
    _ensure_can_write(output_paths, overwrite=overwrite)

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
    product_lookup = {product.product_id: product for product in products}
    all_samples: list[dict[str, Any]] = []
    skipped_no_positive_query_count = 0

    for case in eval_cases:
        query_samples = _build_samples_for_query(
            case=case,
            products_path=products_path,
            product_lookup=product_lookup,
            candidate_k=candidate_k,
            max_positives_per_query=max_positives_per_query,
            max_negatives_per_query=max_negatives_per_query,
            min_positives_per_query=min_positives_per_query,
            device=device,
            project_root=project_root,
        )
        if not query_samples:
            skipped_no_positive_query_count += 1
            continue
        all_samples.extend(query_samples)

    split_by_query_id = split_query_ids(
        (sample["query_id"] for sample in all_samples),
        train_ratio=train_ratio,
        seed=seed,
    )
    train_samples: list[dict[str, Any]] = []
    valid_samples: list[dict[str, Any]] = []
    for sample in all_samples:
        split = split_by_query_id.get(sample["query_id"], "train")
        sample_with_split = {**sample, "split": split}
        if split == "train":
            train_samples.append(sample_with_split)
        else:
            valid_samples.append(sample_with_split)

    meta = _build_meta(
        train_samples=train_samples,
        valid_samples=valid_samples,
        query_count=len(eval_cases),
        used_query_count=len(split_by_query_id),
        skipped_no_positive_query_count=skipped_no_positive_query_count,
        candidate_k=candidate_k,
        max_queries=max_queries,
        max_positives_per_query=max_positives_per_query,
        max_negatives_per_query=max_negatives_per_query,
        min_positives_per_query=min_positives_per_query,
        train_ratio=train_ratio,
        seed=seed,
        eval_source=eval_source,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
        eval_queries_path=eval_queries_path,
        products_path=products_path,
        output_paths=output_paths,
        project_root=project_root,
    )
    _write_dataset_files(
        train_samples=train_samples,
        valid_samples=valid_samples,
        meta=meta,
        output_paths=output_paths,
    )
    return RerankerDatasetBuildResult(
        train_samples=train_samples,
        valid_samples=valid_samples,
        meta=meta,
        output_paths=output_paths,
    )


def label_from_relevance(case: EvalQuery, product_id: str, relevance_grade_value: int) -> int:
    """Map explicit positives or weak metadata grades into binary labels."""
    if case.positive_product_ids:
        return 1 if product_id in set(case.positive_product_ids) else 0
    return 1 if relevance_grade_value >= 2 else 0


def build_training_samples_placeholder() -> list[dict[str, object]]:
    """Compatibility placeholder retained for old imports."""
    return []


def _build_samples_for_query(
    case: EvalQuery,
    products_path: Path,
    product_lookup: dict[str, ProductItem],
    candidate_k: int,
    max_positives_per_query: int,
    max_negatives_per_query: int,
    min_positives_per_query: int,
    device: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    """Build and sample query-product pairs for one query."""
    query_text = _case_query_text(case)
    response = search_text_to_image(
        query_text=query_text,
        top_k=candidate_k,
        products_path=products_path,
        device=device,
        project_root=project_root,
    )
    positive_samples: list[dict[str, Any]] = []
    negative_samples: list[dict[str, Any]] = []

    for result in response.results:
        sample = _build_sample(
            case=case,
            query_text=query_text,
            result=result,
            product_lookup=product_lookup,
        )
        if sample["label"] == 1:
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)

    if len(positive_samples) < min_positives_per_query:
        return []

    sampled_positives = sorted(
        positive_samples,
        key=lambda sample: (
            -int(sample["relevance_grade"]),
            int(sample["recall_rank"]),
            -float(sample["recall_score"]),
        ),
    )[:max_positives_per_query]
    hard_negatives = sorted(
        negative_samples,
        key=lambda sample: (int(sample["recall_rank"]), -float(sample["recall_score"])),
    )[:max_negatives_per_query]
    return sampled_positives + hard_negatives


def _build_sample(
    case: EvalQuery,
    query_text: str,
    result: RetrievalResult,
    product_lookup: dict[str, ProductItem],
) -> dict[str, Any]:
    """Build one JSON-serializable training sample."""
    grade = relevance_grade(case, result, product_lookup=product_lookup)
    label = label_from_relevance(case, result.product_id, grade)
    feature = build_rerank_feature(result, query_text=query_text)
    return {
        "query_id": case.query_id,
        "query_text": query_text,
        "product_id": result.product_id,
        "label": int(label),
        "relevance_grade": int(grade),
        "label_source": case.label_source,
        "source_product_id": case.source_product_id,
        "recall_rank": int(result.recall_rank),
        "recall_score": float(result.score),
        "features": rerank_feature_to_dict(feature),
    }


def _build_meta(
    train_samples: list[dict[str, Any]],
    valid_samples: list[dict[str, Any]],
    query_count: int,
    used_query_count: int,
    skipped_no_positive_query_count: int,
    candidate_k: int,
    max_queries: int,
    max_positives_per_query: int,
    max_negatives_per_query: int,
    min_positives_per_query: int,
    train_ratio: float,
    seed: int,
    eval_source: str,
    query_templates: QueryTemplateMode,
    queries_per_product: int,
    max_query_variants: int,
    eval_queries_path: Path,
    products_path: Path,
    output_paths: dict[str, Path],
    project_root: Path,
) -> dict[str, Any]:
    """Build dataset metadata and label policy documentation."""
    all_samples = train_samples + valid_samples
    positive_count = sum(1 for sample in all_samples if sample["label"] == 1)
    negative_count = sum(1 for sample in all_samples if sample["label"] == 0)
    total_count = len(all_samples)
    generated_queries = eval_source != "eval_queries_jsonl"
    query_generation_mode = query_templates if generated_queries else "eval_queries_jsonl"
    query_template_names = (
        query_template_names_for_mode(query_templates) if generated_queries else []
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_count": query_count,
        "used_query_count": used_query_count,
        "skipped_no_positive_query_count": skipped_no_positive_query_count,
        "train_sample_count": len(train_samples),
        "valid_sample_count": len(valid_samples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / total_count if total_count else 0.0,
        "candidate_k": candidate_k,
        "max_queries": max_queries,
        "max_positives_per_query": max_positives_per_query,
        "max_negatives_per_query": max_negatives_per_query,
        "min_positives_per_query": min_positives_per_query,
        "train_ratio": train_ratio,
        "seed": seed,
        "eval_source": eval_source,
        "query_generation_mode": query_generation_mode,
        "query_templates": query_templates,
        "queries_per_product": queries_per_product,
        "max_query_variants": max_query_variants,
        "query_template_names": query_template_names,
        "query_generation_note": QUERY_GENERATION_NOTE,
        "label_policy": LABEL_POLICY,
        "label_mapping": LABEL_MAPPING,
        "positive_sampling_policy": (
            "For each query, keep at most max_positives_per_query positives sorted by "
            "relevance_grade descending, recall_rank ascending, recall_score descending."
        ),
        "negative_sampling_policy": (
            "For each query, keep at most max_negatives_per_query hard negatives sorted by "
            "recall_rank ascending, recall_score descending."
        ),
        "feature_names": list(FEATURE_NAMES),
        "source_index": _source_index(project_root),
        "eval_queries_path": to_project_relative(eval_queries_path, project_root),
        "products_path": to_project_relative(products_path, project_root),
        "train_path": to_project_relative(output_paths["train_path"], project_root),
        "valid_path": to_project_relative(output_paths["valid_path"], project_root),
        "meta_path": to_project_relative(output_paths["meta_path"], project_root),
        "note": "weak-supervised labels, not large-scale manual annotations",
    }


def _write_dataset_files(
    train_samples: list[dict[str, Any]],
    valid_samples: list[dict[str, Any]],
    meta: dict[str, Any],
    output_paths: dict[str, Path],
) -> None:
    """Write train/valid JSONL files and meta JSON."""
    output_paths["train_path"].parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_paths["train_path"], train_samples)
    _write_jsonl(output_paths["valid_path"], valid_samples)
    with output_paths["meta_path"].open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows with UTF-8 encoding."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def _output_paths(output_dir: Path) -> dict[str, Path]:
    """Return all output file paths."""
    return {
        "train_path": output_dir / TRAIN_FILE,
        "valid_path": output_dir / VALID_FILE,
        "meta_path": output_dir / META_FILE,
    }


def _ensure_can_write(output_paths: dict[str, Path], overwrite: bool) -> None:
    """Reject accidental overwrites unless explicitly requested."""
    if overwrite:
        return
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"reranker dataset 输出已存在，请使用 --overwrite 覆盖: {names}")


def _source_index(project_root: Path) -> dict[str, Any] | None:
    """Read lightweight index metadata for dataset lineage."""
    settings = get_settings()
    index_meta_path = settings.index_dir / INDEX_META_FILE
    meta = load_json_file(index_meta_path)
    if meta is None:
        return None
    return {
        "index_meta_file": to_project_relative(index_meta_path, project_root),
        "index_type": meta.get("index_type"),
        "metric": meta.get("metric"),
        "product_count": meta.get("product_count"),
        "embedding_dim": meta.get("embedding_dim"),
        "source_encoder_name": meta.get("source_encoder_name"),
        "source_model_name": meta.get("source_model_name"),
    }


def _case_query_text(case: EvalQuery) -> str:
    """Return the normalized text query."""
    query_text = str(case.query_text or case.query or "").strip()
    if not query_text:
        raise ValueError(f"query 为空: {case.query_id}")
    return query_text


def _validate_args(
    candidate_k: int,
    max_queries: int,
    max_positives_per_query: int,
    max_negatives_per_query: int,
    min_positives_per_query: int,
    train_ratio: float,
    query_templates: str,
    queries_per_product: int,
    max_query_variants: int,
) -> None:
    """Validate builder parameters before running retrieval."""
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0。")
    if max_queries <= 0:
        raise ValueError("max_queries 必须大于 0。")
    if max_positives_per_query <= 0:
        raise ValueError("max_positives_per_query 必须大于 0。")
    if max_negatives_per_query < 0:
        raise ValueError("max_negatives_per_query 必须大于或等于 0。")
    if min_positives_per_query <= 0:
        raise ValueError("min_positives_per_query 必须大于 0。")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio 必须在 0 和 1 之间。")
    if query_templates not in {"basic", "augmented"}:
        raise ValueError("query_templates 必须是 basic 或 augmented。")
    if queries_per_product <= 0:
        raise ValueError("queries_per_product 必须大于 0。")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants 必须大于 0。")
