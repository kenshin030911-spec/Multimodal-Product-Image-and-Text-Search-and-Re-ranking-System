"""Weak metadata query generation for evaluation and reranker training."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from backend.app.schemas.evaluation import EvalQuery
from backend.app.schemas.product import ProductItem


QueryTemplateMode = Literal["basic", "augmented"]

DEFAULT_QUERY_TEMPLATES: QueryTemplateMode = "basic"
DEFAULT_QUERIES_PER_PRODUCT = 2
DEFAULT_MAX_QUERY_VARIANTS = 12
MAX_QUERY_TOKEN_COUNT = 8
WHITESPACE_RE = re.compile(r"\s+")

QUERY_GENERATION_NOTE = (
    "Weak metadata query generation. Augmented queries are template-based English "
    "queries, not real user search logs. Some broad queries may omit explicit "
    "color or gender, but labels still come from the source product's full "
    "metadata and can be strict. Augmented query evaluation does not prove the "
    "trained reranker is universally better than the rule reranker. This round "
    "does not retrain models or change the search API."
)


@dataclass(frozen=True)
class QueryTemplate:
    """Template definition for weak metadata query generation."""

    name: str
    pattern: str
    fields: tuple[str, ...]


BASIC_TEMPLATE = QueryTemplate(
    name="gender_color_article_type",
    pattern="{gender} {base_colour} {article_type}",
    fields=("gender", "base_colour", "article_type"),
)

AUGMENTED_TEMPLATES: tuple[QueryTemplate, ...] = (
    QueryTemplate(
        name="color_article_type",
        pattern="{base_colour} {article_type}",
        fields=("base_colour", "article_type"),
    ),
    QueryTemplate(
        name="gender_article_type",
        pattern="{gender} {article_type}",
        fields=("gender", "article_type"),
    ),
    QueryTemplate(
        name="gender_color_article_type",
        pattern="{gender} {base_colour} {article_type}",
        fields=("gender", "base_colour", "article_type"),
    ),
    QueryTemplate(
        name="usage_color_article_type",
        pattern="{usage} {base_colour} {article_type}",
        fields=("usage", "base_colour", "article_type"),
    ),
    QueryTemplate(
        name="season_article_type",
        pattern="{season} {article_type}",
        fields=("season", "article_type"),
    ),
    QueryTemplate(
        name="color_sub_category",
        pattern="{base_colour} {sub_category}",
        fields=("base_colour", "sub_category"),
    ),
    QueryTemplate(
        name="usage_article_type",
        pattern="{usage} {article_type}",
        fields=("usage", "article_type"),
    ),
    QueryTemplate(
        name="color_article_type_for_gender",
        pattern="{base_colour} {article_type} for {gender}",
        fields=("base_colour", "article_type", "gender"),
    ),
    QueryTemplate(
        name="usage_article_type_for_gender",
        pattern="{usage} {article_type} for {gender}",
        fields=("usage", "article_type", "gender"),
    ),
    QueryTemplate(
        name="season_color_article_type",
        pattern="{season} {base_colour} {article_type}",
        fields=("season", "base_colour", "article_type"),
    ),
    QueryTemplate(
        name="article_type_only",
        pattern="{article_type}",
        fields=("article_type",),
    ),
    QueryTemplate(
        name="sub_category_only",
        pattern="{sub_category}",
        fields=("sub_category",),
    ),
)


def generate_weak_metadata_queries(
    products: Sequence[ProductItem],
    max_queries: int,
    seed: int,
    query_templates: QueryTemplateMode = DEFAULT_QUERY_TEMPLATES,
    queries_per_product: int = DEFAULT_QUERIES_PER_PRODUCT,
    max_query_variants: int = DEFAULT_MAX_QUERY_VARIANTS,
) -> list[EvalQuery]:
    """Generate deterministic weak metadata eval queries from product metadata."""
    _validate_generation_args(
        max_queries=max_queries,
        query_templates=query_templates,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    )
    if query_templates == "basic":
        return _generate_basic_queries(
            products=products,
            max_queries=max_queries,
            seed=seed,
        )
    return _generate_augmented_queries(
        products=products,
        max_queries=max_queries,
        seed=seed,
        queries_per_product=queries_per_product,
        max_query_variants=max_query_variants,
    )


def query_template_names_for_mode(query_templates: QueryTemplateMode) -> list[str]:
    """Return template names used by a generation mode."""
    if query_templates == "basic":
        return [BASIC_TEMPLATE.name]
    if query_templates == "augmented":
        return [template.name for template in AUGMENTED_TEMPLATES]
    raise ValueError("query_templates 必须是 basic 或 augmented。")


def _generate_basic_queries(
    products: Sequence[ProductItem],
    max_queries: int,
    seed: int,
) -> list[EvalQuery]:
    """Preserve the original one-query-per-product weak metadata generation."""
    eligible = [product for product in products if _has_required_label_fields(product)]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    cases: list[EvalQuery] = []
    for index, product in enumerate(eligible[:max_queries], start=1):
        cases.append(
            _make_eval_query(
                index=index,
                product=product,
                query_text=_build_basic_query_text(product),
                template=BASIC_TEMPLATE,
                query_generation_mode="basic",
            )
        )
    return cases


def _generate_augmented_queries(
    products: Sequence[ProductItem],
    max_queries: int,
    seed: int,
    queries_per_product: int,
    max_query_variants: int,
) -> list[EvalQuery]:
    """Generate multiple realistic template variants per metadata-complete product."""
    eligible = [product for product in products if _has_required_label_fields(product)]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    cases: list[EvalQuery] = []
    seen_query_texts: set[str] = set()
    for product in eligible:
        if len(cases) >= max_queries:
            break
        variants = _render_augmented_variants(product)
        rng.shuffle(variants)
        variants = variants[:max_query_variants]

        selected_for_product = 0
        seen_for_product: set[str] = set()
        for query_text, template in variants:
            if len(cases) >= max_queries or selected_for_product >= queries_per_product:
                break
            if query_text in seen_for_product or query_text in seen_query_texts:
                continue
            seen_for_product.add(query_text)
            seen_query_texts.add(query_text)
            cases.append(
                _make_eval_query(
                    index=len(cases) + 1,
                    product=product,
                    query_text=query_text,
                    template=template,
                    query_generation_mode="augmented",
                )
            )
            selected_for_product += 1
    return cases


def _render_augmented_variants(product: ProductItem) -> list[tuple[str, QueryTemplate]]:
    """Render all valid augmented variants for one product."""
    variants: list[tuple[str, QueryTemplate]] = []
    for template in AUGMENTED_TEMPLATES:
        values = _template_values(product, template)
        if values is None:
            continue
        query_text = normalize_query_text(template.pattern.format(**values))
        if not _is_valid_query_text(query_text):
            continue
        variants.append((query_text, template))
    return variants


def normalize_query_text(text: str) -> str:
    """Lowercase, collapse spaces, and remove duplicate tokens while preserving order."""
    normalized = WHITESPACE_RE.sub(" ", text.strip().lower())
    tokens: list[str] = []
    seen: set[str] = set()
    for token in normalized.split(" "):
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return " ".join(tokens)


def _is_valid_query_text(query_text: str) -> bool:
    """Reject empty or overly long generated queries."""
    if not query_text:
        return False
    return len(query_text.split()) <= MAX_QUERY_TOKEN_COUNT


def _make_eval_query(
    index: int,
    product: ProductItem,
    query_text: str,
    template: QueryTemplate,
    query_generation_mode: QueryTemplateMode,
) -> EvalQuery:
    """Create one EvalQuery from a product and rendered template."""
    return EvalQuery(
        query_id=f"Q{index:06d}",
        query_text=query_text,
        expected_article_type=product.article_type,
        expected_base_colour=product.base_colour,
        expected_gender=product.gender,
        expected_sub_category=product.sub_category,
        positive_product_ids=[],
        source_product_id=product.product_id,
        label_source="weak_metadata",
        query_template_name=template.name,
        query_template_fields=list(template.fields),
        query_generation_mode=query_generation_mode,
    )


def _template_values(product: ProductItem, template: QueryTemplate) -> dict[str, str] | None:
    """Return stripped field values for a template, or None if any field is missing."""
    values: dict[str, str] = {}
    for field_name in template.fields:
        value = getattr(product, field_name)
        if value is None or not str(value).strip():
            return None
        values[field_name] = str(value).strip()
    return values


def _has_required_label_fields(product: ProductItem) -> bool:
    """Require full metadata used by the first-version weak relevance policy."""
    return bool(
        product.article_type
        and product.base_colour
        and product.gender
        and product.sub_category
    )


def _build_basic_query_text(product: ProductItem) -> str:
    """Build the original gender-color-article query text."""
    parts = [
        product.gender,
        product.base_colour,
        product.article_type,
    ]
    return " ".join(str(part).strip().lower() for part in parts if part)


def _validate_generation_args(
    max_queries: int,
    query_templates: str,
    queries_per_product: int,
    max_query_variants: int,
) -> None:
    """Validate generation parameters."""
    if max_queries <= 0:
        raise ValueError("max_queries 必须大于 0。")
    if query_templates not in {"basic", "augmented"}:
        raise ValueError("query_templates 必须是 basic 或 augmented。")
    if queries_per_product <= 0:
        raise ValueError("queries_per_product 必须大于 0。")
    if max_query_variants <= 0:
        raise ValueError("max_query_variants 必须大于 0。")
