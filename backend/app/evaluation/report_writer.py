"""Offline evaluation report readers and writers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative


SUMMARY_FILE = "evaluation_summary.json"
DETAILS_FILE = "evaluation_details.jsonl"
TEXT_SUMMARY_FILE = "evaluation_summary.txt"


def default_eval_report_dir() -> Path:
    """Return the default evaluation report directory."""
    return get_settings().eval_reports_dir


def write_evaluation_reports(
    summary: dict[str, Any],
    details: list[dict[str, Any]],
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Write summary JSON, per-query JSONL details, and a text summary."""
    settings = get_settings()
    output_dir = output_dir or settings.eval_reports_dir
    project_root = project_root or settings.project_root
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / SUMMARY_FILE
    details_path = output_dir / DETAILS_FILE
    text_summary_path = output_dir / TEXT_SUMMARY_FILE
    enriched_summary = dict(summary)
    enriched_summary.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    enriched_summary["summary_path"] = to_project_relative(summary_path, project_root)
    enriched_summary["details_path"] = to_project_relative(details_path, project_root)
    enriched_summary["text_report_path"] = to_project_relative(text_summary_path, project_root)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(enriched_summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with details_path.open("w", encoding="utf-8") as file:
        for detail in details:
            file.write(json.dumps(detail, ensure_ascii=False))
            file.write("\n")

    text_summary_path.write_text(format_text_summary(enriched_summary), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "details_path": details_path,
        "text_report_path": text_summary_path,
    }


def load_evaluation_summary(output_dir: Path | None = None) -> dict[str, Any] | None:
    """Read evaluation_summary.json if it exists and is valid JSON."""
    output_dir = output_dir or default_eval_report_dir()
    summary_path = output_dir / SUMMARY_FILE
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def format_text_summary(summary: dict[str, Any]) -> str:
    """Build a readable summary for CLI output and interview explanation."""
    vector = summary.get("vector_recall") or summary.get("before_rerank", {})
    rule = summary.get("rule_rerank") or summary.get("after_rerank", {})
    binary_trained = summary.get("binary_trained_rerank") or summary.get("trained_rerank")
    pairwise = summary.get("pairwise_rerank")
    delta_rule_vs_vector = summary.get("delta_rule_vs_vector") or summary.get("delta", {})
    delta_binary_trained_vs_vector = (
        summary.get("delta_binary_trained_vs_vector")
        or summary.get("delta_trained_vs_vector")
        or {}
    )
    delta_binary_trained_vs_rule = (
        summary.get("delta_binary_trained_vs_rule")
        or summary.get("delta_trained_vs_rule")
        or {}
    )
    delta_pairwise_vs_vector = summary.get("delta_pairwise_vs_vector") or {}
    delta_pairwise_vs_rule = summary.get("delta_pairwise_vs_rule") or {}
    delta_pairwise_vs_binary = summary.get("delta_pairwise_vs_binary_trained") or {}
    include_binary_trained = bool(summary.get("include_trained_reranker")) and isinstance(
        binary_trained,
        dict,
    )
    include_pairwise = bool(summary.get("include_pairwise_reranker")) and isinstance(
        pairwise,
        dict,
    )
    lines = [
        "Fashion Multimodal Search Evaluation Summary",
        f"prepared: {summary.get('prepared', False)}",
        f"generated_at: {summary.get('generated_at', '')}",
        f"query_count: {summary.get('query_count', 0)}",
        f"metric_k: {summary.get('metric_k')}",
        f"candidate_k: {summary.get('candidate_k')}",
        f"max_queries: {summary.get('max_queries')}",
        f"eval_source: {summary.get('eval_source', '')}",
        f"include_trained_reranker: {summary.get('include_trained_reranker', False)}",
        f"include_pairwise_reranker: {summary.get('include_pairwise_reranker', False)}",
        "",
        "Query Generation:",
        f"mode: {summary.get('query_generation_mode', '')}",
        f"requested_templates: {summary.get('query_templates', '')}",
        f"queries_per_product: {summary.get('queries_per_product', '')}",
        f"max_query_variants: {summary.get('max_query_variants', '')}",
        "template_names: "
        f"{', '.join(summary.get('query_template_names') or [])}",
        f"source: {summary.get('eval_source', '')}",
        "note: "
        + str(
            summary.get(
                "query_generation_note",
                "Weak metadata queries are not real user search logs.",
            )
        ),
        "",
        "Label Policy:",
        str(summary.get("label_policy", "")),
        "",
        "Metrics:",
    ]
    for key in ("precision_at_k", "recall_at_k", "hit_at_k", "mrr", "ndcg_at_k"):
        line = (
            f"- {key}: vector={float(vector.get(key, 0.0)):.6f}, "
            f"rule={float(rule.get(key, 0.0)):.6f}, "
            f"rule_vs_vector={float(delta_rule_vs_vector.get(key, 0.0)):.6f}"
        )
        if include_binary_trained:
            line += (
                f", binary_trained={float(binary_trained.get(key, 0.0)):.6f}, "
                f"binary_trained_vs_vector={float(delta_binary_trained_vs_vector.get(key, 0.0)):.6f}, "
                f"binary_trained_vs_rule={float(delta_binary_trained_vs_rule.get(key, 0.0)):.6f}"
            )
        if include_pairwise:
            line += (
                f", pairwise={float(pairwise.get(key, 0.0)):.6f}, "
                f"pairwise_vs_vector={float(delta_pairwise_vs_vector.get(key, 0.0)):.6f}, "
                f"pairwise_vs_rule={float(delta_pairwise_vs_rule.get(key, 0.0)):.6f}"
            )
            if include_binary_trained:
                line += (
                    f", pairwise_vs_binary_trained="
                    f"{float(delta_pairwise_vs_binary.get(key, 0.0)):.6f}"
                )
        lines.append(line)
    if include_binary_trained or include_pairwise:
        lines.extend(["", "Score Definitions:"])
    if include_binary_trained:
        model_info = summary.get("trained_reranker_model", {})
        lines.extend(
            [
                "Binary Trained Reranker:",
                f"model_type: {model_info.get('model_type', '')}",
                f"framework: {model_info.get('framework', '')}",
                "score_definition: "
                f"{model_info.get('trained_rerank_score_definition', '')}",
                "score_note: probability-like score from predict_proba(features)[1]",
            ]
        )
    if include_pairwise:
        model_info = summary.get("pairwise_reranker_model", {})
        lines.extend(
            [
                "Pairwise Reranker:",
                f"model_type: {model_info.get('model_type', '')}",
                f"framework: {model_info.get('framework', '')}",
                f"score_definition: {model_info.get('score_definition', '')}",
                f"score_note: {model_info.get('score_note', '')}",
                "comparison_note: binary and pairwise scores are not directly comparable; "
                "compare ranking metrics only.",
            ]
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- This is weak-supervised offline evaluation, not large-scale manual annotation.",
            "- Metrics compare enabled rankers by ranking outcomes, not raw score scales.",
            "- Pairwise scores are ordering scores, not calibrated probabilities.",
            "- Pairwise classification F1 does not guarantee better search ranking.",
            "- If pairwise_rerank or trained_rerank is worse than rule_rerank, the report keeps that result as-is.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report_placeholder(output_path: Path) -> None:
    """Compatibility placeholder for old calls."""
    _ = output_path
    return None
