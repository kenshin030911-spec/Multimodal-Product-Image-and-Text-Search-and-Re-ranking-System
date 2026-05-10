"""Offline case analysis for trained reranker evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.data.validators import to_project_relative
from backend.app.evaluation.report_writer import DETAILS_FILE, SUMMARY_FILE


CASE_ANALYSIS_JSON_FILE = "rerank_case_analysis.json"
CASE_ANALYSIS_TEXT_FILE = "rerank_case_analysis.txt"
DEFAULT_TOP_N = 5
DEFAULT_THRESHOLD = 0.01

CATEGORY_NAMES = (
    "trained_better_than_rule",
    "trained_worse_than_rule",
    "trained_better_than_vector",
    "rule_better_than_trained",
    "all_same",
    "no_positive_found",
)

WEAK_SUPERVISION_NOTE = (
    "当前评估是弱监督评估，不是人工标注；案例解释只能作为辅助，不能替代人工判断。"
)


@dataclass(frozen=True)
class CaseAnalysisResult:
    """In-memory result returned by case analysis."""

    report: dict[str, Any]
    output_paths: dict[str, Path]


def default_summary_path() -> Path:
    """Return the default evaluation summary path."""
    return get_settings().eval_reports_dir / SUMMARY_FILE


def default_details_path() -> Path:
    """Return the default evaluation details path."""
    return get_settings().eval_reports_dir / DETAILS_FILE


def default_output_dir() -> Path:
    """Return the default case-analysis output directory."""
    return get_settings().eval_reports_dir


def analyze_evaluation_cases(
    summary_path: Path | None = None,
    details_path: Path | None = None,
    output_dir: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    threshold: float = DEFAULT_THRESHOLD,
    project_root: Path | None = None,
) -> CaseAnalysisResult:
    """Analyze trained reranker stability and write JSON/TXT reports."""
    _validate_args(top_n=top_n, threshold=threshold)
    settings = get_settings()
    project_root = project_root or settings.project_root
    summary_path = summary_path or default_summary_path()
    details_path = details_path or default_details_path()
    output_dir = output_dir or default_output_dir()

    summary = _read_json_object(summary_path, label="evaluation summary")
    details = _read_jsonl(details_path, label="evaluation details")
    _validate_trained_report(summary=summary, details=details)

    cases = [analyze_case(detail, threshold=threshold) for detail in details]
    counts_by_category = _count_categories(cases)
    top_improved_cases = sorted(
        cases,
        key=lambda case: case["delta_trained_vs_rule"].get("ndcg_at_k", 0.0),
        reverse=True,
    )[:top_n]
    top_degraded_cases = sorted(
        cases,
        key=lambda case: case["delta_trained_vs_rule"].get("ndcg_at_k", 0.0),
    )[:top_n]
    unchanged_cases_sample = [
        case for case in cases if case["primary_category"] == "all_same"
    ][:top_n]
    summary_stats = build_stability_summary(summary, threshold=threshold)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_path": to_project_relative(summary_path, project_root),
        "details_path": to_project_relative(details_path, project_root),
        "threshold": threshold,
        "top_n": top_n,
        "summary_stats": {
            **summary_stats,
            "query_count": len(cases),
            "metric_k": summary.get("metric_k"),
            "candidate_k": summary.get("candidate_k"),
            "max_queries": summary.get("max_queries"),
        },
        "counts_by_category": counts_by_category,
        "top_improved_cases": top_improved_cases,
        "top_degraded_cases": top_degraded_cases,
        "unchanged_cases_sample": unchanged_cases_sample,
        "notes": [
            "当前评估是弱监督评估，不是人工标注。",
            "trained 和 rule 的整体差距可能很小。",
            "案例解释只能作为辅助，不能替代人工判断。",
            "trained 在部分 query 上变差也要如实报告。",
            "本轮结论只能决定是否值得做 optional API 接入，不应直接证明 trained reranker 全面更优。",
        ],
    }

    output_paths = _write_case_analysis_reports(
        report=report,
        output_dir=output_dir,
        project_root=project_root,
    )
    return CaseAnalysisResult(report=report, output_paths=output_paths)


def analyze_case(detail: dict[str, Any], threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Build one query-level case summary with category labels and explanation."""
    if "trained_metrics" not in detail:
        raise ValueError(
            "evaluation details missing trained_metrics; run "
            "python backend/scripts/run_evaluation.py --include-trained-reranker ... first."
        )

    vector_metrics = _metric_dict(detail.get("vector_metrics", {}))
    rule_metrics = _metric_dict(detail.get("rule_metrics", {}))
    trained_metrics = _metric_dict(detail.get("trained_metrics", {}))
    delta_trained_vs_rule = _metric_delta(trained_metrics, rule_metrics)
    delta_trained_vs_vector = _metric_delta(trained_metrics, vector_metrics)
    positive_count = int(detail.get("positive_count", 0) or 0)
    vector_top_k = _top_k_rows(detail.get("vector_top_k", []))
    rule_top_k = _top_k_rows(detail.get("rule_top_k", []))
    trained_top_k = _top_k_rows(detail.get("trained_top_k", []))
    primary_category = classify_primary_category(
        positive_count=positive_count,
        rule_metrics=rule_metrics,
        trained_metrics=trained_metrics,
        threshold=threshold,
    )
    categories = classify_categories(
        positive_count=positive_count,
        vector_metrics=vector_metrics,
        rule_metrics=rule_metrics,
        trained_metrics=trained_metrics,
        threshold=threshold,
    )
    return {
        "query_id": detail.get("query_id"),
        "query_text": detail.get("query_text"),
        "label_source": detail.get("label_source"),
        "positive_count": positive_count,
        "primary_category": primary_category,
        "categories": categories,
        "vector_metrics": vector_metrics,
        "rule_metrics": rule_metrics,
        "trained_metrics": trained_metrics,
        "delta_trained_vs_rule": delta_trained_vs_rule,
        "delta_trained_vs_vector": delta_trained_vs_vector,
        "vector_top_k": vector_top_k,
        "rule_top_k": rule_top_k,
        "trained_top_k": trained_top_k,
        "explanation": explain_case(
            primary_category=primary_category,
            rule_top_k=rule_top_k,
            trained_top_k=trained_top_k,
            delta_trained_vs_rule=delta_trained_vs_rule,
        ),
    }


def classify_primary_category(
    positive_count: int,
    rule_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Classify the main trained-vs-rule outcome for one query."""
    if positive_count == 0:
        return "no_positive_found"

    ndcg_delta = trained_metrics.get("ndcg_at_k", 0.0) - rule_metrics.get("ndcg_at_k", 0.0)
    if ndcg_delta > threshold:
        return "trained_better_than_rule"
    if ndcg_delta < -threshold:
        return "trained_worse_than_rule"

    mrr_delta = trained_metrics.get("mrr", 0.0) - rule_metrics.get("mrr", 0.0)
    if mrr_delta > threshold:
        return "trained_better_than_rule"
    if mrr_delta < -threshold:
        return "trained_worse_than_rule"
    return "all_same"


def classify_categories(
    positive_count: int,
    vector_metrics: dict[str, float],
    rule_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[str]:
    """Return all category labels matched by one query."""
    categories: list[str] = []
    if positive_count == 0:
        categories.append("no_positive_found")

    trained_vs_vector_ndcg = trained_metrics.get("ndcg_at_k", 0.0) - vector_metrics.get(
        "ndcg_at_k",
        0.0,
    )
    trained_vs_rule_ndcg = trained_metrics.get("ndcg_at_k", 0.0) - rule_metrics.get(
        "ndcg_at_k",
        0.0,
    )
    if trained_vs_vector_ndcg > threshold:
        categories.append("trained_better_than_vector")
    if trained_vs_rule_ndcg > threshold:
        categories.append("trained_better_than_rule")
    if trained_vs_rule_ndcg < -threshold:
        categories.extend(["trained_worse_than_rule", "rule_better_than_trained"])
    if _all_rankings_same_within_threshold(
        vector_metrics=vector_metrics,
        rule_metrics=rule_metrics,
        trained_metrics=trained_metrics,
        threshold=threshold,
    ):
        categories.append("all_same")
    return categories or ["all_same"]


def build_stability_summary(
    summary: dict[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Build overall trained-reranker stability labels from summary metrics."""
    vector = _metric_dict(summary.get("vector_recall", {}))
    rule = _metric_dict(summary.get("rule_rerank", {}))
    trained = _metric_dict(summary.get("trained_rerank", {}))
    if not trained:
        raise ValueError(
            "evaluation summary missing trained_rerank; run "
            "python backend/scripts/run_evaluation.py --include-trained-reranker ... first."
        )

    delta_trained_vs_vector = _metric_dict(
        summary.get("delta_trained_vs_vector")
        or _metric_delta(trained, vector)
    )
    delta_trained_vs_rule = _metric_dict(
        summary.get("delta_trained_vs_rule")
        or _metric_delta(trained, rule)
    )
    trained_vs_vector_ndcg = delta_trained_vs_vector.get("ndcg_at_k", 0.0)
    trained_vs_rule_ndcg = delta_trained_vs_rule.get("ndcg_at_k", 0.0)
    trained_vs_vector_positive = trained_vs_vector_ndcg > threshold
    trained_vs_rule_positive = trained_vs_rule_ndcg > threshold
    trained_vs_rule_negative = trained_vs_rule_ndcg < -threshold
    trained_vs_rule_comparable = abs(trained_vs_rule_ndcg) <= threshold

    if trained_vs_rule_negative:
        stability_label = "trained_worse_than_rule"
        recommendation = "暂不建议接 API"
    elif trained_vs_rule_positive:
        stability_label = "trained_better_than_rule"
        recommendation = "可考虑接入 API 做 optional/default 对比"
    elif trained_vs_rule_comparable:
        stability_label = "comparable"
        recommendation = "可作为 optional mode 接 API，但默认仍用 rule"
    else:
        stability_label = "unknown"
        recommendation = "需要人工查看"

    return {
        "vector_recall": vector,
        "rule_rerank": rule,
        "trained_rerank": trained,
        "delta_trained_vs_vector": delta_trained_vs_vector,
        "delta_trained_vs_rule": delta_trained_vs_rule,
        "trained_vs_vector_positive": trained_vs_vector_positive,
        "trained_vs_rule_positive": trained_vs_rule_positive,
        "trained_vs_rule_comparable": trained_vs_rule_comparable,
        "trained_vs_rule_negative": trained_vs_rule_negative,
        "stability_label": stability_label,
        "recommendation": recommendation,
    }


def explain_case(
    primary_category: str,
    rule_top_k: list[dict[str, Any]],
    trained_top_k: list[dict[str, Any]],
    delta_trained_vs_rule: dict[str, float],
) -> str:
    """Generate a conservative explanation from relevance-grade distribution only."""
    rule_high_count = _high_relevance_count(rule_top_k)
    trained_high_count = _high_relevance_count(trained_top_k)
    rule_first_high_rank = _first_high_relevance_rank(rule_top_k)
    trained_first_high_rank = _first_high_relevance_rank(trained_top_k)
    ndcg_delta = delta_trained_vs_rule.get("ndcg_at_k", 0.0)

    if primary_category == "no_positive_found":
        return "positive_count 为 0，无法基于当前弱监督标签判断排序好坏。"
    if trained_high_count > rule_high_count:
        return "trained 将更多 relevance_grade>=3 的商品排入 top_k。"
    if (
        trained_first_high_rank is not None
        and rule_first_high_rank is not None
        and trained_first_high_rank < rule_first_high_rank
    ):
        return "trained 将更高 relevance_grade 的商品提前。"
    if trained_high_count < rule_high_count:
        return "trained top_k 中高 relevance_grade 结果减少，排名下降。"
    if ndcg_delta < 0:
        return "trained 排序指标下降，但仅凭 relevance_grade 分布无法判断原因，需要人工查看。"
    if ndcg_delta > 0:
        return "trained 排序指标提升，但具体原因需要人工查看。"
    return "需要人工查看。"


def format_case_analysis_text(report: dict[str, Any]) -> str:
    """Format case-analysis output for README/interview use."""
    stats = report["summary_stats"]
    counts = report["counts_by_category"]
    lines = [
        "Trained Reranker Case Analysis",
        "",
        "Inputs",
        f"- summary_path: {report['summary_path']}",
        f"- details_path: {report['details_path']}",
        f"- threshold: {report['threshold']}",
        f"- top_n: {report['top_n']}",
        "",
        "Overall Three-Way Metrics",
    ]
    for metric_name in ("precision_at_k", "recall_at_k", "hit_at_k", "mrr", "ndcg_at_k"):
        lines.append(
            f"- {metric_name}: vector={_metric_value(stats['vector_recall'], metric_name):.6f}, "
            f"rule={_metric_value(stats['rule_rerank'], metric_name):.6f}, "
            f"trained={_metric_value(stats['trained_rerank'], metric_name):.6f}"
        )
    lines.extend(
        [
            "",
            "Stability",
            f"- stability_label: {stats['stability_label']}",
            f"- recommendation: {stats['recommendation']}",
            f"- trained_vs_vector_positive: {stats['trained_vs_vector_positive']}",
            f"- trained_vs_rule_positive: {stats['trained_vs_rule_positive']}",
            f"- trained_vs_rule_comparable: {stats['trained_vs_rule_comparable']}",
            f"- trained_vs_rule_negative: {stats['trained_vs_rule_negative']}",
            "",
            "Category Counts",
        ]
    )
    lines.extend(f"- {name}: {counts.get(name, 0)}" for name in CATEGORY_NAMES)
    lines.extend(["", "Top Improved Cases"])
    lines.extend(_format_case_list(report["top_improved_cases"]))
    lines.extend(["", "Top Degraded Cases"])
    lines.extend(_format_case_list(report["top_degraded_cases"]))
    lines.extend(["", "Unchanged Samples"])
    lines.extend(_format_case_list(report["unchanged_cases_sample"]))
    lines.extend(
        [
            "",
            "Notes",
            "- 当前评估是弱监督评估，不是人工标注。",
            "- trained 和 rule 的整体差距可能很小。",
            "- trained 分类指标不等于排序稳定提升。",
            "- 案例解释只能作为辅助，不能替代人工判断。",
            "- trained 在部分 query 上变差也要如实报告。",
            "- 本轮结论只能决定是否值得做 optional API 接入，不应直接证明 trained reranker 全面更优。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_case_analysis_reports(
    report: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Path]:
    """Write JSON and text case-analysis reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / CASE_ANALYSIS_JSON_FILE
    text_path = output_dir / CASE_ANALYSIS_TEXT_FILE
    json_report = {
        **report,
        "json_report_path": to_project_relative(json_path, project_root),
        "text_report_path": to_project_relative(text_path, project_root),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    text_path.write_text(format_case_analysis_text(json_report), encoding="utf-8")
    report.update(
        {
            "json_report_path": to_project_relative(json_path, project_root),
            "text_report_path": to_project_relative(text_path, project_root),
        }
    )
    return {"json_path": json_path, "text_path": text_path}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object with a clear missing-file error."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """Read JSONL rows with clear validation errors."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} must be a JSON object.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} contains no cases: {path}")
    return rows


def _validate_trained_report(summary: dict[str, Any], details: list[dict[str, Any]]) -> None:
    """Ensure inputs are from a trained-reranker evaluation run."""
    if "trained_rerank" not in summary:
        raise ValueError(
            "evaluation summary missing trained_rerank; run "
            "python backend/scripts/run_evaluation.py --include-trained-reranker ... first."
        )
    for detail in details:
        if "trained_metrics" not in detail:
            raise ValueError(
                "evaluation details missing trained_metrics; run "
                "python backend/scripts/run_evaluation.py --include-trained-reranker ... first."
            )


def _validate_args(top_n: int, threshold: float) -> None:
    """Validate case-analysis parameters."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    if threshold < 0:
        raise ValueError("threshold must be greater than or equal to 0.")


def _metric_delta(target: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """Return target minus baseline for every observed metric key."""
    return {
        key: target.get(key, 0.0) - baseline.get(key, 0.0)
        for key in sorted(set(target) | set(baseline))
    }


def _metric_dict(value: object) -> dict[str, float]:
    """Normalize metric objects into float dictionaries."""
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(metric_value)
        for key, metric_value in value.items()
        if isinstance(metric_value, (int, float))
    }


def _top_k_rows(value: object) -> list[dict[str, Any]]:
    """Normalize top-k rows while preserving useful fields."""
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _count_categories(cases: list[dict[str, Any]]) -> dict[str, int]:
    """Count multi-label categories across all cases."""
    counts = {name: 0 for name in CATEGORY_NAMES}
    for case in cases:
        for category in case.get("categories", []):
            counts[category] = counts.get(category, 0) + 1
    return counts


def _all_rankings_same_within_threshold(
    vector_metrics: dict[str, float],
    rule_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    threshold: float,
) -> bool:
    """Return whether vector/rule/trained NDCG and MRR are effectively tied."""
    metric_names = ("ndcg_at_k", "mrr")
    pairs = (
        (vector_metrics, rule_metrics),
        (vector_metrics, trained_metrics),
        (rule_metrics, trained_metrics),
    )
    return all(
        abs(left.get(metric_name, 0.0) - right.get(metric_name, 0.0)) <= threshold
        for left, right in pairs
        for metric_name in metric_names
    )


def _high_relevance_count(rows: list[dict[str, Any]]) -> int:
    """Count strong relevance grades in a top-k list."""
    return sum(1 for row in rows if int(row.get("relevance_grade", 0) or 0) >= 3)


def _first_high_relevance_rank(rows: list[dict[str, Any]]) -> int | None:
    """Return the first rank with relevance_grade >= 3."""
    for index, row in enumerate(rows, start=1):
        if int(row.get("relevance_grade", 0) or 0) >= 3:
            return int(row.get("final_rank", index) or index)
    return None


def _metric_value(metrics: dict[str, float], metric_name: str) -> float:
    """Read one metric value as float."""
    return float(metrics.get(metric_name, 0.0))


def _format_case_list(cases: list[dict[str, Any]]) -> list[str]:
    """Format a compact case list for text output."""
    if not cases:
        return ["- none"]
    lines: list[str] = []
    for case in cases:
        delta = case.get("delta_trained_vs_rule", {})
        lines.append(
            "- "
            f"{case.get('query_id')}: {case.get('primary_category')}, "
            f"ndcg_delta={float(delta.get('ndcg_at_k', 0.0)):.6f}, "
            f"mrr_delta={float(delta.get('mrr', 0.0)):.6f}, "
            f"query=\"{case.get('query_text', '')}\""
        )
        lines.append(f"  explanation: {case.get('explanation', '')}")
    return lines
