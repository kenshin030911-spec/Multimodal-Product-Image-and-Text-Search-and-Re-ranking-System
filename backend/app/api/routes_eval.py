"""Evaluation summary API."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.evaluation.eval_runner import get_placeholder_summary
from backend.app.evaluation.report_writer import load_evaluation_summary
from backend.app.schemas.evaluation import EvaluationMetrics, EvaluationSummaryResponse

router = APIRouter(prefix="/evaluation")


@router.get("/summary", response_model=EvaluationSummaryResponse)
def evaluation_summary() -> EvaluationSummaryResponse:
    """Read the latest offline evaluation report without running evaluation."""
    settings = get_settings()
    summary = load_evaluation_summary(settings.eval_reports_dir)
    if summary is None:
        return get_placeholder_summary()

    return EvaluationSummaryResponse(
        metrics=EvaluationMetrics.model_validate(summary.get("metrics", {})),
        prepared=bool(summary.get("prepared", True)),
        placeholder=False,
        generated_at=summary.get("generated_at"),
        query_count=int(summary.get("query_count", 0)),
        metric_k=summary.get("metric_k"),
        candidate_k=summary.get("candidate_k"),
        max_queries=summary.get("max_queries"),
        eval_source=summary.get("eval_source"),
        label_policy=summary.get("label_policy"),
        include_trained_reranker=bool(summary.get("include_trained_reranker", False)),
        vector_recall=_float_dict(summary.get("vector_recall", {})),
        rule_rerank=_float_dict(summary.get("rule_rerank", {})),
        trained_rerank=_float_dict(summary.get("trained_rerank", {})),
        delta_rule_vs_vector=_float_dict(summary.get("delta_rule_vs_vector", {})),
        delta_trained_vs_vector=_float_dict(summary.get("delta_trained_vs_vector", {})),
        delta_trained_vs_rule=_float_dict(summary.get("delta_trained_vs_rule", {})),
        before_rerank=_float_dict(summary.get("before_rerank", {})),
        after_rerank=_float_dict(summary.get("after_rerank", {})),
        delta=_float_dict(summary.get("delta", {})),
        summary_path=summary.get("summary_path"),
        details_path=summary.get("details_path"),
        text_report_path=summary.get("text_report_path"),
        message=str(summary.get("message", "已读取离线评估报告。")),
    )


def _float_dict(value: object) -> dict[str, float]:
    """Normalize metric dictionaries from JSON into float values."""
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(metric_value)
        for key, metric_value in value.items()
        if isinstance(metric_value, (int, float))
    }
