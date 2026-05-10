"""离线检索指标。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from backend.app.schemas.evaluation import EvaluationMetrics


def empty_metrics() -> EvaluationMetrics:
    """返回零值指标，兼容旧占位调用。"""
    return EvaluationMetrics()


def precision_at_k(relevance_grades: Sequence[float | int], k: int) -> float:
    """计算 Precision@K，相关性等级大于 0 视为相关。"""
    _validate_k(k)
    if not relevance_grades:
        return 0.0
    relevant_count = _count_relevant(_top_k(relevance_grades, k))
    return relevant_count / k


def recall_at_k(
    relevance_grades: Sequence[float | int],
    positive_count: int,
    k: int,
) -> float:
    """计算 Recall@K；没有正例时返回 0。"""
    _validate_k(k)
    if positive_count <= 0 or not relevance_grades:
        return 0.0
    relevant_count = _count_relevant(_top_k(relevance_grades, k))
    return relevant_count / positive_count


def hit_at_k(relevance_grades: Sequence[float | int], k: int) -> float:
    """计算 Hit@K，Top-K 内至少一个相关结果即为 1。"""
    _validate_k(k)
    return 1.0 if _count_relevant(_top_k(relevance_grades, k)) > 0 else 0.0


def mrr(relevance_grades: Sequence[float | int], k: int | None = None) -> float:
    """计算 MRR；如果传入 k，则只看 Top-K。"""
    if k is not None:
        _validate_k(k)
        grades = _top_k(relevance_grades, k)
    else:
        grades = list(relevance_grades)

    for index, grade in enumerate(grades, start=1):
        if grade > 0:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    relevance_grades: Sequence[float | int],
    k: int,
    ideal_relevance_grades: Sequence[float | int] | None = None,
) -> float:
    """计算 graded NDCG@K，支持显式传入全集理想相关性。"""
    _validate_k(k)
    if not relevance_grades:
        return 0.0

    dcg = _dcg(_top_k(relevance_grades, k))
    ideal_source = ideal_relevance_grades if ideal_relevance_grades is not None else relevance_grades
    ideal_grades = sorted((float(grade) for grade in ideal_source), reverse=True)
    ideal_dcg = _dcg(_top_k(ideal_grades, k))
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg / ideal_dcg


def calculate_ranking_metrics(
    relevance_grades: Sequence[float | int],
    positive_count: int,
    k: int,
    ideal_relevance_grades: Sequence[float | int] | None = None,
) -> dict[str, float]:
    """一次性计算单个 query 的核心排序指标。"""
    return {
        "precision_at_k": precision_at_k(relevance_grades, k),
        "recall_at_k": recall_at_k(relevance_grades, positive_count, k),
        "hit_at_k": hit_at_k(relevance_grades, k),
        "mrr": mrr(relevance_grades, k),
        "ndcg_at_k": ndcg_at_k(
            relevance_grades,
            k,
            ideal_relevance_grades=ideal_relevance_grades,
        ),
    }


def aggregate_metrics(metric_rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """对多个 query 的指标取均值；空输入返回核心指标零值。"""
    if not metric_rows:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }

    keys = sorted({key for row in metric_rows for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in metric_rows) / len(metric_rows)
        for key in keys
    }


def _validate_k(k: int) -> None:
    """校验 K，避免指标静默接受无效参数。"""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数。")


def _top_k(values: Sequence[float | int], k: int) -> list[float]:
    """截取 Top-K 并转成 float。"""
    return [float(value) for value in values[:k]]


def _count_relevant(relevance_grades: Sequence[float | int]) -> int:
    """统计相关等级大于 0 的结果数。"""
    return sum(1 for grade in relevance_grades if grade > 0)


def _dcg(relevance_grades: Sequence[float | int]) -> float:
    """计算 DCG，使用 2^rel - 1 的 graded gain。"""
    return sum(
        (math.pow(2.0, float(grade)) - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(relevance_grades, start=1)
    )
