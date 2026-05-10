"""Tests for trained reranker case analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from backend.app.evaluation.case_analysis import (
    CASE_ANALYSIS_JSON_FILE,
    CASE_ANALYSIS_TEXT_FILE,
    analyze_case,
    analyze_evaluation_cases,
)


def test_analyze_case_classifies_trained_better_and_vector_multilabel() -> None:
    """A case can have a primary category plus additional category labels."""
    case = analyze_case(
        _detail(
            query_id="Q1",
            positive_count=3,
            vector_ndcg=0.20,
            rule_ndcg=0.50,
            trained_ndcg=0.70,
            rule_top_grades=[2, 1],
            trained_top_grades=[3, 3],
        ),
        threshold=0.01,
    )

    assert case["primary_category"] == "trained_better_than_rule"
    assert "trained_better_than_rule" in case["categories"]
    assert "trained_better_than_vector" in case["categories"]
    assert "更多 relevance_grade>=3" in case["explanation"]


def test_analyze_case_classifies_trained_worse_than_rule() -> None:
    """A trained reranker regression is reported as worse than rule."""
    case = analyze_case(
        _detail(
            query_id="Q2",
            positive_count=2,
            vector_ndcg=0.55,
            rule_ndcg=0.80,
            trained_ndcg=0.60,
            rule_top_grades=[3, 3],
            trained_top_grades=[2, 1],
        ),
        threshold=0.01,
    )

    assert case["primary_category"] == "trained_worse_than_rule"
    assert "trained_worse_than_rule" in case["categories"]
    assert "rule_better_than_trained" in case["categories"]
    assert "结果减少" in case["explanation"]


def test_analyze_case_classifies_all_same() -> None:
    """Near-identical NDCG/MRR values are classified as all_same."""
    case = analyze_case(
        _detail(
            query_id="Q3",
            positive_count=1,
            vector_ndcg=0.50,
            rule_ndcg=0.505,
            trained_ndcg=0.501,
            vector_mrr=0.50,
            rule_mrr=0.505,
            trained_mrr=0.501,
        ),
        threshold=0.01,
    )

    assert case["primary_category"] == "all_same"
    assert "all_same" in case["categories"]


def test_analyze_case_classifies_no_positive_found() -> None:
    """positive_count=0 takes primary classification precedence."""
    case = analyze_case(
        _detail(
            query_id="Q4",
            positive_count=0,
            vector_ndcg=0.0,
            rule_ndcg=0.0,
            trained_ndcg=0.0,
        ),
        threshold=0.01,
    )

    assert case["primary_category"] == "no_positive_found"
    assert "no_positive_found" in case["categories"]


def test_analyze_case_uses_mrr_tie_break_when_ndcg_is_close() -> None:
    """MRR decides the primary category when NDCG difference is below threshold."""
    case = analyze_case(
        _detail(
            query_id="Q5",
            positive_count=1,
            vector_ndcg=0.50,
            rule_ndcg=0.50,
            trained_ndcg=0.505,
            vector_mrr=0.40,
            rule_mrr=0.50,
            trained_mrr=0.70,
        ),
        threshold=0.01,
    )

    assert case["primary_category"] == "trained_better_than_rule"


def test_analyze_evaluation_cases_writes_json_and_text(tmp_path: Path) -> None:
    """Case analysis writes both structured and readable reports."""
    paths = _write_analysis_inputs(tmp_path)

    result = analyze_evaluation_cases(
        summary_path=paths["summary"],
        details_path=paths["details"],
        output_dir=tmp_path / "outputs" / "eval_reports",
        top_n=2,
        threshold=0.01,
        project_root=tmp_path,
    )

    json_path = tmp_path / "outputs" / "eval_reports" / CASE_ANALYSIS_JSON_FILE
    text_path = tmp_path / "outputs" / "eval_reports" / CASE_ANALYSIS_TEXT_FILE
    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    saved_text = text_path.read_text(encoding="utf-8")

    assert json_path.is_file()
    assert text_path.is_file()
    assert result.report["summary_stats"]["stability_label"] == "trained_worse_than_rule"
    assert result.report["summary_stats"]["recommendation"] == "暂不建议接 API"
    assert saved_json["counts_by_category"]["trained_better_than_rule"] >= 1
    assert saved_json["counts_by_category"]["trained_worse_than_rule"] >= 1
    assert saved_json["counts_by_category"]["no_positive_found"] == 1
    assert "Trained Reranker Case Analysis" in saved_text
    assert "弱监督评估" in saved_text
    assert "Top Improved Cases" in saved_text
    assert "Top Degraded Cases" in saved_text
    assert "Unchanged Samples" in saved_text


def test_analyze_evaluation_cases_rejects_missing_trained_metrics(tmp_path: Path) -> None:
    """Details without trained_metrics fail with the requested command hint."""
    summary_path = tmp_path / "evaluation_summary.json"
    details_path = tmp_path / "evaluation_details.jsonl"
    summary_path.write_text(
        json.dumps({"trained_rerank": {"ndcg_at_k": 0.5}}),
        encoding="utf-8",
    )
    details_path.write_text(
        json.dumps({"query_id": "Q1", "vector_metrics": {}, "rule_metrics": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--include-trained-reranker"):
        analyze_evaluation_cases(
            summary_path=summary_path,
            details_path=details_path,
            output_dir=tmp_path,
            project_root=tmp_path,
        )


def test_analyze_evaluation_cases_help() -> None:
    """analyze_evaluation_cases.py --help is available without loading models."""
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "backend/scripts/analyze_evaluation_cases.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--summary-path" in completed.stdout
    assert "--threshold" in completed.stdout


def test_analyze_evaluation_cases_cli_runs_with_tmp_inputs(tmp_path: Path) -> None:
    """CLI can analyze tmp_path inputs and write reports."""
    paths = _write_analysis_inputs(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "analysis"

    completed = subprocess.run(
        [
            sys.executable,
            "backend/scripts/analyze_evaluation_cases.py",
            "--summary-path",
            str(paths["summary"]),
            "--details-path",
            str(paths["details"]),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
            "--threshold",
            "0.01",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "stability_label" in completed.stdout
    assert (output_dir / CASE_ANALYSIS_JSON_FILE).is_file()
    assert (output_dir / CASE_ANALYSIS_TEXT_FILE).is_file()


def _write_analysis_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write a compact summary/details pair covering all main categories."""
    report_dir = tmp_path / "outputs" / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "evaluation_summary.json"
    details_path = report_dir / "evaluation_details.jsonl"
    summary = {
        "metric_k": 10,
        "candidate_k": 50,
        "max_queries": 5,
        "vector_recall": {"ndcg_at_k": 0.60, "mrr": 0.70, "precision_at_k": 0.7},
        "rule_rerank": {"ndcg_at_k": 0.90, "mrr": 0.95, "precision_at_k": 0.9},
        "trained_rerank": {"ndcg_at_k": 0.84, "mrr": 0.93, "precision_at_k": 0.88},
        "delta_trained_vs_vector": {"ndcg_at_k": 0.24, "mrr": 0.23},
        "delta_trained_vs_rule": {"ndcg_at_k": -0.06, "mrr": -0.02},
    }
    details = [
        _detail("Q1", 2, 0.20, 0.50, 0.70, rule_top_grades=[2], trained_top_grades=[3]),
        _detail("Q2", 2, 0.55, 0.80, 0.60, rule_top_grades=[3], trained_top_grades=[1]),
        _detail("Q3", 1, 0.50, 0.505, 0.501),
        _detail("Q4", 0, 0.0, 0.0, 0.0),
        _detail("Q5", 1, 0.50, 0.50, 0.505, vector_mrr=0.4, rule_mrr=0.5, trained_mrr=0.7),
    ]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with details_path.open("w", encoding="utf-8") as file:
        for detail in details:
            file.write(json.dumps(detail))
            file.write("\n")
    return {"summary": summary_path, "details": details_path}


def _detail(
    query_id: str,
    positive_count: int,
    vector_ndcg: float,
    rule_ndcg: float,
    trained_ndcg: float,
    *,
    vector_mrr: float = 0.5,
    rule_mrr: float = 0.5,
    trained_mrr: float = 0.5,
    rule_top_grades: list[int] | None = None,
    trained_top_grades: list[int] | None = None,
) -> dict:
    """Build one evaluation detail row."""
    return {
        "query_id": query_id,
        "query_text": f"query {query_id}",
        "label_source": "weak_metadata",
        "positive_count": positive_count,
        "vector_metrics": {"ndcg_at_k": vector_ndcg, "mrr": vector_mrr},
        "rule_metrics": {"ndcg_at_k": rule_ndcg, "mrr": rule_mrr},
        "trained_metrics": {"ndcg_at_k": trained_ndcg, "mrr": trained_mrr},
        "vector_top_k": _top_rows([1, 2]),
        "rule_top_k": _top_rows(rule_top_grades or [2, 2]),
        "trained_top_k": _top_rows(trained_top_grades or [2, 2]),
    }


def _top_rows(grades: list[int]) -> list[dict]:
    """Build compact top-k rows with relevance grades."""
    return [
        {
            "product_id": f"p{index}",
            "recall_rank": index,
            "final_rank": index,
            "score": 1.0 / index,
            "recall_score": 1.0 / index,
            "score_name": "test_score",
            "relevance_grade": grade,
        }
        for index, grade in enumerate(grades, start=1)
    ]
