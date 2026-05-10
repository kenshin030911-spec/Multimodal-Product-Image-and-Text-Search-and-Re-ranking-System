"""Classification metrics for lightweight reranker training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float],
) -> dict[str, Any]:
    """Calculate binary classification metrics with safe single-class handling."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    sample_count = int(y_true_array.size)
    warnings: list[str] = []

    if sample_count == 0:
        warnings.append("metric input is empty; scalar metrics are set to 0 or null.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "average_precision": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "positive_rate": 0.0,
            "sample_count": 0,
            "warnings": warnings,
        }

    unique_classes = set(int(value) for value in np.unique(y_true_array))
    roc_auc: float | None
    average_precision: float | None
    if len(unique_classes) < 2:
        warnings.append(
            "y_true contains a single class; roc_auc and average_precision are null.",
        )
        roc_auc = None
        average_precision = None
    else:
        roc_auc = float(roc_auc_score(y_true_array, y_score_array))
        average_precision = float(average_precision_score(y_true_array, y_score_array))

    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(y_true_array, y_pred_array, zero_division=0),
        ),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "confusion_matrix": confusion_matrix(
            y_true_array,
            y_pred_array,
            labels=[0, 1],
        ).astype(int).tolist(),
        "positive_rate": float(np.mean(y_true_array == 1)),
        "sample_count": sample_count,
        "warnings": warnings,
    }
