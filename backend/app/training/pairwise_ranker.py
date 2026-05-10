"""Pairwise logistic ranker with item-level scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.app.training.reranker_trainer import normalize_class_weight


@dataclass
class PairwiseLogisticRanker:
    """Train logistic regression on feature differences and score individual items."""

    feature_names: list[str]
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    solver: str = "lbfgs"
    seed: int = 2026
    scaler: StandardScaler | None = None
    classifier: LogisticRegression | None = None
    coef_: np.ndarray | None = None

    def fit_item_scaler(self, item_x: Sequence[Sequence[float]]) -> "PairwiseLogisticRanker":
        """Fit StandardScaler on item-level feature rows."""
        item_x_array = _as_2d_float_array(item_x, expected_width=len(self.feature_names))
        self.scaler = StandardScaler()
        self.scaler.fit(item_x_array)
        return self

    def transform_items(self, item_x: Sequence[Sequence[float]]) -> np.ndarray:
        """Transform item-level features using the fitted scaler."""
        if self.scaler is None:
            raise ValueError("PairwiseLogisticRanker scaler is not fitted.")
        item_x_array = _as_2d_float_array(item_x, expected_width=len(self.feature_names))
        return self.scaler.transform(item_x_array)

    def fit_classifier(
        self,
        pair_x_diff: Sequence[Sequence[float]],
        pair_y: Sequence[int],
    ) -> "PairwiseLogisticRanker":
        """Fit LogisticRegression on already-standardized feature differences."""
        x_diff = _as_2d_float_array(pair_x_diff, expected_width=len(self.feature_names))
        y = np.asarray(pair_y, dtype=int)
        if x_diff.shape[0] == 0:
            raise ValueError("pairwise training data is empty.")
        if set(int(value) for value in np.unique(y)) != {0, 1}:
            raise ValueError("pairwise training targets must contain both 0 and 1.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0.")

        self.classifier = LogisticRegression(
            fit_intercept=False,
            class_weight=normalize_class_weight(self.class_weight),
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.seed,
        )
        self.classifier.fit(x_diff, y)
        self.coef_ = np.asarray(self.classifier.coef_[0], dtype=float)
        return self

    def fit(
        self,
        item_x: Sequence[Sequence[float]],
        pair_x_diff: Sequence[Sequence[float]],
        pair_y: Sequence[int],
    ) -> "PairwiseLogisticRanker":
        """Fit item scaler and pairwise classifier from prebuilt scaled diff rows."""
        self.fit_item_scaler(item_x)
        return self.fit_classifier(pair_x_diff=pair_x_diff, pair_y=pair_y)

    def score_items(self, item_x: Sequence[Sequence[float]]) -> np.ndarray:
        """Return item-level ranking scores: coef dot standardized features."""
        if self.coef_ is None:
            raise ValueError("PairwiseLogisticRanker classifier is not fitted.")
        x_scaled = self.transform_items(item_x)
        return x_scaled @ self.coef_

    def predict_pairwise(self, pair_x_diff: Sequence[Sequence[float]]) -> np.ndarray:
        """Predict pairwise preference labels for diff rows."""
        if self.classifier is None:
            raise ValueError("PairwiseLogisticRanker classifier is not fitted.")
        x_diff = _as_2d_float_array(pair_x_diff, expected_width=len(self.feature_names))
        return self.classifier.predict(x_diff)

    def predict_pairwise_scores(self, pair_x_diff: Sequence[Sequence[float]]) -> np.ndarray:
        """Return positive-class pairwise probabilities for diff rows."""
        if self.classifier is None:
            raise ValueError("PairwiseLogisticRanker classifier is not fitted.")
        x_diff = _as_2d_float_array(pair_x_diff, expected_width=len(self.feature_names))
        return self.classifier.predict_proba(x_diff)[:, 1]


def feature_dicts_to_matrix(
    feature_dicts: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Convert feature dictionaries into a matrix using stable feature order."""
    rows: list[list[float]] = []
    for row_index, feature_dict in enumerate(feature_dicts):
        row: list[float] = []
        for feature_name in feature_names:
            if feature_name not in feature_dict:
                raise ValueError(f"feature row {row_index} missing feature {feature_name}")
            try:
                row.append(float(feature_dict[feature_name]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"feature row {row_index} feature {feature_name} must be numeric.",
                ) from exc
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _as_2d_float_array(
    values: Sequence[Sequence[float]] | np.ndarray,
    expected_width: int,
) -> np.ndarray:
    """Convert input to a two-dimensional float array and validate width."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional.")
    if array.shape[1] != expected_width:
        raise ValueError(
            f"feature matrix width mismatch: expected {expected_width}, got {array.shape[1]}",
        )
    return array
