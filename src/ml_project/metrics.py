"""Central metric definitions with a single, tested class order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from .config import CLASS_ORDER


def _as_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"Expected an (n, {len(CLASS_ORDER)}) probability matrix")
    if not np.isfinite(p).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if (p < -1e-12).any():
        raise ValueError("Probabilities cannot be negative")
    row_sums = p.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Every probability row must have positive mass")
    return p / row_sums


def labels_to_indices(labels: Iterable[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(CLASS_ORDER)}
    values = np.asarray(list(labels))
    unknown = set(np.unique(values)) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown outcome labels: {sorted(unknown)}")
    return np.asarray([mapping[value] for value in values], dtype=int)


def ranked_probability_score(
    y_true: Sequence[str], probabilities: np.ndarray
) -> float:
    """Mean multiclass RPS in ordered classes A, D, H.

    The score is normalized by ``K - 1`` and therefore lies in [0, 1].
    Reversing the full order to H, D, A gives the same scalar, but fixing the
    order avoids accidental column permutations elsewhere.
    """

    p = _as_probability_matrix(probabilities)
    idx = labels_to_indices(y_true)
    observed = np.eye(len(CLASS_ORDER), dtype=float)[idx]
    return float(
        np.mean(
            np.sum(
                (np.cumsum(p, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1])
                ** 2,
                axis=1,
            )
            / (len(CLASS_ORDER) - 1)
        )
    )


def multiclass_brier_score(
    y_true: Sequence[str], probabilities: np.ndarray
) -> float:
    """Mean sum-of-squares multiclass Brier score (range 0--2)."""

    p = _as_probability_matrix(probabilities)
    observed = np.eye(len(CLASS_ORDER))[labels_to_indices(y_true)]
    return float(np.mean(np.sum((p - observed) ** 2, axis=1)))


def expected_calibration_error(
    y_true: Sequence[str], probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label equal-width expected calibration error."""

    p = _as_probability_matrix(probabilities)
    y = labels_to_indices(y_true)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        right_closed = i == n_bins - 1
        mask = (confidence >= edges[i]) & (
            (confidence <= edges[i + 1]) if right_closed else (confidence < edges[i + 1])
        )
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def reliability_table(
    y_true: Sequence[str], probabilities: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Return top-label and one-vs-rest reliability bins."""

    p = _as_probability_matrix(probabilities)
    y = labels_to_indices(y_true)
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    series = [("top_label", p.max(axis=1), p.argmax(axis=1) == y)]
    series.extend(
        (label, p[:, j], y == j) for j, label in enumerate(CLASS_ORDER)
    )
    for label, confidence, observed in series:
        for i in range(n_bins):
            mask = (confidence >= edges[i]) & (
                (confidence <= edges[i + 1]) if i == n_bins - 1 else (confidence < edges[i + 1])
            )
            rows.append(
                {
                    "series": label,
                    "bin": i,
                    "lower": edges[i],
                    "upper": edges[i + 1],
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                    "observed_frequency": float(observed[mask].mean()) if mask.any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: Sequence[str], probabilities: np.ndarray
) -> dict[str, float]:
    p = _as_probability_matrix(probabilities)
    idx = labels_to_indices(y_true)
    return {
        "rps": ranked_probability_score(y_true, p),
        "log_loss": float(log_loss(idx, p, labels=np.arange(len(CLASS_ORDER)))),
        "brier": multiclass_brier_score(y_true, p),
        "ece": expected_calibration_error(y_true, p),
        "accuracy": float(accuracy_score(idx, p.argmax(axis=1))),
    }


def regression_metrics(y_true: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if y.shape != pred.shape:
        raise ValueError("Targets and predictions must have identical shape")
    correlation = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else np.nan
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "correlation": correlation,
    }


@dataclass
class MarginProbabilityMapper:
    """Map a scalar margin forecast to H/D/A probabilities.

    A multinomial logistic model is fitted on calibration data using only the
    out-of-sample margin prediction. This avoids arbitrary rounding thresholds.
    """

    random_state: int = 42

    def fit(self, margin_prediction: Sequence[float], y: Sequence[str]):
        from sklearn.linear_model import LogisticRegression

        self.model_ = LogisticRegression(max_iter=2000, random_state=self.random_state)
        self.model_.fit(np.asarray(margin_prediction).reshape(-1, 1), np.asarray(y))
        return self

    def predict_proba(self, margin_prediction: Sequence[float]) -> np.ndarray:
        raw = self.model_.predict_proba(np.asarray(margin_prediction).reshape(-1, 1))
        aligned = np.zeros((len(raw), len(CLASS_ORDER)), dtype=float)
        for source, label in enumerate(self.model_.classes_):
            aligned[:, CLASS_ORDER.index(label)] = raw[:, source]
        return aligned


def logits_to_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(raw_scores, dtype=float), axis=1)
