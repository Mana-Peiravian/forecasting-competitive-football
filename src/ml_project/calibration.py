"""Holdout-only multiclass Platt and isotonic probability calibration."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class MulticlassProbabilityCalibrator:
    def __init__(self, method: str = "platt", random_state: int = 42):
        if method not in {"platt", "isotonic"}:
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method
        self.random_state = random_state

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
        return p / p.sum(axis=1, keepdims=True)

    def fit(self, probabilities: np.ndarray, y: np.ndarray):
        p = self._normalize(probabilities)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = p.shape[1]
        if self.method == "platt":
            # Multinomial logistic scaling of log probabilities is the natural
            # multiclass extension of Platt's logistic score calibration.
            self.model_ = LogisticRegression(max_iter=3000, random_state=self.random_state)
            self.model_.fit(np.log(p), y)
        else:
            self.models_ = []
            for class_index in range(self.n_classes_):
                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(p[:, class_index], (y == class_index).astype(float))
                self.models_.append(model)
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        p = self._normalize(probabilities)
        if self.method == "platt":
            raw = self.model_.predict_proba(np.log(p))
            aligned = np.zeros((len(raw), self.n_classes_), dtype=float)
            for source, class_index in enumerate(self.model_.classes_):
                aligned[:, int(class_index)] = raw[:, source]
            return self._normalize(aligned)
        raw = np.column_stack(
            [model.predict(p[:, class_index]) for class_index, model in enumerate(self.models_)]
        )
        zero = raw.sum(axis=1) <= 1e-15
        raw[zero] = p[zero]
        return self._normalize(raw)


class PhaseCalibrator:
    """Separate calibrators for early, middle, and late match snapshots."""

    def __init__(self, method: str = "platt", random_state: int = 42):
        self.method = method
        self.random_state = random_state

    @staticmethod
    def phase(minutes: np.ndarray) -> np.ndarray:
        minute = np.asarray(minutes)
        return np.where(minute <= 30, "early", np.where(minute <= 60, "middle", "late"))

    def fit(self, probabilities: np.ndarray, y: np.ndarray, minutes: np.ndarray):
        phases = self.phase(minutes)
        self.calibrators_ = {}
        for phase in ("early", "middle", "late"):
            mask = phases == phase
            self.calibrators_[phase] = MulticlassProbabilityCalibrator(
                self.method, self.random_state
            ).fit(probabilities[mask], y[mask])
        return self

    def predict_proba(self, probabilities: np.ndarray, minutes: np.ndarray) -> np.ndarray:
        phases = self.phase(minutes)
        output = np.zeros_like(np.asarray(probabilities, dtype=float))
        for phase, calibrator in self.calibrators_.items():
            mask = phases == phase
            output[mask] = calibrator.predict_proba(np.asarray(probabilities)[mask])
        return output

