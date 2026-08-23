"""Load and serve frozen production artifacts without retraining."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.config import APISettings
from api.errors import (
    InferenceUnavailableError,
    InvalidRequestError,
    MatchNotFoundError,
    SnapshotNotFoundError,
)


PUBLIC_CLASS_ORDER = ("away_win", "draw", "home_win")
ARTIFACT_VERSION = "final-lock-2026-08-20"


def _relative_artifact(filename: str) -> str:
    return f"models/reproducible/final/{filename}"


def _ordered_probabilities(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    """Align an estimator's numeric class labels to the frozen A/D/H order."""

    raw = np.asarray(estimator.predict_proba(frame), dtype=float)
    classes = np.asarray(estimator.classes_)
    aligned = np.zeros((len(raw), 3), dtype=float)
    for source, class_index in enumerate(classes):
        aligned[:, int(class_index)] = raw[:, source]
    aligned = np.clip(aligned, 0.0, 1.0)
    return aligned / aligned.sum(axis=1, keepdims=True)


def _probabilities(values: np.ndarray) -> dict[str, float]:
    away, draw, home = (float(value) for value in np.asarray(values, dtype=float))
    return {"home_win": home, "draw": draw, "away_win": away}


def _predicted_class(values: np.ndarray) -> str:
    return PUBLIC_CLASS_ORDER[int(np.asarray(values).argmax())]


class ModelRegistry:
    """Application-lifetime cache for models, manifests, and feature stores."""

    _model_files = {
        "prematch_classifier": "prematch_classifier_random_forest.joblib",
        "prematch_calibrator": "prematch_calibrator_random_forest_platt.joblib",
        "prematch_regressor": "prematch_regressor_random_forest.joblib",
        "prematch_margin_mapper": "prematch_margin_mapper_random_forest.joblib",
        "snapshot_classifier": "snapshot_classifier_gradient_boosting.joblib",
        "snapshot_calibrator": "snapshot_phase_calibrator_gradient_boosting_platt.joblib",
        "snapshot_regressor": "snapshot_regressor_xgboost.joblib",
        "snapshot_margin_mapper": "snapshot_margin_mapper_xgboost.joblib",
    }

    def __init__(self, settings: APISettings):
        self.settings = settings
        self.ready = False
        self._load_error: str | None = None
        self.models: dict[str, Any] = {}

    def load(self) -> None:
        """Load all inference resources once. Fail closed with a safe status."""

        try:
            self.feature_manifest = json.loads(
                self.settings.feature_manifest_path.read_text(encoding="utf-8")
            )
            self.snapshot_manifest = json.loads(
                self.settings.snapshot_manifest_path.read_text(encoding="utf-8")
            )
            self.prematch_columns = self.feature_manifest["groups"]["combined"]
            self.snapshot_columns = self.snapshot_manifest["groups"]["combined"]
            self.snapshot_minutes = tuple(self.snapshot_manifest["snapshot_minutes"])

            prematch = pd.read_parquet(self.settings.prematch_features_path)
            snapshots = pd.read_parquet(self.settings.snapshot_features_path)
            if prematch["match_id"].duplicated().any():
                raise ValueError("Duplicate match feature rows")
            if snapshots.duplicated(["match_id", "snapshot_minute"]).any():
                raise ValueError("Duplicate snapshot feature rows")
            if set(self.prematch_columns) - set(prematch.columns):
                raise ValueError("Prematch feature schema mismatch")
            if set(self.snapshot_columns) - set(snapshots.columns):
                raise ValueError("Snapshot feature schema mismatch")

            self.prematch = prematch.sort_values(["clean_date", "match_id"]).reset_index(drop=True)
            self.snapshots = snapshots.sort_values(
                ["clean_date", "match_id", "snapshot_minute"]
            ).reset_index(drop=True)
            self._prematch_index = self.prematch.set_index("match_id")
            if not self._prematch_index.index.is_unique:
                raise ValueError("Duplicate match feature rows")
            for key, filename in self._model_files.items():
                self.models[key] = joblib.load(self.settings.model_dir / filename)

            expected_feature_counts = {
                "prematch_classifier": len(self.prematch_columns),
                "prematch_regressor": len(self.prematch_columns),
                "snapshot_classifier": len(self.snapshot_columns),
                "snapshot_regressor": len(self.snapshot_columns),
            }
            for key, expected in expected_feature_counts.items():
                actual = len(getattr(self.models[key], "feature_names_in_", ()))
                if actual != expected:
                    raise ValueError("Frozen model feature schema mismatch")

            self.ready = True
            self._load_error = None
        except Exception:
            self.ready = False
            self._load_error = "Required frozen inference resources are unavailable."

    @property
    def safe_load_error(self) -> str | None:
        return self._load_error

    def _require_ready(self) -> None:
        if not self.ready:
            raise InferenceUnavailableError(
                self._load_error or "Required frozen inference resources are unavailable."
            )

    def model_metadata(self) -> list[dict[str, object]]:
        definitions = [
            {
                "name": "random_forest",
                "task": "prematch_outcome",
                "feature_set": "combined",
                "n_features": 391,
                "calibrated": True,
                "calibration": "multiclass_platt",
                "artifact": _relative_artifact(self._model_files["prematch_classifier"]),
            },
            {
                "name": "random_forest",
                "task": "prematch_margin",
                "feature_set": "combined",
                "n_features": 391,
                "calibrated": False,
                "calibration": None,
                "artifact": _relative_artifact(self._model_files["prematch_regressor"]),
            },
            {
                "name": "gradient_boosting",
                "task": "inplay_outcome",
                "feature_set": "combined_snapshot",
                "n_features": 507,
                "calibrated": False,
                "calibration": "phase_platt_available_as_alternative",
                "artifact": _relative_artifact(self._model_files["snapshot_classifier"]),
            },
            {
                "name": "xgboost",
                "task": "inplay_margin",
                "feature_set": "combined_snapshot",
                "n_features": 507,
                "calibrated": False,
                "calibration": None,
                "artifact": _relative_artifact(self._model_files["snapshot_regressor"]),
            },
        ]
        return [{**item, "version": ARTIFACT_VERSION, "available": self.ready} for item in definitions]

    @staticmethod
    def _match_summary(row: pd.Series) -> dict[str, object]:
        return {
            "match_id": int(row["match_id"]),
            "competition": "La Liga",
            "season": str(row["season_name"]),
            "match_date": pd.Timestamp(row["clean_date"]).date(),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "split": str(row["split_role"]),
        }

    def list_matches(
        self,
        *,
        competition: str | None,
        season: str | None,
        team: str | None,
        date_from: date | None,
        date_to: date | None,
        split: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, object]:
        self._require_ready()
        if date_from and date_to and date_from > date_to:
            raise InvalidRequestError(
                "invalid_date_range",
                "date_from must be on or before date_to.",
                {"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
            )
        frame = self.prematch
        mask = pd.Series(True, index=frame.index)
        if competition and competition.casefold() not in {"la liga", "laliga", "spain la liga"}:
            mask &= False
        if season:
            mask &= frame["season_name"].astype(str).str.casefold().eq(season.casefold())
        if team:
            term = team.casefold()
            mask &= frame["home_team"].str.casefold().str.contains(term, regex=False) | frame[
                "away_team"
            ].str.casefold().str.contains(term, regex=False)
        if date_from:
            mask &= pd.to_datetime(frame["clean_date"]).dt.date.ge(date_from)
        if date_to:
            mask &= pd.to_datetime(frame["clean_date"]).dt.date.le(date_to)
        if split:
            mask &= frame["split_role"].eq(split)
        selected = frame.loc[mask]
        page = selected.iloc[offset : offset + limit]
        return {
            "total": int(len(selected)),
            "limit": limit,
            "offset": offset,
            "items": [self._match_summary(row) for _, row in page.iterrows()],
        }

    def match_detail(self, match_id: int) -> dict[str, object]:
        row = self._match_row(match_id)
        result = self._match_summary(row)
        available = self.snapshots["match_id"].eq(match_id).any()
        result.update(
            {
                "prematch_prediction_available": True,
                "inplay_prediction_available": bool(available),
                "supported_snapshot_minutes": list(self.snapshot_minutes) if available else [],
            }
        )
        return result

    def _match_row(self, match_id: int) -> pd.Series:
        self._require_ready()
        try:
            row = self._prematch_index.loc[match_id].copy()
            row["match_id"] = match_id
            return row
        except KeyError as error:
            raise MatchNotFoundError(match_id) from error

    @staticmethod
    def _frame(row_or_frame: pd.Series | pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if isinstance(row_or_frame, pd.Series):
            frame = row_or_frame.loc[columns].to_frame().T
        else:
            frame = row_or_frame.loc[:, columns]
        return frame.astype(float)

    def predict_prematch_outcome(self, match_id: int) -> dict[str, object]:
        row = self._match_row(match_id)
        frame = self._frame(row, self.prematch_columns)
        raw = _ordered_probabilities(self.models["prematch_classifier"], frame)[0]
        calibrated = self.models["prematch_calibrator"].predict_proba(raw.reshape(1, -1))[0]
        return {
            "match_id": match_id,
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "task": "prematch_outcome",
            "probabilities": _probabilities(calibrated),
            "raw_probabilities": _probabilities(raw),
            "predicted_class": _predicted_class(calibrated),
            "model": "random_forest",
            "calibrated": True,
            "calibration": "multiclass_platt",
            "class_order": list(PUBLIC_CLASS_ORDER),
        }

    def predict_prematch_margin(self, match_id: int) -> dict[str, object]:
        row = self._match_row(match_id)
        frame = self._frame(row, self.prematch_columns)
        margin = float(np.clip(self.models["prematch_regressor"].predict(frame)[0], -5.0, 5.0))
        derived = self.models["prematch_margin_mapper"].predict_proba([margin])[0]
        return {
            "match_id": match_id,
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "task": "prematch_margin",
            "expected_goal_margin": margin,
            "derived_outcome_probabilities": _probabilities(derived),
            "model": "random_forest",
            "calibrated": False,
            "probability_mapping": "calibration_block_multinomial_logistic",
        }

    def predict_prematch(self, match_id: int) -> dict[str, object]:
        row = self._match_row(match_id)
        return {
            "match_id": match_id,
            "match_date": pd.Timestamp(row["clean_date"]).date(),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "split": str(row["split_role"]),
            "outcome": self.predict_prematch_outcome(match_id),
            "margin": self.predict_prematch_margin(match_id),
        }

    def _snapshot_rows(self, match_id: int) -> pd.DataFrame:
        self._match_row(match_id)
        rows = self.snapshots[self.snapshots["match_id"].eq(match_id)].sort_values("snapshot_minute")
        if rows.empty:
            raise SnapshotNotFoundError(match_id, 0)
        return rows

    def _predict_snapshot_frame(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        frame = self._frame(rows, self.snapshot_columns)
        raw = _ordered_probabilities(self.models["snapshot_classifier"], frame)
        minutes = rows["snapshot_minute"].to_numpy(dtype=int)
        phase_calibrator = self.models["snapshot_calibrator"]
        phases = phase_calibrator.phase(minutes)
        calibrated = np.zeros_like(raw)
        for phase, calibrator in phase_calibrator.calibrators_.items():
            mask = phases == phase
            if mask.any():
                calibrated[mask] = calibrator.predict_proba(raw[mask])
        margin = np.clip(self.models["snapshot_regressor"].predict(frame), -5.0, 5.0)
        derived = self.models["snapshot_margin_mapper"].predict_proba(margin)
        return {"raw": raw, "calibrated": calibrated, "margin": margin, "derived": derived}

    def predict_inplay(self, match_id: int, minute: int) -> dict[str, object]:
        rows = self._snapshot_rows(match_id)
        selected = rows[rows["snapshot_minute"].eq(minute)]
        if selected.empty:
            raise SnapshotNotFoundError(match_id, minute)
        row = selected.iloc[0]
        forecast = self._predict_snapshot_frame(selected)
        raw = forecast["raw"][0]
        calibrated = forecast["calibrated"][0]
        return {
            "match_id": match_id,
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "task": "inplay",
            "snapshot": {
                "minute": minute,
                "second": 0,
                "boundary": "event_seconds_exact <= snapshot_seconds",
            },
            "score": {"home": int(row["home_score"]), "away": int(row["away_score"])},
            "probabilities": _probabilities(raw),
            "calibrated_probabilities": _probabilities(calibrated),
            "predicted_class": _predicted_class(raw),
            "expected_final_margin": float(forecast["margin"][0]),
            "derived_outcome_probabilities": _probabilities(forecast["derived"][0]),
            "models": {"outcome": "gradient_boosting", "margin": "xgboost"},
            "calibrated": False,
            "calibration_alternative": "phase_platt",
        }

    def timeline(self, match_id: int) -> dict[str, object]:
        rows = self._snapshot_rows(match_id)
        forecast = self._predict_snapshot_frame(rows)
        points = []
        for index, (_, row) in enumerate(rows.iterrows()):
            points.append(
                {
                    "minute": int(row["snapshot_minute"]),
                    "score": {"home": int(row["home_score"]), "away": int(row["away_score"])},
                    "probabilities": _probabilities(forecast["raw"][index]),
                    "calibrated_probabilities": _probabilities(forecast["calibrated"][index]),
                    "expected_final_margin": float(forecast["margin"][index]),
                }
            )
        first = rows.iloc[0]
        return {
            "match_id": match_id,
            "home_team": str(first["home_team"]),
            "away_team": str(first["away_team"]),
            "historical_replay": True,
            "boundary": "event_seconds_exact <= snapshot_seconds",
            "points": points,
        }
