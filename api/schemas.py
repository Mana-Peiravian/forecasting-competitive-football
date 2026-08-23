"""Typed public request and response contracts for API v1."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


OutcomeLabel = Literal["away_win", "draw", "home_win"]
SplitRole = Literal["train", "validation", "calibration", "pilot_holdout", "final_test"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(StrictModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Safe human-readable message.")
    details: dict[str, object] | None = Field(default=None, description="Optional safe context.")


class ErrorResponse(StrictModel):
    error: ErrorDetail


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    service: Literal["forecasting-competitive-football"]
    api_version: Literal["v1"]
    models_loaded: bool

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "forecasting-competitive-football",
                "api_version": "v1",
                "models_loaded": True,
            }
        },
    )


class ModelMetadata(StrictModel):
    name: str
    task: str
    version: str
    feature_set: str
    n_features: int = Field(gt=0)
    calibrated: bool
    calibration: str | None = None
    artifact: str = Field(description="Repository-relative artifact identifier, never a local path.")
    available: bool


class ModelsResponse(StrictModel):
    class_order: list[Literal["away_win", "draw", "home_win"]]
    models: list[ModelMetadata]


class MatchSummary(StrictModel):
    match_id: int = Field(gt=0)
    competition: Literal["La Liga"]
    season: str
    match_date: date
    home_team: str
    away_team: str
    split: SplitRole


class MatchesResponse(StrictModel):
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    items: list[MatchSummary]


class MatchDetail(MatchSummary):
    prematch_prediction_available: bool
    inplay_prediction_available: bool
    supported_snapshot_minutes: list[int]


class PrematchRequest(StrictModel):
    match_id: int = Field(
        gt=0,
        description="StatsBomb match identifier present in the frozen feature store.",
        examples=[267670],
    )


class Probabilities(StrictModel):
    home_win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away_win: float = Field(ge=0.0, le=1.0)


class PrematchOutcomeResponse(StrictModel):
    match_id: int
    home_team: str
    away_team: str
    task: Literal["prematch_outcome"]
    probabilities: Probabilities = Field(description="Platt-calibrated Model 1 probabilities.")
    raw_probabilities: Probabilities
    predicted_class: OutcomeLabel
    model: Literal["random_forest"]
    calibrated: Literal[True]
    calibration: Literal["multiclass_platt"]
    class_order: list[Literal["away_win", "draw", "home_win"]]


class PrematchMarginResponse(StrictModel):
    match_id: int
    home_team: str
    away_team: str
    task: Literal["prematch_margin"]
    expected_goal_margin: float = Field(
        ge=-5.0,
        le=5.0,
        description="Expected clipped home-goals minus away-goals margin.",
    )
    derived_outcome_probabilities: Probabilities = Field(
        description="Probabilities derived by the separately fitted Model 2 margin mapper."
    )
    model: Literal["random_forest"]
    calibrated: Literal[False]
    probability_mapping: Literal["calibration_block_multinomial_logistic"]


class CombinedPrematchResponse(StrictModel):
    match_id: int
    match_date: date
    home_team: str
    away_team: str
    split: SplitRole
    outcome: PrematchOutcomeResponse
    margin: PrematchMarginResponse


class InPlayRequest(StrictModel):
    match_id: int = Field(gt=0, examples=[267670])
    minute: int = Field(
        ge=0,
        le=90,
        multiple_of=5,
        description="Frozen snapshot minute: one of 0, 5, ..., 90.",
        examples=[60],
    )
    second: Literal[0] = Field(
        default=0,
        description="The public frozen grid is minute-aligned; only zero is supported.",
    )


class Snapshot(StrictModel):
    minute: int = Field(ge=0, le=90)
    second: Literal[0]
    boundary: Literal["event_seconds_exact <= snapshot_seconds"]


class Score(StrictModel):
    home: int = Field(ge=0)
    away: int = Field(ge=0)


class InPlayModels(StrictModel):
    outcome: Literal["gradient_boosting"]
    margin: Literal["xgboost"]


class InPlayResponse(StrictModel):
    match_id: int
    home_team: str
    away_team: str
    task: Literal["inplay"]
    snapshot: Snapshot
    score: Score
    probabilities: Probabilities = Field(
        description="Raw gradient-boosting probabilities: the final headline Model 3 output."
    )
    calibrated_probabilities: Probabilities = Field(
        description="Phase-Platt alternative; lower final ECE but worse final RPS."
    )
    predicted_class: OutcomeLabel
    expected_final_margin: float = Field(ge=-5.0, le=5.0)
    derived_outcome_probabilities: Probabilities
    models: InPlayModels
    calibrated: Literal[False] = Field(
        description="False because `probabilities` is the raw headline output."
    )
    calibration_alternative: Literal["phase_platt"]


class TimelinePoint(StrictModel):
    minute: int
    score: Score
    probabilities: Probabilities
    calibrated_probabilities: Probabilities
    expected_final_margin: float


class TimelineResponse(StrictModel):
    match_id: int
    home_team: str
    away_team: str
    historical_replay: Literal[True]
    boundary: Literal["event_seconds_exact <= snapshot_seconds"]
    points: list[TimelinePoint]
