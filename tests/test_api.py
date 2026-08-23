from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.config import APISettings
from api.errors import InferenceUnavailableError
from api.main import app
from api.model_registry import ModelRegistry


MATCH_ID = 267670


def probability_sum(payload):
    return sum(payload[key] for key in ("home_win", "draw", "away_win"))


def test_health_and_model_metadata_load_frozen_artifacts_once():
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "service": "forecasting-competitive-football",
            "api_version": "v1",
            "models_loaded": True,
        }
        models = client.get("/api/v1/models")
        assert models.status_code == 200
        assert len(models.json()["models"]) == 4
        assert all(item["available"] for item in models.json()["models"])


def test_match_lookup_is_filtered_and_paginated():
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/matches",
            params={"season": "2016/2017", "team": "Barcelona", "split": "final_test", "limit": 5},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 34
        assert len(payload["items"]) == 5
        assert all(item["split"] == "final_test" for item in payload["items"])

        detail = client.get(f"/api/v1/matches/{MATCH_ID}")
        assert detail.status_code == 200
        assert detail.json()["supported_snapshot_minutes"] == list(range(0, 91, 5))


def test_model1_and_model2_match_saved_end_to_end_predictions():
    expected_class = pd.read_csv("outputs/final_evaluation/prematch_classification_predictions.csv")
    expected_class = expected_class[
        expected_class.match_id.eq(MATCH_ID)
        & expected_class.model.eq("random_forest")
        & expected_class.method.eq("platt")
    ].iloc[0]
    expected_margin = pd.read_csv("outputs/final_evaluation/prematch_regression_predictions.csv")
    expected_margin = expected_margin[
        expected_margin.match_id.eq(MATCH_ID) & expected_margin.model.eq("random_forest")
    ].iloc[0]

    with TestClient(app) as client:
        outcome = client.post(
            "/api/v1/predict/prematch/outcome", json={"match_id": MATCH_ID}
        )
        margin = client.post(
            "/api/v1/predict/prematch/margin", json={"match_id": MATCH_ID}
        )
        combined = client.post("/api/v1/predict/prematch", json={"match_id": MATCH_ID})

    assert outcome.status_code == margin.status_code == combined.status_code == 200
    probabilities = outcome.json()["probabilities"]
    assert np.isclose(probability_sum(probabilities), 1.0)
    assert np.allclose(
        [probabilities["away_win"], probabilities["draw"], probabilities["home_win"]],
        [expected_class.p_A, expected_class.p_D, expected_class.p_H],
    )
    assert np.isclose(margin.json()["expected_goal_margin"], expected_margin.predicted_margin)
    assert np.isclose(probability_sum(margin.json()["derived_outcome_probabilities"]), 1.0)
    assert combined.json()["outcome"]["model"] == outcome.json()["model"]
    assert combined.json()["margin"]["model"] == margin.json()["model"]
    assert np.allclose(
        list(combined.json()["outcome"]["probabilities"].values()),
        list(outcome.json()["probabilities"].values()),
    )
    assert np.isclose(
        combined.json()["margin"]["expected_goal_margin"],
        margin.json()["expected_goal_margin"],
    )


def test_model3_and_timeline_match_saved_end_to_end_predictions():
    minute = 60
    expected_class = pd.read_parquet(
        "outputs/final_evaluation/snapshot_classification_predictions.parquet"
    )
    expected_class = expected_class[
        expected_class.match_id.eq(MATCH_ID)
        & expected_class.snapshot_minute.eq(minute)
        & expected_class.model.eq("gradient_boosting")
        & expected_class.method.eq("raw")
    ].iloc[0]
    expected_margin = pd.read_parquet(
        "outputs/final_evaluation/snapshot_regression_predictions.parquet"
    )
    expected_margin = expected_margin[
        expected_margin.match_id.eq(MATCH_ID)
        & expected_margin.snapshot_minute.eq(minute)
        & expected_margin.model.eq("xgboost")
    ].iloc[0]

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/inplay",
            json={"match_id": MATCH_ID, "minute": minute, "second": 0},
        )
        timeline = client.get(f"/api/v1/matches/{MATCH_ID}/timeline")

    assert response.status_code == 200
    payload = response.json()
    assert np.isclose(probability_sum(payload["probabilities"]), 1.0)
    assert np.isclose(probability_sum(payload["calibrated_probabilities"]), 1.0)
    assert np.allclose(
        [
            payload["probabilities"]["away_win"],
            payload["probabilities"]["draw"],
            payload["probabilities"]["home_win"],
        ],
        [expected_class.p_A, expected_class.p_D, expected_class.p_H],
    )
    assert np.isclose(payload["expected_final_margin"], expected_margin.predicted_margin)
    assert timeline.status_code == 200
    assert len(timeline.json()["points"]) == 19
    assert timeline.json()["historical_replay"] is True


def test_api_rejects_unknown_matches_and_invalid_snapshot_times():
    with TestClient(app) as client:
        missing = client.post("/api/v1/predict/prematch", json={"match_id": 1})
        bad_range = client.get(
            "/api/v1/matches", params={"date_from": "2017-01-02", "date_to": "2017-01-01"}
        )
        bad_minute = client.post(
            "/api/v1/predict/inplay",
            json={"match_id": MATCH_ID, "minute": 61, "second": 0},
        )
        bad_second = client.post(
            "/api/v1/predict/inplay",
            json={"match_id": MATCH_ID, "minute": 60, "second": 1},
        )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "match_not_found"
    assert bad_range.status_code == 400
    assert bad_range.json()["error"]["code"] == "invalid_date_range"
    assert bad_minute.status_code == 422
    assert bad_second.status_code == 422


def test_registry_fails_closed_without_exposing_private_paths(tmp_path):
    private_model_dir = tmp_path / "private-model-location"
    registry = ModelRegistry(
        replace(APISettings.from_environment(), model_dir=private_model_dir)
    )

    registry.load()

    assert registry.ready is False
    assert registry.safe_load_error == "Required frozen inference resources are unavailable."
    assert str(private_model_dir) not in registry.safe_load_error
    assert all(not model["available"] for model in registry.model_metadata())
    with pytest.raises(InferenceUnavailableError, match="frozen inference resources"):
        registry.predict_prematch(MATCH_ID)
