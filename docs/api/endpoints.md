# Endpoints

Every route is versioned under `/api/v1`. Request bodies reject unknown fields. Response models use named probability fields and never expose absolute paths.

## Service health

| Detail | Value |
|---|---|
| Purpose | Check application and frozen-resource readiness |
| Method/path | `GET /api/v1/health` |
| Input | None |
| Output | Status, service name, API version, model-load flag |
| Errors | None for ordinary degraded state |
| Model | None |
| Calibration | Not applicable |

## Model metadata

| Detail | Value |
|---|---|
| Purpose | Enumerate public headline predictors and safe artifact identifiers |
| Method/path | `GET /api/v1/models` |
| Input | None |
| Output | Class order and model metadata list |
| Errors | None; unavailable artifacts are marked `available: false` |
| Notes | Repository-relative artifact names only |

## Match lookup

| Detail | Value |
|---|---|
| Purpose | Search the bounded 414-match catalogue |
| Method/path | `GET /api/v1/matches` |
| Query | `competition`, `season`, `team`, `date_from`, `date_to`, `split`, `limit`, `offset` |
| Output | `total`, page parameters, safe match summaries |
| Errors | `400` reversed date range; `422` invalid query; `503` store unavailable |
| Limits | `1 <= limit <= 100`; default 20 |

Team matching is case-insensitive substring search over home and away names. This release contains La Liga only.

## Match metadata

| Detail | Value |
|---|---|
| Purpose | Describe one known match and available inference modes |
| Method/path | `GET /api/v1/matches/{match_id}` |
| Input | Positive integer path ID |
| Output | Teams, date, season, split, and supported snapshot minutes |
| Errors | `404` unknown ID; `422` invalid ID type; `503` store unavailable |

## Model 1: pre-match outcome

| Detail | Value |
|---|---|
| Purpose | Predict the final A/D/H distribution at kickoff |
| Method/path | `POST /api/v1/predict/prematch/outcome` |
| Input | `{"match_id": 267670}` |
| Output | Calibrated and raw probabilities, class, model, calibration metadata |
| Errors | `404`, `422`, `503` |
| Model | Random forest |
| Calibration | Multiclass Platt on the disjoint calibration block |
| Information available | Frozen leakage-safe pre-match histories and market context only |

Example response from the saved match:

```json
{
  "match_id": 267670,
  "home_team": "Barcelona",
  "away_team": "Real Betis",
  "task": "prematch_outcome",
  "probabilities": {"home_win": 0.768509, "draw": 0.135443, "away_win": 0.096049},
  "raw_probabilities": {"home_win": 0.757813, "draw": 0.155750, "away_win": 0.086436},
  "predicted_class": "home_win",
  "model": "random_forest",
  "calibrated": true,
  "calibration": "multiclass_platt",
  "class_order": ["away_win", "draw", "home_win"]
}
```

## Model 2: pre-match margin

| Detail | Value |
|---|---|
| Purpose | Predict clipped final `home goals − away goals` |
| Method/path | `POST /api/v1/predict/prematch/margin` |
| Input | `{"match_id": 267670}` |
| Output | Expected margin and probabilities derived from the scalar forecast |
| Errors | `404`, `422`, `503` |
| Model | Random forest |
| Calibration | No direct margin calibration; probability mapper fitted on calibration block |
| Limitation | Derived probabilities are not direct Model 1 output |

For match `267670`, the saved expected margin is `+3.186` home goals after clipping to `[-5, 5]`.

## Combined pre-match

| Detail | Value |
|---|---|
| Purpose | Retrieve Models 1 and 2 in one call |
| Method/path | `POST /api/v1/predict/prematch` |
| Input | `{"match_id": 267670}` |
| Output | Match metadata plus nested `outcome` and `margin` objects |
| Errors | `404`, `422`, `503` |

## Model 3: in-play snapshot

| Detail | Value |
|---|---|
| Purpose | Update final outcome and margin from observed match state |
| Method/path | `POST /api/v1/predict/inplay` |
| Input | `match_id`, five-minute `minute`, `second: 0` |
| Output | Snapshot/score, raw and phase-Platt probabilities, expected margin, models |
| Errors | `404`, `422`, `503` |
| Models | Gradient boosting outcome; XGBoost margin |
| Calibration | Raw headline plus phase-Platt alternative |
| Boundary | `event_seconds_exact <= snapshot_seconds` |

Example response fields at minute 60:

```json
{
  "match_id": 267670,
  "home_team": "Barcelona",
  "away_team": "Real Betis",
  "task": "inplay",
  "snapshot": {"minute": 60, "second": 0, "boundary": "event_seconds_exact <= snapshot_seconds"},
  "score": {"home": 5, "away": 1},
  "probabilities": {"home_win": 0.866818, "draw": 0.097109, "away_win": 0.036072},
  "calibrated_probabilities": {"home_win": 0.920880, "draw": 0.074259, "away_win": 0.004861},
  "predicted_class": "home_win",
  "expected_final_margin": 4.306898,
  "derived_outcome_probabilities": {"home_win": 0.994755, "draw": 0.005127, "away_win": 0.000118},
  "models": {"outcome": "gradient_boosting", "margin": "xgboost"},
  "calibrated": false,
  "calibration_alternative": "phase_platt"
}
```

The probability values above are rounded from the saved fitted outputs; the live response contains full floating-point precision.

## Historical timeline

| Detail | Value |
|---|---|
| Purpose | Replay every supported Model 3 point for a known match |
| Method/path | `GET /api/v1/matches/{match_id}/timeline` |
| Input | Positive integer match ID |
| Output | Match metadata and 19 points with score, probabilities, and expected margin |
| Errors | `404`, `422`, `503` |
| Notes | `historical_replay: true`; not a live event feed |

The route batches model calls across the 19 stored rows, making it suitable for a frontend probability chart.

## Explanation availability

There is no `/explain` endpoint in v1. The [Explainability](../models/explainability.md) page documents why the offline SHAP evidence is not yet a correct public request-time contract.
