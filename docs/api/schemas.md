# Request and response schemas

The canonical schemas live in the generated [OpenAPI document](openapi.json). This page highlights validation and interpretation rules.

## Request schemas

### `PrematchRequest`

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `match_id` | integer | `> 0` | Known StatsBomb ID in the frozen store |

### `InPlayRequest`

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `match_id` | integer | `> 0` | Known StatsBomb ID |
| `minute` | integer | `0 <= minute <= 90`, multiple of 5 | Frozen snapshot minute |
| `second` | integer literal | exactly `0` | Minute-aligned public grid |

Unknown request fields are rejected. The API deliberately does not accept arbitrary feature vectors, file paths, or user-provided serialized objects.

## Probability schema

```json
{
  "home_win": 0.47,
  "draw": 0.28,
  "away_win": 0.25
}
```

Every field is between zero and one; rows sum to approximately one. Although internal training order is A–D–H, public names eliminate ordering ambiguity.

## Expected margin

`expected_goal_margin` and `expected_final_margin` are floating-point values in `[-5, 5]`:

```text
positive → expected home advantage
zero     → even expected result
negative → expected away advantage
```

The value is an expectation under the fitted regressor, not a guaranteed or necessarily integer final score.

## Match splits

The `split` field is one of:

- `train`
- `validation`
- `calibration`
- `pilot_holdout`
- `final_test`

It describes the historical scientific protocol. Calling the API never changes a match’s role.

## Calibration fields

- Model 1: `probabilities` is calibrated, `raw_probabilities` is retained, `calibrated: true`.
- Model 2: the scalar is uncalibrated; `derived_outcome_probabilities` comes from the fitted margin mapper.
- Model 3: `probabilities` is the raw headline and `calibrated_probabilities` is the phase-Platt alternative; `calibrated: false` refers to the headline field.

## Error schema

```json
{
  "error": {
    "code": "match_not_found",
    "message": "Match 1 was not found in the frozen feature store.",
    "details": {"match_id": 1}
  }
}
```

Validation errors use FastAPI’s standard `422` detail list. Server-side errors never include a stack trace or local path.
