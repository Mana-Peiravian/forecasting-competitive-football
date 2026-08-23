# Error handling

The API separates schema validation, missing public resources, and server readiness. Errors do not expose stack traces, environment variables, or local filesystem paths.

## Status codes

| Status | Code / source | When it occurs | Caller action |
|---:|---|---|---|
| 400 | `invalid_date_range` | `date_from` is later than `date_to` | Correct the filter range |
| 404 | `match_not_found` | Positive ID is not in the frozen match store | Search `/matches` and retry with a returned ID |
| 404 | `snapshot_not_found` | Match has no requested stored point | Use the match’s `supported_snapshot_minutes` |
| 422 | FastAPI validation | Non-positive ID, bad type, minute outside 0–90/not divisible by five, nonzero second, unknown field | Correct the request body/query |
| 503 | `inference_unavailable` | A required model, manifest, or feature store did not load | Check deployment mounts/paths and health |
| 500 | `internal_error` | Unexpected server failure | Inspect private server logs; do not retry indefinitely |

## Application error body

```json
{
  "error": {
    "code": "match_not_found",
    "message": "Match 1 was not found in the frozen feature store.",
    "details": {"match_id": 1}
  }
}
```

## Validation example

Request:

```json
{"match_id": 267670, "minute": 61, "second": 0}
```

Response: `422 Unprocessable Entity`, because `minute` must be a multiple of five. FastAPI returns a structured `detail` list identifying the field and constraint.

## Degraded startup

The registry attempts one startup load. If any required inference resource is missing or incompatible:

- `/api/v1/health` returns `status: "degraded"` and `models_loaded: false`;
- `/api/v1/models` retains safe metadata with `available: false`;
- match and prediction routes return `503`;
- the response does not disclose which local path failed.

Operators should validate container contents and configured paths through private deployment logs and filesystem checks.
