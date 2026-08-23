# API quickstart

## Install and run

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn api.main:app --reload
```

To honor `API_HOST` and `API_PORT` directly, use `python -m api` instead.

The editable install makes the serialized `ml_project` calibrator and mapper classes importable. Startup loads the frozen artifacts and feature stores once.

Open:

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>

## Check readiness

```bash
curl "http://127.0.0.1:8000/api/v1/health"
```

Expected response:

```json
{
  "status": "ok",
  "service": "forecasting-competitive-football",
  "api_version": "v1",
  "models_loaded": true
}
```

If a required artifact or feature store is unavailable, the process stays observable with `status: "degraded"`; inference calls return a safe `503` without a local path or stack trace.

## Find a known match

```bash
curl "http://127.0.0.1:8000/api/v1/matches?season=2016%2F2017&team=Barcelona&split=final_test&limit=5"
```

The examples use public final-cohort match ID `267670` (Barcelona vs Real Betis, 2016-08-20).

## Combined pre-match forecast

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/prematch" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670}'
```

## In-play snapshot

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/inplay" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670,"minute":60,"second":0}'
```

At that stored historical snapshot the score is Barcelona 5–1 Real Betis. The endpoint uses only features whose event source time is at or before 60:00.

## Configuration

Copy `.env.example` values into your process environment or container configuration. Do not commit an actual `.env`. Important settings include model/feature paths, allowed browser origins, and `PUBLIC_API_BASE_URL`.

For every route and error, continue to [Endpoints](endpoints.md) and [Error Handling](errors.md).
