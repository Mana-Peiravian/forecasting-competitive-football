# API examples

All examples target the default local server and use public match ID `267670`. Start the service first with `uvicorn api.main:app --reload`.

## cURL

### Health and models

```bash
curl "http://127.0.0.1:8000/api/v1/health"
curl "http://127.0.0.1:8000/api/v1/models"
```

### Match lookup and metadata

```bash
curl "http://127.0.0.1:8000/api/v1/matches?season=2016%2F2017&team=Barcelona&limit=5"
curl "http://127.0.0.1:8000/api/v1/matches/267670"
```

### Model 1 outcome

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/prematch/outcome" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670}'
```

### Model 2 margin

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/prematch/margin" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670}'
```

### Combined pre-match

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/prematch" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670}'
```

### Model 3 snapshot and timeline

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/inplay" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670,"minute":60,"second":0}'

curl "http://127.0.0.1:8000/api/v1/matches/267670/timeline"
```

## Python with `requests`

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

health = requests.get(f"{BASE_URL}/api/v1/health", timeout=30)
health.raise_for_status()
print(health.json())

matches = requests.get(
    f"{BASE_URL}/api/v1/matches",
    params={"season": "2016/2017", "team": "Barcelona", "limit": 5},
    timeout=30,
)
matches.raise_for_status()
print(matches.json()["items"])

prematch = requests.post(
    f"{BASE_URL}/api/v1/predict/prematch",
    json={"match_id": 267670},
    timeout=30,
)
prematch.raise_for_status()
print(prematch.json()["outcome"]["probabilities"])

inplay = requests.post(
    f"{BASE_URL}/api/v1/predict/inplay",
    json={"match_id": 267670, "minute": 60, "second": 0},
    timeout=30,
)
inplay.raise_for_status()
print(inplay.json()["score"], inplay.json()["probabilities"])

timeline = requests.get(
    f"{BASE_URL}/api/v1/matches/267670/timeline",
    timeout=30,
)
timeline.raise_for_status()
print(len(timeline.json()["points"]))
```

The repository also includes `FootballForecastClient` in `api/client.py`; see [API Clients](../developer/clients.md).

## JavaScript `fetch`

```javascript
const baseUrl = "http://127.0.0.1:8000";

async function json(response) {
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

const health = await json(fetch(`${baseUrl}/api/v1/health`));
console.log(health);

const matches = await json(fetch(
  `${baseUrl}/api/v1/matches?season=2016%2F2017&team=Barcelona&limit=5`
));
console.log(matches.items);

const prematch = await json(fetch(`${baseUrl}/api/v1/predict/prematch`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ match_id: 267670 })
}));
console.log(prematch.outcome.probabilities);

const inplay = await json(fetch(`${baseUrl}/api/v1/predict/inplay`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ match_id: 267670, minute: 60, second: 0 })
}));
console.log(inplay.score, inplay.probabilities);

const timeline = await json(fetch(`${baseUrl}/api/v1/matches/267670/timeline`));
console.log(timeline.points);
```

When browser JavaScript runs from GitHub Pages, its origin must appear in `ALLOWED_ORIGINS`. The repository default includes the discovered project owner’s Pages origin; use environment configuration for any other deployment.

## Explanations

No request-time explanation route exists in API v1. Use the saved [SHAP and timeline evidence](../models/explainability.md); do not code against an endpoint that would misrepresent model support.
