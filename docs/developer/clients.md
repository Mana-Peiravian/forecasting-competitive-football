# API clients

## Included Python client

`api/client.py` wraps the small v1 surface using `requests`:

```python
from api.client import FootballForecastClient

client = FootballForecastClient("http://127.0.0.1:8000")

print(client.health())
print(client.matches(season="2016/2017", team="Barcelona", limit=5))
print(client.predict_prematch(267670))
print(client.predict_inplay(267670, minute=60))
print(client.timeline(267670))
```

The client calls `raise_for_status()` and returns decoded JSON. Set a different base URL only after deploying FastAPI; the GitHub Pages URL is not an API URL.

## Direct Python integration

For applications that already use `requests`, the [copy-paste examples](../api/examples.md#python-with-requests) may be preferable. Keep a finite timeout and handle `404`, `422`, and `503` distinctly.

## Browser/Node integration

Use `fetch` with JSON bodies. If a browser page and API are on different origins, configure the page’s exact origin in `ALLOWED_ORIGINS`. Origins contain scheme, host, and optional port—not the repository path.

```javascript
const response = await fetch("http://127.0.0.1:8000/api/v1/predict/inplay", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ match_id: 267670, minute: 60, second: 0 })
});

if (!response.ok) throw new Error(await response.text());
const forecast = await response.json();
```

## Static replay client

The Pages demo intentionally fetches `assets/data/demo-matches.json`, not FastAPI. This lets every visitor inspect the historical final predictions without a public Python server. The JSON declares `historical_backtest: true` and `live_inference: false`.
