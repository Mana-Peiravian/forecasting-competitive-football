"""Tiny requests-based client for the public API surface."""

from __future__ import annotations

from typing import Any

import requests


class FootballForecastClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(
            method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def matches(self, **filters: Any) -> dict[str, Any]:
        return self._request("GET", "/api/v1/matches", params=filters)

    def predict_prematch(self, match_id: int) -> dict[str, Any]:
        return self._request("POST", "/api/v1/predict/prematch", json={"match_id": match_id})

    def predict_inplay(self, match_id: int, minute: int, second: int = 0) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/predict/inplay",
            json={"match_id": match_id, "minute": minute, "second": second},
        )

    def timeline(self, match_id: int) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/matches/{match_id}/timeline")
