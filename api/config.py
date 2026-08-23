"""Environment-driven API configuration without a secrets dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _origins() -> tuple[str, ...]:
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "https://mana-peiravian.github.io,http://localhost:8000,http://127.0.0.1:8000",
    )
    return tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class APISettings:
    """Resolved runtime settings.

    Paths may be absolute or repository-relative. They are never included in
    public API responses.
    """

    host: str
    port: int
    model_dir: Path
    prematch_features_path: Path
    snapshot_features_path: Path
    feature_manifest_path: Path
    snapshot_manifest_path: Path
    allowed_origins: tuple[str, ...]
    public_api_base_url: str

    @classmethod
    def from_environment(cls) -> "APISettings":
        return cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            model_dir=_configured_path("MODEL_DIR", "models/reproducible/final"),
            prematch_features_path=_configured_path(
                "PREMATCH_FEATURES_PATH", "outputs/features/prematch_features.parquet"
            ),
            snapshot_features_path=_configured_path(
                "SNAPSHOT_FEATURES_PATH", "outputs/snapshots/snapshot_features.parquet"
            ),
            feature_manifest_path=_configured_path(
                "FEATURE_MANIFEST_PATH", "outputs/features/feature_manifest.json"
            ),
            snapshot_manifest_path=_configured_path(
                "SNAPSHOT_MANIFEST_PATH", "outputs/snapshots/snapshot_manifest.json"
            ),
            allowed_origins=_origins(),
            public_api_base_url=os.getenv(
                "PUBLIC_API_BASE_URL", "http://127.0.0.1:8000"
            ).rstrip("/"),
        )
