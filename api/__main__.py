"""Run the API with its environment-driven host and port settings."""

from __future__ import annotations

import uvicorn

from api.config import APISettings


def main() -> None:
    settings = APISettings.from_environment()
    uvicorn.run("api.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
