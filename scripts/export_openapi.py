"""Generate or verify the static OpenAPI document used by GitHub Pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app  # noqa: E402


OUTPUT = ROOT / "docs" / "api" / "openapi.json"


def schema() -> dict:
    return app.openapi()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed schema differs from the FastAPI application.",
    )
    args = parser.parse_args()
    generated = schema()
    if args.check:
        if not OUTPUT.exists() or json.loads(OUTPUT.read_text(encoding="utf-8")) != generated:
            raise SystemExit("docs/api/openapi.json is stale; run scripts/export_openapi.py")
        print(f"OpenAPI schema is synchronized: {OUTPUT.relative_to(ROOT)}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(generated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
