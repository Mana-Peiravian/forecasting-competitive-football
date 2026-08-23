"""Validate built-site internal links and API-reference assets."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SITE_PREFIX = "/forecasting-competitive-football/"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in {"href", "src", "spec-url"}:
                self.targets.append(value)


def _exists(source: Path, target: str) -> bool:
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return True
    if parsed.path.startswith(SITE_PREFIX):
        destination = (SITE / unquote(parsed.path[len(SITE_PREFIX) :])).resolve()
    elif parsed.path.startswith("/"):
        return False
    else:
        destination = (source.parent / unquote(parsed.path)).resolve()
    try:
        destination.relative_to(SITE.resolve())
    except ValueError:
        return False
    if destination.is_dir() or parsed.path.endswith("/"):
        destination = destination / "index.html"
    return destination.is_file()


def main() -> None:
    if not SITE.is_dir():
        raise SystemExit("site/ does not exist; run: mkdocs build --strict")
    broken: list[str] = []
    for html in SITE.rglob("*.html"):
        parser = LinkParser()
        parser.feed(html.read_text(encoding="utf-8"))
        for target in parser.targets:
            if not _exists(html, target):
                broken.append(f"{html.relative_to(SITE)} -> {target}")
    schema_path = SITE / "api" / "openapi.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = {
        "/api/v1/health",
        "/api/v1/models",
        "/api/v1/predict/prematch",
        "/api/v1/predict/inplay",
        "/api/v1/matches/{match_id}/timeline",
    }
    missing = required - set(schema.get("paths", {}))
    if missing:
        broken.append(f"OpenAPI missing paths: {sorted(missing)}")
    if broken:
        raise SystemExit("Broken documentation references:\n" + "\n".join(broken))
    print(f"Checked {len(list(SITE.rglob('*.html')))} HTML files; internal links and API assets are valid.")


if __name__ == "__main__":
    main()
