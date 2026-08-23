import json
from pathlib import Path

from api.main import app


def test_exported_openapi_is_valid_and_synchronized():
    exported_path = Path("docs/api/openapi.json")
    assert exported_path.exists(), "Run: python scripts/export_openapi.py"
    exported = json.loads(exported_path.read_text(encoding="utf-8"))
    generated = app.openapi()
    assert exported == generated
    assert generated["openapi"].startswith("3.")
    assert "/api/v1/predict/inplay" in generated["paths"]
