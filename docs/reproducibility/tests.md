# Tests and validation

## Complete test suite

```bash
python -m pytest
```

Coverage includes:

- P1 zone boundaries, supra blocks, transition direction, eigen residual, leakage/recovery/switching formulas;
- exact 30:00.000 versus 30:00.001 snapshot inclusion;
- hand-checked RPS, probability normalization, Brier, and ECE;
- scratch FIGS paper toy, tree revisit, multiclass probabilities, determinism, zero-rule behavior, backfitting, and optional reference parity;
- multiclass and phase calibrator normalization/order;
- API health, metadata, lookup, invalid input, response schemas, probability sums, and timelines;
- genuine end-to-end Model 1, Model 2, and Model 3 predictions compared with saved final outputs;
- generated OpenAPI equality with the FastAPI application schema.

## Focused checks

```bash
python -m pytest tests/test_api.py
python scripts/export_openapi.py --check
python scripts/build_docs_data.py
mkdocs build --strict
```

The API tests use FastAPI’s in-process `TestClient`; no external server is required.

## Documentation checks

Strict MkDocs build treats warnings as failures. The generated site is then inspected for:

- missing image/assets;
- broken relative links;
- valid OpenAPI JSON;
- Swagger and ReDoc references to the same schema;
- project-subpath-safe URLs;
- accidental absolute local paths and secret-like content.

## Validation evidence

The [validation report](../validation_report.md) records data coverage, split/leakage checks, P1/P2 fidelity, model/calibration evidence, and the original scientific test count. The site implementation adds API and documentation checks without altering the finished ML protocol.
