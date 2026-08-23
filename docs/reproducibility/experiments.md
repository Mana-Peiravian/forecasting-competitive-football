# Experiments and final evaluation

## Development stages

```bash
football-ml-reproduce --stage development
football-ml-reproduce --stage artifacts
```

The development stage performs candidate selection, calibration, pilot evaluation, ablations, imbalance experiments, compute profiling, and Model 3 phase experiments. The artifacts stage regenerates scientific figures, tables, and explanation outputs.

These commands can take time but are ordinary repeatable research stages.

## Temporal roles

| Role | Matches | Use |
|---|---:|---|
| Train | 189 | Base fitting during selection |
| Validation | 50 | Forward temporal selection |
| Calibration | 62 | Platt/isotonic and margin mapping only |
| Pilot holdout | 79 | Already-opened development evidence; included in final base refit |
| Final test | 34 | Chronologically later one-shot evaluation only |

## Final evaluation is intentionally not repeatable

`scripts/run_final_evaluation.py` validated the pre-final lock, fit locked models without final labels, produced forecasts, wrote the open marker, and then scored exactly once. It now exits rather than opening again:

```text
outputs/final_evaluation/FINAL_TEST_OPENED.json
```

Do not delete this marker to manufacture a fresh result. Ordinary reproduction intentionally excludes the final evaluator.

## Report and documentation artifacts

```bash
python scripts/build_report.py
python scripts/build_docs_data.py
python scripts/export_openapi.py
mkdocs build --strict
```

Documentation generation reads saved results only. It never invokes development or final evaluation.

## Determinism

The centralized seed is 42. Rows are sorted by stable date/ID/event keys, manifests freeze ordered feature lists, and bootstrap sampling is seeded. External library/platform differences can still affect low-level floating-point behavior; compatibility is guarded through tests and artifact parity checks.
