# Leakage prevention

Football forecasting is unusually easy to leak because final scores, later events, season summaries, and odds joins can silently expose the answer. The project treats information boundaries as testable invariants.

## Pre-match boundary

- Team histories are shifted before any rolling or expanding aggregate.
- Four retained source-time fields must be strictly earlier than the current kickoff.
- Scores and targets are excluded from the feature Parquet.
- Final-season rows are forbidden from development selection, calibration, ablation, and fitting code.

## In-play boundary

At snapshot time `t`, the inclusion predicate is exact:

```python
event_seconds_exact <= snapshot_seconds
```

The millisecond boundary test proves that an event at 30:00.000 is included while 30:00.001 is excluded. `feature_source_max_seconds` is retained for every snapshot and may never exceed the cutoff.

## Split isolation

Every match has exactly one temporal role. All 19 snapshots from that match inherit the same role; no match contributes one snapshot to training and another to evaluation. Calibration uses 62 independent matches, not “1,178 independent observations” merely because each has repeated snapshots.

## One-shot final evaluation

Before opening final labels, 21 source, configuration, feature, split, and evaluator files were frozen by SHA-256. The evaluator produced forecasts, wrote the permanent open marker, then read labels and scored once. `FINAL_TEST_OPENED.json` now prevents a second run.

## Market integrity

Odds join keys are date and explicit team aliases only. Scores, results, and match statistics cannot influence whether or how an odds row is matched.

## API integrity

The public API accepts known `match_id` values and a validated five-minute snapshot. It does not accept arbitrary feature vectors, file paths, or user-supplied pickle/Joblib content. That keeps the public contract aligned with the tested production feature boundaries.

See [Tests & Validation](../reproducibility/tests.md) for the executable checks.
