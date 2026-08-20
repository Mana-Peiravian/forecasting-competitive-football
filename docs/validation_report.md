# Validation report

## Data

- 414 unique matches: 380 complete La Liga 2015/16 fixtures plus the 34 matches available in StatsBomb open data for 2016/17.
- 1,420,167 unique event IDs; complete event coverage for all 414 matches.
- 14,899 lineup rows, 657 players, 23 teams.
- Odds coverage: 414/414 exact date/home/away joins; no duplicate keys and no excluded matches.
- 360: explicitly unavailable for both selected seasons; an empty schema and availability manifest are retained.
- All target-free prematch and snapshot rows are unique at their declared grain.

## Leakage and splits

- Roles: train 189, validation 50, calibration 62, opened pilot 79, sealed final 34.
- Four per-row source-time columns show that event and P1 histories precede kickoff.
- 7,866 snapshots: 19 per match at minutes 0--90 in five-minute increments.
- No event has `feature_source_max_seconds > snapshot_seconds`.
- Every match has exactly one split role; snapshots never cross roles.
- Final targets were opened once after a 21-file SHA-256 lock; see `outputs/final_evaluation/FINAL_TEST_OPENED.json`.

## P1

- All 760 development team-match rows reproduce historical notebook metrics.
- Largest absolute metric difference: 2.72e-15; maximum eigen residual: 7.86e-14.
- Supra shape is 40 by 40, blocks are direction-tested, and completed-pass and transition truth tables are unit-tested.
- Orientation checks pass for both home and away teams.

## P2 / scratch FIGS

- Production source has no `imodels` import.
- Paper toy, existing-tree revisit, multiclass probability, determinism, zero-rule, and backfitting tests pass.
- Regression and multiclass outputs match independent `imodels` reference behavior to 1e-12 on fixed toys.
- Classification uses residualized one-hot least squares plus softmax, matching the official reference; the paper-prose Gini discrepancy is documented.

## Models and evaluation

- Every required prematch and snapshot model family has a result row, including dummy and frozen-market baselines.
- Platt and isotonic calibrators use only the calibration role; snapshot calibration is phase-specific.
- SMOTE, Borderline-SMOTE, and ADASYN are training-only prematch experiments. No synthetic snapshot exists.
- RPS, log loss, multiclass Brier, ECE, accuracy, MAE, RMSE, and correlation are centrally defined and tested.
- Final RPS/MAE intervals use seeded bootstrap resampling; snapshots resample whole matches.
- SHAP support/sampling status is machine-readable. Multiclass sklearn GradientBoostingClassifier is the sole unsupported TreeExplainer combination.

## Automated checks

`python -m pytest` passes 16 tests. The project itself has a consistent pinned
dependency set. `pip check` reports unrelated pre-existing global conflicts for
Streamlit/TensorFlow.js; neither is a project dependency nor used by this work.

