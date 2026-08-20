# Forecasting Competitive Football

Submission-ready, leakage-safe pre-match and in-play forecasting from raw
StatsBomb events, Football-Data odds, a P1 multilayer network replication, and
a from-scratch P2 FIGS implementation.

## Headline result

The protocol was frozen before opening a chronologically later 2016/17 cohort.
On its 34 available matches, Platt-calibrated random forest obtained RPS 0.120
(95% bootstrap interval 0.070--0.185) versus 0.131 for the raw market. The
paired difference interval crosses zero, so the project does **not** claim to
beat bookmakers. Raw in-play gradient boosting obtained overall RPS 0.116 and
improved to 0.082 at minute 90; the frozen market remained 0.131.

StatsBomb exposes only 34 Barcelona-centered 2016/17 matches, not the full
season, so final uncertainty and selection bias are central limitations.

## Quick start

```powershell
conda env create -f environment.yml
conda activate football-ml
python -m pip install -e .
python -m pytest
```

Equivalent pip pins are in `requirements.txt`. The tested runtime is Python
3.13.7. To validate the delivered artifacts:

```powershell
football-ml-reproduce --stage validate
```

To rebuild ordinary stages (network reconstruction is the slowest):

```powershell
football-ml-reproduce --stage data
football-ml-reproduce --stage features
football-ml-reproduce --stage development
football-ml-reproduce --stage artifacts
```

The final evaluator is intentionally one-shot. It now refuses to run because
`outputs/final_evaluation/FINAL_TEST_OPENED.json` proves the sealed cohort was
already scored exactly once. Do not delete that marker merely to obtain a new
result.

## Scientific protocol

- Development backbone: all 380 La Liga 2015/16 matches.
- Temporal roles: 189 train, 50 validation, 62 calibration, 79 opened pilot.
- Final: 34 chronologically later StatsBomb 2016/17 matches.
- Classification order: A, D, H; RPS is the headline proper score.
- Margin: home minus away, clipped to [-5, 5].
- Snapshots: minutes 0, 5, ..., 90 with exact `event_seconds_exact <= t`.
- Synthetic sampling: training-only prematch experiments; never snapshots.
- Calibration: Platt and isotonic on the calibration block; Model 3 by phase.
- Uncertainty: match bootstrap; all snapshots from a sampled match stay together.

The original 79-match “test” appears throughout the received notebooks. It is
therefore labelled `pilot_holdout` everywhere and is never presented as final.

## Repository map

- `src/ml_project/`: tested P1, scratch FIGS, features, snapshots, metrics,
  calibration, model factories, and CLI.
- `scripts/`: deterministic build, development, one-shot final, explainability,
  artifact, and report entry points.
- `tests/`: formula, directionality, leakage-boundary, FIGS toy/reference,
  backfitting, probability, and calibration tests.
- `data/normalized/`: relational store and SHA-256 provenance manifest.
- `outputs/features/`: frozen schemas and temporal split manifest.
- `outputs/p1/`: all-match P1 matrices, metrics, z-scores, and validation.
- `outputs/experiments/`: development/pilot selection, calibration, imbalance,
  ablation, resource, and kernel-scaling evidence.
- `outputs/final_evaluation/`: one-shot final predictions, metrics, intervals,
  reliability bins, errors, and marker.
- `outputs/figures/`, `outputs/tables/`, `outputs/explainability/`: report assets.
- `report/final_report.pdf`: compiled submission paper.
- `docs/project_completion_audit.md`: before/after requirement audit.

## Paper fidelity

P1 uses 4 x 5 pitch zones, successful-pass intra-layer edges, boundary-aware
possession-change inter-layer edges, a 40 x 40 supra-adjacency matrix, the
dominant **right** eigenvector, and the published leakage/recovery/switching
formulas. The results are a methodological replication on StatsBomb 2015/16,
not a numerical reproduction of the Opta 2018/19 paper.

Scratch FIGS globally compares every current leaf with a fresh root, residualizes
against all other trees, revisits earlier trees, supports a total split budget,
multiclass softmax probabilities, and optional supplement-style backfitting.
Production code never imports `imodels`; that package is test-only reference
evidence. See `docs/validation_report.md` for parity tolerances.

## Data and licensing

StatsBomb open data: <https://github.com/statsbomb/open-data>. Please retain the
StatsBomb attribution required by its licence. Odds come from Football-Data SP1
season CSVs. The join uses only date and explicit team aliases; scores and match
statistics are not join inputs. Details are in `docs/data_dictionary.md`.

## Honest remaining actions

The supplied directory had no `.git` metadata, so genuine historical commits
cannot be exported or reconstructed; see `docs/git_history_unavailable.txt`.
No TA sign-off record was supplied; the student must attach genuine P1/P2
approval before submission as described in `docs/ta_signoff_required.md`.
The optional API/dashboard bonus was deliberately not attempted.
