# Limitations and claim boundaries

## Data coverage

The fresh final cohort contains only 34 Barcelona-centred matches from 2016/17, not a complete league season. This produces opponent, label, and history-coverage shift. Confidence intervals are wide and many apparent ranking differences are not conclusive.

## Temporal scope

One complete development season is too little to establish multi-season robustness. The 79-match 2015/16 tail was already inspected and is honestly retained as `pilot_holdout`, not renamed as a fresh final test.

## Calibration

Only 62 independent matches fit calibrators. Repeated snapshots add rows but not independent final outcomes. Phase calibration can reduce ECE while worsening RPS, especially under distribution shift.

## Market comparison

The benchmark is frozen pre-match B365 probability, not a live odds path. The project does not model staking, liquidity, limits, price movement, or transaction costs. It does not claim profitability or bookmaker superiority.

## P1 interpretation

The network method is adapted from Opta 2018/19 to StatsBomb 2015/16. Formula and implementation fidelity do not imply numerical reproduction. Centrality/points association is descriptive and observational, not causal.

## P2 interpretation

Scratch FIGS is reference-faithful and compact, but not uniformly more accurate than ensembles. Its strongest result is late in matches; that does not imply a universally superior interpretable learner.

## Explainability

SHAP explains fitted prediction behavior under its model assumptions. It does not prove that a feature causes a football outcome. The headline Model 3 multiclass gradient-boosting estimator is unsupported by the chosen TreeExplainer, so no request-time explanation is fabricated.

## API and demo

The API serves known matches and frozen five-minute snapshots. It is not a live feed or an arbitrary-fixture forecasting engine. The Pages demo is stored held-out replay JSON. GitHub Pages cannot run FastAPI.

## External academic/history items

The originally supplied directory lacked historical Git metadata and genuine TA sign-off. Software cannot reconstruct or backdate either. The repository now has Git history for subsequent work, but the missing earlier history remains documented.
