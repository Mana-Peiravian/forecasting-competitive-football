# Market comparison and calibration

Football-Data B365 odds are converted to de-vigged H/D/A probabilities and frozen at kickoff. They serve two distinct roles:

1. an allowed pre-match feature group;
2. an external forecast benchmark held constant throughout Model 3 timelines.

## Model 1 versus the market

| Forecast | Final RPS | 95% bootstrap interval |
|---|---:|---:|
| Random forest + Platt | 0.120 | 0.070–0.185 |
| Raw market | 0.131 | 0.066–0.210 |

The paired mean RPS difference is −0.0116 in favor of RF/Platt, with interval `[-0.0495, 0.0168]`. Because zero is included, the evidence is compatible with both a small benefit and a small disadvantage.

## Calibration results

| Forecast | RPS | ECE |
|---|---:|---:|
| RF raw | 0.131 | 0.157 |
| RF Platt | 0.120 | 0.112 |
| Market raw | 0.131 | 0.122 |
| Market Platt | 0.126 | 0.101 |
| Model 3 GB raw | 0.116 | 0.112 |
| Model 3 GB phase-Platt | 0.134 | 0.055 |

Calibration is not an automatic improvement to every scoring rule. Model 3 phase-Platt is substantially better under ECE but worse under RPS and log loss, reflecting the small 62-match phase calibration sample.

## Minute-by-minute baseline

The raw market stays at RPS 0.131 at every snapshot because it is frozen at kickoff. Raw gradient boosting begins at 0.136, improves to 0.104 at halftime, 0.087 at minute 75, and 0.082 at minute 90. The comparison makes the source of value explicit: late improvement comes from observed match state, not updated bookmaker odds.

## Scientific claim boundary

The project neither models transaction costs nor constructs a betting strategy. “Lower historical RPS” is not synonymous with “profitable,” and the selected final sample is too small and biased for a market-efficiency claim.
