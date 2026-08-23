# Model 2 — signed final goal margin

Model 2 predicts a scalar target:

```text
signed margin = home goals - away goals
target        = clip(signed margin, -5, 5)
```

Interpretation is direct:

- positive: expected home advantage;
- zero: an even expected result;
- negative: expected away advantage.

Clipping fixes the modeling target and limits rare blowouts from dominating squared-loss learners. The API applies the same bounds to predictions.

## Inputs

Model 2 uses the same 391 kickoff-time predictors as Model 1. The frozen public estimator is the final **random forest**, which records the best final MAE among the locked candidates.

## Metrics

MAE is the headline because it stays in goal units and is robust to a few large errors. RMSE and Pearson correlation are reported alongside it. On the 34-match final cohort, random forest obtains MAE **1.479**, RMSE 1.792, and correlation 0.766.

![Final Model 2 MAE](../assets/images/results/model2_mae.png)

## Margin-to-outcome mapping

The project also tests whether a scalar margin forecast can recover H/D/A probabilities. A multinomial logistic model is fitted on the disjoint calibration block using only Model 2’s out-of-sample margin prediction. It is not a rounding rule and it is not direct Model 1 output.

The API labels these values `derived_outcome_probabilities` and the mapping `calibration_block_multinomial_logistic`. The random-forest margin mapper reaches final converted RPS 0.116, but users should retain the distinction between:

- Model 1: direct probabilistic classification;
- Model 2: regression followed by a separately fitted probability mapping.

## API representation

`POST /api/v1/predict/prematch/margin` returns `expected_goal_margin`, teams, the regressor identity, and derived probabilities. `POST /api/v1/predict/prematch` returns Models 1 and 2 together.
