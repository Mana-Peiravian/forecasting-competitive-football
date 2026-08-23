# Model 1 — pre-match outcome

Model 1 estimates the complete ordered outcome distribution at kickoff:

```text
input  = leakage-safe historical pre-match features
output = P(Away), P(Draw), P(Home)
```

The public API uses the final **random forest** and its disjoint-holdout **multiclass Platt calibrator**. Internally the frozen order is `A, D, H`; API responses use named keys.

## Prediction-time information

The 391-feature combined vector contains conventional event histories, P1 network histories, de-vigged market probabilities, rest, season progress, calendar context, and market overround. Every team history was shifted by one match before rolling/expanding aggregation.

No current-match event, final score, outcome, or target is present.

## Objective and metrics

Ranked Probability Score is the headline because football outcomes are ordered and the task is probabilistic. With ordered classes A–D–H, RPS penalizes both probability misallocation and how far it lies from the observed class.

The project also retains log loss, multiclass Brier score, ten-bin top-label Expected Calibration Error, and accuracy. Accuracy is not the sole objective: a confident wrong prediction and a cautious one have the same argmax error but very different forecast quality.

## Final result

Random-forest/Platt reaches final RPS 0.120 on 34 matches, versus 0.131 for the raw market. The paired mean difference is −0.0116, but its 95% interval `[-0.0495, 0.0168]` includes zero. The result is therefore an uncertain small-cohort estimate, not a bookmaker-beating claim.

![Final Model 1 RPS](../assets/images/results/model1_rps.png)

## Calibration

The base random forest fits train + validation + pilot. Its raw probabilities on the separate calibration block are transformed by multinomial logistic regression over log probabilities. Final labels never fit the calibrator.

Platt scaling improves final RF RPS from 0.131 to 0.120 and ECE from 0.157 to 0.112. Calibration is model- and sample-dependent; it worsens some other candidates, which is why raw and calibrated outputs remain separately visible.

## API representation

`POST /api/v1/predict/prematch/outcome` returns calibrated headline probabilities, raw probabilities, the predicted class, model identity, calibration method, and explicit class order. See [Endpoints](../api/endpoints.md#model-1-pre-match-outcome).
