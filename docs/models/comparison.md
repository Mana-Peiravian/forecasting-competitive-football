# Model comparison

All required families are evaluated under fixed temporal roles: dummy, SVC/SVR, exact and Nystroem kernels, random forest, gradient boosting, XGBoost, LightGBM, P1-only linear models, scratch FIGS, market, and frozen pre-match baselines.

## Final headline configurations

| Task | Public model | Final score | Reason for public selection |
|---|---|---:|---|
| Model 1 outcome | Random forest + Platt | RPS 0.120 | Best final calibrated pre-match RPS; named headline in the report |
| Model 2 margin | Random forest | MAE 1.479 | Best final prematch MAE |
| Model 3 outcome | Gradient boosting, raw | RPS 0.116 | Best overall final snapshot RPS |
| Model 3 margin | XGBoost | MAE 1.276 | Best final snapshot MAE |

These choices describe the frozen public inference surface. They do not imply that final labels were used for development retuning: every candidate and configuration was locked before the one-shot evaluation, and the site exposes the report’s final headline artifacts.

## Feature ablation

On development validation, market-only inputs are strongest for both the random-forest classifier and regressor. Adding hundreds of conventional and network histories does not reliably improve them. Scratch FIGS also changes materially by feature group but does not show a stable P1 lift.

The correct conclusion is not “P1 improves FIGS.” It is that P1 supplies a scientifically meaningful representation whose predictive contribution is inconsistent under this data volume and temporal shift.

## FIGS trade-off

Scratch FIGS is compact and directly inspectable, but its final performance is not uniformly competitive with ensembles. Its strongest behavior is late-match classification after phase calibration, where a small additive structure can exploit score and time state effectively.

## Market comparison

Several learned pre-match outputs have point-estimate RPS near or below the raw market. For the headline RF/Platt difference, the paired interval crosses zero. The project makes no profitability or market-superiority claim.

The complete generated tables are on the [Final Results](../results/index.md) page.
