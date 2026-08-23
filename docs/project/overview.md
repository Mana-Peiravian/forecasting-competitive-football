# Project overview

The project asks a compact but demanding question: how much can be learned about a football match **before kickoff**, and how should that forecast change as events arrive—without allowing future information into the past?

It delivers three related tasks.

| Task | Information boundary | Output | Headline metric |
|---|---|---|---|
| Model 1 — pre-match classification | Leakage-safe history and market information available at kickoff | `P(Away)`, `P(Draw)`, `P(Home)` | Ranked Probability Score (RPS) |
| Model 2 — pre-match regression | Same kickoff-time feature vector | Clipped signed final goal margin | Mean Absolute Error (MAE) |
| Model 3 — in-play classification/regression | Frozen pre-match prior plus events at or before time `t` | Updated outcome probabilities and final margin | RPS and MAE versus minute |

The class order is fixed internally as **A, D, H**. The API presents descriptive keys—`away_win`, `draw`, and `home_win`—so callers do not need to infer column positions.

## Research components

Two papers are integrated as functional parts of the system:

- **P1 — A multilayer network framework for soccer analysis** supplies a domain representation of passing and possession-change structure across two team layers.
- **P2 — Fast Interpretable Greedy-Tree Sums** supplies an additive, low-split-budget predictive model implemented from scratch for the coursework.

P1 features and P2 FIGS are evaluated alongside dummy, kernel, random-forest, gradient-boosting, XGBoost, LightGBM, and market baselines. The integration is empirical: network features do not consistently improve generalization, and the documentation preserves that negative result.

## Evaluation protocol

La Liga 2015/16 forms the 380-match development backbone: 189 train, 50 validation, 62 calibration, and 79 opened pilot matches. A chronologically later 2016/17 StatsBomb subset supplies 34 final matches. All model, feature, and calibration choices were frozen before those final labels were opened once.

The final cohort is Barcelona-centred because that is what StatsBomb Open Data exposes for the season. It is useful as a fresh temporal check, but not representative of a complete league. Conclusions are therefore framed as historical backtesting with wide uncertainty—not future deployment performance.

## Engineering deliverables

- Normalized relational data and provenance hashes
- Production feature and exact snapshot builders
- Saved frozen estimators, calibrators, and margin mappers
- Automated unit, leakage, calibration, FIGS parity, and API tests
- Versioned FastAPI inference surface
- Generated OpenAPI, Swagger UI, and ReDoc
- MkDocs Material documentation deployed by GitHub Actions
- A static, clearly labelled held-out replay that requires no backend

The [implementation audit](../site_implementation_audit.md) maps every reused component to its final public path.
