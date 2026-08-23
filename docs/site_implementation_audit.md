# Documentation and API implementation audit

**Audit date:** 2026-08-23  
**Repository:** `Mana-Peiravian/forecasting-competitive-football`  
**Scope:** the complete tracked working tree, including source, notebooks, normalized/processed data, frozen models, final outputs, report, tests, and deployment files.

The finished ML work is intact and does not need retraining. The public inference surface can be built from the frozen final Joblib artifacts plus the already materialized leakage-safe feature tables. GitHub Pages will host only static documentation and a replay demo; Python inference remains a separately deployable FastAPI service.

| Component | Existing state | Reusable? | Required changes | Final path |
|---|---|---:|---|---|
| ML package | Tested `ml_project` package with data, P1, FIGS, feature, snapshot, calibration, metric, and modeling modules | Yes | Preserve; import its class order and load its serialized custom classes | `src/ml_project/` |
| Pre-match feature pipeline | 391 frozen predictors with shift-before-roll histories and strict source-time checks | Yes | Expose saved rows by `match_id`; do not recreate a simplified feature pipeline | `src/ml_project/features.py`, `outputs/features/prematch_features.parquet` |
| In-play feature pipeline | 507 combined predictors at 19 five-minute snapshots; exact boundary is `event_seconds_exact <= snapshot_seconds` | Yes | Restrict public snapshot requests to the frozen 0, 5, ..., 90 grid and use saved rows | `src/ml_project/snapshots.py`, `outputs/snapshots/snapshot_features.parquet` |
| Model 1 classifier | All locked final classifiers exist; report headline is random forest with Platt calibration | Yes | Load once at API startup and preserve class order `A, D, H` | `models/reproducible/final/prematch_classifier_random_forest.joblib` |
| Model 1 calibrator | Fitted holdout-only multiclass Platt calibrator | Yes | Apply to raw RF probabilities | `models/reproducible/final/prematch_calibrator_random_forest_platt.joblib` |
| Model 2 regressor | All locked final regressors exist; final best MAE is random forest | Yes | Load once; clip expected margin to `[-5, 5]` | `models/reproducible/final/prematch_regressor_random_forest.joblib` |
| Model 2 probability mapper | Calibration-block multinomial mapper from scalar margin to A/D/H | Yes | Label probabilities explicitly as derived from Model 2 | `models/reproducible/final/prematch_margin_mapper_random_forest.joblib` |
| Model 3 outcome classifier | All locked snapshot classifiers exist; raw gradient boosting is the final RPS headline | Yes | Return raw headline probabilities and the fitted phase-Platt alternative without conflating them | `models/reproducible/final/snapshot_classifier_gradient_boosting.joblib` |
| Model 3 phase calibrator | Fitted early/middle/late Platt calibrator | Yes | Load once and identify that it improves ECE but worsened final RPS | `models/reproducible/final/snapshot_phase_calibrator_gradient_boosting_platt.joblib` |
| Model 3 margin regressor | All locked snapshot regressors exist; XGBoost has the best final MAE | Yes | Load once and pair it with its fitted probability mapper | `models/reproducible/final/snapshot_regressor_xgboost.joblib` |
| Match catalogue | 414 public StatsBomb fixtures with teams, dates, seasons, split roles, odds, and feature rows | Yes | Add filtered, paginated lookup and safe metadata responses | `outputs/features/prematch_features.parquet` |
| StatsBomb events | 1,420,167 normalized events across all 414 matches | Yes, offline | Not required per API call because exact snapshots are already frozen; needed to build new snapshot grids | `data/normalized/events.parquet` |
| Final held-out replay | 34-match 2016/17 cohort, 646 snapshots, saved predictions and scores | Yes | Generate a small static JSON replay for Pages; label it as stored historical backtesting | `docs/assets/data/demo-matches.json` |
| P1 analysis | Complete matrices, metrics, tables, and seven scientific figures | Yes | Reuse selected figures and document Opta-to-StatsBomb adaptation | `outputs/p1/`, `outputs/figures/p1/`, `docs/papers/p1.md` |
| P2 FIGS | From-scratch tested implementation plus exported classifier/regressor rules | Yes | Document algorithm, fidelity nuance, split budget, and actual negative/positive results | `src/ml_project/figs.py`, `outputs/tables/models/`, `docs/papers/p2.md` |
| Calibration and SHAP | Final reliability outputs, global/local SHAP tables, and figures are saved | Yes, offline | Document them; do not expose a request-time SHAP endpoint because the headline Model 3 classifier is unsupported by TreeExplainer and no stable public explanation contract exists | `outputs/explainability/`, `outputs/figures/explainability/` |
| Final-defence demo | No deployable dashboard or API exists; the report contains a stored 19-snapshot explanation timeline | Partly | Add a static Pages replay and a separately deployable API | `docs/demo.md`, `api/` |
| FastAPI backend | Absent | No | Add a versioned frozen-artifact inference service with typed schemas, CORS configuration, and tests | `api/` |
| OpenAPI | Absent | No | Generate from FastAPI; never hand-edit the static copy | `scripts/export_openapi.py`, `docs/api/openapi.json` |
| Documentation site | Existing `docs/` contains audit/data/validation Markdown only | Partly | Add MkDocs Material structure, navigation, search, diagrams, API docs, results, and developer guides | `mkdocs.yml`, `docs/` |
| GitHub Pages workflow | No `.github/` workflows are present | No | Add build/artifact/deploy workflow; never run training | `.github/workflows/docs.yml` |
| Runtime environment | ML dependencies are pinned; FastAPI is absent | Partly | Add pinned API dependencies and optional container packaging | `pyproject.toml`, `requirements.txt`, `Dockerfile` |
| Documentation environment | Absent | No | Add only site, OpenAPI-export, and static-data-generation dependencies | `requirements-docs.txt` |
| Report | Final 12-page PDF and LaTeX source exist | Yes | Publish the reasonably sized final PDF and summarize it rather than duplicating it | `docs/assets/report/final_report.pdf`, `docs/about/report.md` |

## Audit conclusions

- **A — Existing inference API:** none.
- **B — Trained artifacts:** complete frozen final suites exist for Models 1–3, including prematch calibrators, snapshot phase calibrators, and margin-to-outcome mappers.
- **C — Reused feature functions/resources:** pre-match features come from the production shift-before-roll pipeline; in-play features come from the production exact-boundary snapshot pipeline. Public inference uses their frozen materialized rows and ordered manifests.
- **D — Safe public surface:** match lookup, safe metadata, pre-match outcome/margin, combined pre-match output, five-minute in-play snapshots, and full stored timelines.
- **E — StatsBomb event dependency:** new matches or new arbitrary snapshot times require StatsBomb events and upstream history reconstruction. The supported frozen match/snapshot API does not retrain and does not reread raw events per request.
- **F — Calls without retraining:** every implemented `/api/v1` endpoint. Model and feature artifacts load once during application lifespan.

## Deliberate exclusions

- No arbitrary user-supplied feature vectors or serialized model uploads.
- No arbitrary-second in-play inference: the frozen, validated public grid is 0–90 in five-minute increments with `second = 0`.
- No request-time SHAP endpoint. Existing SHAP evidence remains available as static scientific output; fabricated or model-incompatible explanations are not substituted.
- No claim that GitHub Pages executes Python, performs live inference, receives a live feed, or demonstrates profitability.
