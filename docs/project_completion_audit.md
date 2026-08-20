# Project Completion Audit

**Audit date:** 2026-08-20  
**Repository:** `ML_project_so_far`  
**Purpose:** pre-implementation reproducibility and rubric audit for the Phase 3 submission.

This document was created before implementation changes. It records the state of the repository as received. The status table is intended to remain the authoritative completion tracker; later work should update the final-status columns without erasing the initial evidence.

## Sources reviewed end to end

- `Final_Project_Machine_Learning.pdf` (23 pages), including the required model suites, evaluation protocol, report structure, submission checklist, and TA sign-off language.
- `P1 -  A multilayer network framework for soccer analysis.pdf` (11 pages), including the 4 by 5 pitch partition, supra-adjacency construction, right-eigenvector centrality, leakage, recovery, and switching definitions.
- `P2 - Fast Interpretable Greedy-Tree Sums.pdf` (10 pages), including simultaneous tree growth, residualization, split selection, classification/regression losses, and complexity.
- The official P2 supplementary appendix (`pnas.2310151122.sapp.pdf`, 21 pages), obtained from the publisher's Europe PMC supplementary-files endpoint and reviewed in full. This includes the runtime derivation, backfitting description, and additional FIGS examples.
- The official `imodels/tree/figs.py` reference implementation, used only as an independent behavioral reference for the planned from-scratch implementation.
- All seven notebooks, including source, saved outputs, execution counts, and cross-notebook assumptions.
- Every CSV, Parquet, JSON, Joblib, YAML, Markdown, and ignore file in the repository.

## Initial repository inventory

| Artifact class | Initial state |
|---|---|
| Git repository | **Missing.** The supplied directory has no `.git` metadata; `git status` and `git log` fail. Historical commits cannot be reconstructed honestly. |
| Notebooks | 7 notebooks, mostly executed, covering integration, event ingestion, P1 metrics, shifted P1 features, modeling data, conventional baselines, and an `imodels` FIGS experiment. |
| Processed data | 28 CSV and 7 Parquet files. Main event table has 1,295,354 events from all 380 La Liga 2015/16 matches. |
| Saved models | 16 Joblib files. Most are not loadable in the current environment because they were serialized with incompatible scikit-learn versions or depend on absent XGBoost, LightGBM, or `imodels`. |
| Python package/source | **Missing.** No `src/`, reusable library, CLI, or test suite. |
| Report | **Missing.** No paper source or compiled PDF. |
| Figures/tables | **Missing** as durable submission artifacts; figures exist only in notebook output cells. |
| Environment | One aspirational `environment.yml` containing versions unavailable or different from the actual runtime. No verified lock file. |
| README | Two lines; no setup, data provenance, experiment protocol, artifact map, or reproduction command. |

## Data and evaluation evidence found

- The integrated match table contains all 380 La Liga 2015/16 fixtures with scores and de-vigged B365 H/D/A market probabilities. All rows are complete.
- The raw odds source, normalized relational tables, and explicit join-exclusion/mapping artifacts are absent. The integration notebook documents a strict date/home-team/away-team join and aliases, but the final table omits the original decimal odds.
- Event ingestion logs report 380 successful matches. The consolidated event Parquet covers 380 match IDs and dates 2015-08-21 through 2016-05-15.
- P1 match metrics contain 760 match-team rows; zone metrics contain 15,200 match-team-zone rows. Zone-level nulls occur where a mathematical denominator is zero.
- Prematch network features are correctly shifted before rolling/expanding aggregation in the notebook, but there is no durable row-level `source_match_time` proof or automated leakage assertion.
- The conventional feature set has only six predictors (rolling goals for/against and limited history counts), substantially below the requested strong historical event/discipline/shot/xG feature baseline.
- The snapshot notebook is a one-match prototype. It uses `minute <= t`, which includes events such as 30:47 in a 30:00 snapshot and therefore violates the exact-boundary requirement. Model 3 is otherwise absent.
- The baseline notebook imports `CalibratedClassifierCV` but never fits it. Files labelled as calibration results contain raw validation scores, so the required Platt/isotonic calibration comparison is absent.
- The current chronological 79-match test begins on 2016-04-02. It appears in model selection, model comparison, saved predictions, error analysis, and FIGS experiments. It is therefore an **opened pilot holdout**, not an untouched final test set. All existing metrics must be relabelled pilot/development results.
- The FIGS notebook imports `imodels.FIGSClassifier` and `FIGSRegressor`; it is not a from-scratch P2 implementation.

## Requirement-by-requirement completion matrix

Status values: **Complete**, **Partial**, **Missing**, or **Incorrect**. “Initial status” describes the repository as received.

| ID | Requirement | Initial status | Initial evidence / gap | Required completion action |
|---:|---|---|---|---|
| R01 | Reusable project package and commands | Missing | Notebook-only workflow | Add `src/ml_project`, scripts/CLI, configuration, and tests. |
| R02 | Reproducible environment | Incorrect | Declared versions do not match installed/available versions | Pin a verified environment and record actual package versions. |
| R03 | Raw-data provenance and download instructions | Partial | Notebook URLs/StatsBomb API calls exist; raw sources absent | Add manifest, source URLs, checksums where practical, and deterministic fetch/build commands. |
| R04 | Relational competitions table | Missing | No normalized table | Materialize and document. |
| R05 | Relational matches table | Partial | Integrated match table exists but is not a normalized raw match table | Materialize normalized match metadata. |
| R06 | Relational teams table | Missing | Team strings appear in other tables | Materialize stable team IDs/names and alias table. |
| R07 | Relational players table | Missing | Event player fields exist | Materialize player IDs/names. |
| R08 | Relational lineups table | Missing | Not retained | Fetch/materialize where available. |
| R09 | Relational events table | Complete | 1,295,354 events, 380 matches | Add schema/data dictionary and validation tests. |
| R10 | StatsBomb 360 handling | Missing | Competition/season has no retained 360 table | Explicitly document availability and materialize if available; otherwise record non-availability rather than fabricate. |
| R11 | Odds table with raw decimal odds | Partial | De-vigged probabilities retained; raw B365 columns omitted | Retain raw odds plus implied and de-vigged probabilities. |
| R12 | Auditable odds date/team mapping | Partial | Aliases and strict join in notebook only | Export alias map, join diagnostics, duplicates, unmatched/excluded rows. |
| R13 | No score/statistics used in odds join | Complete | Join keys are clean date/home/away | Lock with test and documentation. |
| R14 | Outcome target H/D/A | Complete | Derivable and already used | Centralize definition and class order. |
| R15 | Goal-margin target clipped to [-5, 5] | Partial | Margin exists; clipping needs central validation | Centralize and test. |
| R16 | P1 4 by 5 pitch zones | Complete | `GRID_X_BINS=4`, `GRID_Y_BINS=5` | Move to reusable module and test boundary cases. |
| R17 | P1 attacking orientation validation | Missing | No documented or empirical validation | Test coordinate convention empirically and produce diagnostics. |
| R18 | P1 intra-layer completed-pass edges | Complete | Successful passes only, source-to-destination zones | Reimplement/test in package. |
| R19 | P1 inter-layer possession-change edges | Partial | Boundary-aware same-period team-change logic is present | Add synthetic truth-table tests and document terminal possession handling. |
| R20 | P1 40 by 40 home/away supra-adjacency | Complete | Correct block layout is present | Add shape/block tests and durable matrices for examples. |
| R21 | P1 dominant right-eigenvector centrality | Complete | `eig(A)` right eigenvector, max real eigenvalue | Add residual/non-negativity normalization tests and handle reducible graphs explicitly. |
| R22 | P1 individual-layer centralities | Missing | Only supra centrality retained | Compute/save individual-layer centrality for requested analyses. |
| R23 | P1 leakage metric | Complete | Inter-layer out-degree / total in-degree | Test zero-denominator policy. |
| R24 | P1 recovery metric | Complete | Inter-layer in-degree / total in-degree | Test zero-denominator policy. |
| R25 | P1 switching metric | Complete | Half sum of zone inter-in/out to pass-in ratios | Document interpretation and test formula. |
| R26 | P1 z-score summaries | Partial | Match/team z-score columns exist | Add league/team/zone reproducible tables and plots. |
| R27 | P1 example match multilayer network | Missing | Notebook-level exploratory plots only | Produce publication-ready saved figure. |
| R28 | P1 league-average network | Missing | No durable artifact | Produce saved average network figure. |
| R29 | P1 zone heatmaps and team comparisons | Missing | No durable artifact | Produce requested heatmaps/team figures. |
| R30 | P1 distributions/table/centrality-vs-points | Missing | No submission artifacts | Produce figures and comparison table with caveats. |
| R31 | Leakage-safe historical P1 features | Partial | Shift-before-rolling is present | Add explicit max-source-time metadata and assertions. |
| R32 | Strong conventional historical features | Missing | Only six weak features | Add form, shots/xG, discipline, pass/pressure/recovery, home/away and expanding features. |
| R33 | Prematch context features | Partial | Some counts/history variables | Add calendar/rest/season progress and carefully justified context. |
| R34 | Market probabilities used as benchmark/features | Complete | Market H/D/A retained and used | Preserve as separately reported benchmark. |
| R35 | Frozen feature schema | Missing | Notebook-local lists | Save ordered schemas and feature-group manifests. |
| R36 | Fresh untouched final test | Incorrect | 79-match tail is repeatedly opened | Freeze it as pilot and establish a later-season final evaluation set before final evaluation. |
| R37 | Temporal train/validation/calibration partitions | Partial | Expanding CV and a validation block exist | Formalize non-overlapping temporal roles and persist split manifest. |
| R38 | Match-grouped snapshot split | Missing | Model 3 absent | Guarantee all snapshots from a match share a split. |
| R39 | Exact snapshot boundary | Incorrect | Uses integer-minute comparison | Use event timestamps/seconds with `event_time <= snapshot_time` and test 30:00/30:01 boundaries. |
| R40 | Snapshot cumulative features | Missing | Minimal prototype | Build score, cards, shots/xG, possession/event and team-difference state. |
| R41 | Snapshot recent-window features | Missing | No recent windows | Add leakage-safe recent 5/10-minute features. |
| R42 | Frozen prematch features in Model 3 | Missing | No production snapshot dataset | Join prematch-only features to every snapshot. |
| R43 | Model 1 required classifier suite | Partial | SVC/RF/GB/XGB/LGB plus market tried | Add dummy, scratch P1/P2 variants, verified installs, fixed protocol. |
| R44 | Model 2 required regressor suite | Partial | SVR/KRR/Nystroem/RF/GB/XGB/LGB tried | Add dummy and scratch P2; lock protocol. |
| R45 | Model 3 required classifier/regressor suite | Missing | Not implemented | Implement dummy/frozen, RF, GB, XGB, LGB, and scratch P2. |
| R46 | Kernel scalability discussion/evidence | Partial | Exact and Nystroem models tried | Add timing/memory/scaling evidence and explicit O(n^2) limitation. |
| R47 | Actual Platt calibration | Missing | Imported, not fitted | Fit only on calibration partition; save pre/post metrics. |
| R48 | Actual isotonic calibration | Missing | Not fitted | Fit only on calibration partition; save pre/post metrics. |
| R49 | Model 3 phase-wise calibration | Missing | Model 3 absent | Fit/report by match phase without snapshot leakage. |
| R50 | Reliability diagrams and ECE | Missing | ECE scalar only; no diagrams | Add top-label and per-class reliability diagnostics. |
| R51 | Classification RPS headline metric | Partial | RPS computed | Centralize class order/normalization and test hand calculations. |
| R52 | Classification log loss/Brier/ECE | Partial | Scalars computed inconsistently | Standardize definitions and report confidence intervals where practical. |
| R53 | Regression MAE/RMSE/correlation | Complete | Existing pilot metrics include all three | Re-evaluate under locked splits. |
| R54 | Convert margin predictions to H/D/A | Missing | No conversion evaluation | Fit an ordered/probabilistic mapping on non-test data and report classification metrics. |
| R55 | Model 3 metric versus minute | Missing | Model 3 absent | Produce per-minute/phase curves with uncertainty/counts. |
| R56 | Frozen-prematch Model 3 comparator | Missing | Model 3 absent | Repeat prematch probabilities at every snapshot. |
| R57 | Imbalance: vanilla/class-weight/SMOTE/Borderline/ADASYN | Missing | No comparison | Run on training folds only; no synthetic snapshots. |
| R58 | Scratch FIGS, no `imodels` import | Missing | Existing notebook directly imports `imodels` | Implement independently in project source. |
| R59 | FIGS simultaneous global split search | Missing | Delegated to dependency | Implement and unit-test leaf/current-tree/new-root competition. |
| R60 | FIGS regression residual updates | Missing | Delegated to dependency | Implement and compare against toy/reference behavior. |
| R61 | FIGS multiclass classification/probabilities | Missing | Delegated to dependency | Implement documented multiclass objective/probability mapping and tests. |
| R62 | FIGS revisits existing trees | Missing | Delegated to dependency | Add paper-motivated interaction test. |
| R63 | FIGS split-budget interpretability | Partial | Split grid 2--20 exists for dependency | Re-run scratch estimator and export structures/rules. |
| R64 | FIGS optional backfitting | Missing | Not implemented | Implement or explicitly scope out with supplement-grounded rationale; preferred implementation. |
| R65 | FIGS validation against paper/supplement/reference | Missing | No scratch estimator | Add toy, determinism, shape, probability, and tolerance tests. |
| R66 | Ablation: conventional vs market vs network vs combined | Missing | Feature combinations not systematically reported | Run locked ablation for Models 1 and 2. |
| R67 | Scratch FIGS ablations | Missing | Dependency only | Include scratch FIGS in the same feature-group matrix. |
| R68 | SHAP global summaries | Missing | No SHAP artifacts | Run supported tree models and document sampled background/evaluation. |
| R69 | SHAP local explanations and worst cases | Missing | No artifacts | Export representative and worst-error explanations. |
| R70 | One full Model 3 explanation timeline | Missing | Model 3 absent | Export snapshot-by-snapshot probability/features/explanations for one match. |
| R71 | Error analysis | Partial | Some pilot predictions exist | Add upset/draw/large-margin/phase/market-disagreement slices and examples. |
| R72 | Compute wall time and peak memory | Missing | No resource telemetry | Instrument major stages/models. |
| R73 | Determinism and seeds | Partial | Some notebook seeds | Centralize seed, deterministic ordering, and reproducibility checks. |
| R74 | Automated tests | Missing | No test suite | Add unit/integration/leakage tests and run them. |
| R75 | Validation report | Missing | No central data/model QA output | Generate machine-readable and human-readable validation summaries. |
| R76 | Submission report, required sections 1--9 | Missing | No report | Write and compile a paper within the 25-page main-text limit. |
| R77 | Report Appendix A/B | Missing | No report | Add reproducibility details and supplementary tables/figures. |
| R78 | Honest limitations and non-claims | Missing | No report | Explicitly separate P1 concept replication from dataset-specific paper replication. |
| R79 | Comprehensive README | Missing | Two lines | Add setup, provenance, commands, outputs, and interpretation. |
| R80 | Demo/API | Missing (optional) | None | Add a practical local inference/demo artifact if time permits; do not let bonus work displace required work. |
| R81 | Exported Git history | Missing/unrecoverable | Supplied directory is not a Git checkout | Record limitation; initialize only if useful for future tracking, never fabricate prior history. |
| R82 | Human TA sign-off | Pending external action | Cannot be performed by code | Provide checklist and clearly mark as human action. |

## Critical blockers and chosen resolutions

1. **The original final test is contaminated by repeated inspection.** It will be renamed `pilot_holdout`. A chronologically later La Liga season will be acquired and sealed as `final_test`; it will not participate in model selection, calibration, thresholds, feature definitions, or ablations.
2. **Serialized estimators are not a reproducible foundation.** Existing Joblib files will be retained as historical artifacts but not used for final conclusions. Models will be retrained under a verified environment.
3. **The project lacks raw relational artifacts.** Normalized tables and provenance manifests will be rebuilt from official StatsBomb open data and the documented Football-Data odds source.
4. **No Git metadata exists.** The final submission can include the current repository state and an explicit `git_history_unavailable.txt`, but it cannot truthfully recreate historical commits.
5. **StatsBomb 360 availability is source-dependent.** Absence will be documented explicitly for selected competition/seasons. Synthetic 360 data will not be created.

## Scientific non-claims fixed at audit time

- Reproducing P1 formulas and figures on StatsBomb La Liga data is a **conceptual/methodological replication**, not an exact numerical reproduction of the paper's Opta-based 2018/19 results.
- Market probabilities are a strong external benchmark and a legitimate feature group; beating them is not assumed.
- Any final-test result will be reported once under the frozen protocol. Pilot results are developmental evidence only.
- Missing Git history and human TA approval will remain openly marked rather than being silently represented as complete.

## Final completion audit (2026-08-20)

The initial matrix above is preserved as evidence of the received state. After
implementation, requirements **R01--R79 are complete** under the honest source
limitations documented below. In particular, R10 is satisfied by explicit,
machine-readable proof that the selected seasons have no StatsBomb 360 files;
it does not imply that unavailable 360 data was created. R36 is satisfied by a
fresh, chronologically later 34-match cohort, with its Barcelona-centred source
selection reported as a major limitation rather than treated as a full season.

| ID | Final status | Final evidence |
|---:|---|---|
| R01--R09 | Complete | Installable `src/ml_project`, CLI, normalized competitions/matches/teams/players/lineups/events, provenance and schemas. |
| R10 | Complete with source limitation | Empty typed 360 table plus competition-season availability manifest proves non-availability. |
| R11--R15 | Complete | Raw B365 odds, exact alias joins, de-vigging, outcome and clipped margin definitions/tests. |
| R16--R30 | Complete | Tested P1 module, all 414 matrices, exact notebook parity, orientation validation, individual/supra centrality, z-scores, publication figures and tables. |
| R31--R42 | Complete | 391 frozen prematch predictors, four source-time proofs, 7,866 exact snapshots, cumulative/recent state, match grouping and no post-boundary events. |
| R43--R57 | Complete | Full required suites for Models 1/2/3, market/frozen/dummy baselines, actual Platt/isotonic calibration, phase calibration, margin mapping, imbalance experiments and full metrics. |
| R58--R65 | Complete | Independent scratch FIGS, global candidate competition, residual updates, old-tree revisits, multiclass softmax, split budgets, backfitting, paper toys and 1e-12 reference parity. |
| R66--R73 | Complete | Feature/FIGS ablations, supported-tree SHAP, worst-10 local explanations, full timeline, error slices, wall time/RSS, kernel scaling and deterministic seed. |
| R74--R79 | Complete | 16 passing tests, validation report, 12-page compiled report with Sections 1--9 and Appendices A/B, comprehensive README and non-claims. |
| R80 | Not attempted (optional) | API/dashboard bonus deliberately omitted so required work remained the priority. |
| R81 | Unrecoverable external item | Supplied directory had no `.git`; genuine historical commits cannot be reconstructed. See `docs/git_history_unavailable.txt`. |
| R82 | Pending human action | No TA sign-off record was supplied. Genuine P1/P2 approval must be attached by the student. See `docs/ta_signoff_required.md`. |

### Final artifact identities

- Final split manifest SHA-256: `1a736049c2b0ef62db201eb8e53a6bf739727c8bb5f30c378703dd3a3a8d435b`
- Final feature manifest SHA-256: `0ab3623416ce846fda9390f5e837c7697ae409756de256c8651af6d167cd81ee`
- Pre-final 21-file lock SHA-256: `3d730d7d3ad2e774008d914f85f6e4d065d17819e9e7ed91578654b9ab1c239f`
- Project configuration (`pyproject.toml`) SHA-256: `208bee38ba8540e0beddc8a0f1c8bb45ff214cf3e316f19aee904870b858e37b`
- Report PDF SHA-256: `0e375d0cc53bed2a528f412fd4ffc8ce0e83729bf41433b2a4c943024a62e339`
- Test command/result: `python -m pytest` -- **16 passed**.
- Final evaluation: completed exactly once; permanent marker is
  `outputs/final_evaluation/FINAL_TEST_OPENED.json` and now prevents rerun.
- Report: `report/final_report.pdf`, 12 A4 pages including appendices (main
  text ends on page 10, below the 25-page cap).
- Ordinary reproduction/validation: `football-ml-reproduce --stage validate`;
  separate `data`, `features`, `development`, and `artifacts` stages are
  documented in the README. The final stage is intentionally non-repeatable.

### Final scientific findings

- Platt random forest final pre-match RPS is 0.120 versus 0.131 for raw market;
  the paired 95% interval includes zero, so no market-beating claim is made.
- Raw in-play gradient boosting final overall RPS is 0.116 and minute-90 RPS is
  0.082; frozen market stays at 0.131.
- Network-only and network-augmented features do not consistently improve
  validation performance. This negative result is reported prominently.
- Scratch FIGS is reference-faithful and interpretable, but is not uniformly
  more accurate; its strongest result is late-match phase-calibrated prediction.
- Remaining submission actions are exactly R81 (historical Git metadata,
  unrecoverable) and R82 (genuine human TA confirmation). No software remedy is
  represented as having completed either item.
