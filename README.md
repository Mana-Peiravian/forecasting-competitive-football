# Forecasting Competitive Football

[![Documentation](https://github.com/Mana-Peiravian/forecasting-competitive-football/actions/workflows/docs.yml/badge.svg)](https://github.com/Mana-Peiravian/forecasting-competitive-football/actions/workflows/docs.yml)

**[Read the full GitHub Pages documentation](https://mana-peiravian.github.io/forecasting-competitive-football/)** · [API quickstart](https://mana-peiravian.github.io/forecasting-competitive-football/api/quickstart/) · [Swagger UI](https://mana-peiravian.github.io/forecasting-competitive-football/api/swagger/) · [Final report](report/final_report.pdf)

Leakage-safe pre-match and in-play football forecasting from raw StatsBomb events, Football-Data bookmaker odds, a P1 multilayer-network replication, and a from-scratch P2 FIGS implementation.

## Project

The finished pipeline delivers three tasks:

- **Model 1:** pre-match `P(Away)`, `P(Draw)`, and `P(Home)`; headline metric RPS.
- **Model 2:** expected clipped signed final margin, `home goals − away goals`; headline metric MAE.
- **Model 3:** five-minute outcome and margin updates from a frozen pre-match prior plus events satisfying `event_seconds_exact <= snapshot_seconds`.

La Liga 2015/16 supplies 380 development matches. A chronologically later 34-match, Barcelona-centred StatsBomb 2016/17 subset was cryptographically sealed and evaluated exactly once after the protocol was frozen.

## Headline results

On the 34 final matches, Platt-calibrated random forest obtains pre-match RPS **0.120** (95% bootstrap interval 0.070–0.185) versus **0.131** for the raw market. The paired difference interval crosses zero, so the project does **not** claim to beat bookmakers.

Random forest has the best final Model 2 MAE (**1.479**). Raw in-play gradient boosting obtains overall RPS **0.116** and minute-90 RPS **0.082**; XGBoost has the best Model 3 margin MAE (**1.276**).

StatsBomb exposes only 34 selected 2016/17 matches rather than the full league. Selection bias, incomplete later-season histories, and wide uncertainty are central limitations.

## Installation

Tested with Python 3.11–3.13 (finished environment: Python 3.13.7).

```bash
conda env create -f environment.yml
conda activate football-ml
python -m pip install -e .
python -m pytest
```

Equivalent pip pins are in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

## API quickstart

The repository includes a separately deployable FastAPI backend. GitHub Pages hosts its documentation and static replay; Pages does **not** execute Python.

```bash
uvicorn api.main:app --reload
```

Then open:

- Local Swagger: <http://127.0.0.1:8000/docs>
- Local ReDoc: <http://127.0.0.1:8000/redoc>
- Health: <http://127.0.0.1:8000/api/v1/health>

Example:

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/predict/prematch" \
  -H "Content-Type: application/json" \
  -d '{"match_id":267670}'
```

Public v1 routes:

- `GET /api/v1/health`
- `GET /api/v1/models`
- `GET /api/v1/matches`
- `GET /api/v1/matches/{match_id}`
- `POST /api/v1/predict/prematch/outcome`
- `POST /api/v1/predict/prematch/margin`
- `POST /api/v1/predict/prematch`
- `POST /api/v1/predict/inplay`
- `GET /api/v1/matches/{match_id}/timeline`

Models and feature stores load once at startup; no endpoint retrains. Configuration is environment-driven through the variables shown in `.env.example`. A focused container runtime is provided by `requirements-api.txt` and `Dockerfile`.

## Documentation

```bash
python -m pip install -r requirements-docs.txt
python scripts/build_docs_data.py
python scripts/export_openapi.py
mkdocs serve
```

Strict production build:

```bash
mkdocs build --strict
```

`.github/workflows/docs.yml` regenerates the small static assets/OpenAPI, builds MkDocs Material, uploads the Pages artifact, and deploys through GitHub Actions. It never runs training or the one-shot final evaluator.

## Reproducibility

Validate delivered artifacts and tests:

```bash
football-ml-reproduce --stage validate
```

Rebuild ordinary stages as needed:

```bash
football-ml-reproduce --stage data
football-ml-reproduce --stage features
football-ml-reproduce --stage development
football-ml-reproduce --stage artifacts
```

Network reconstruction is the slowest stage. The final evaluator is intentionally not part of ordinary reproduction and now refuses to rerun because `outputs/final_evaluation/FINAL_TEST_OPENED.json` proves the sealed cohort was scored once.

## Scientific protocol

- Roles: 189 train, 50 validation, 62 calibration, 79 opened pilot, 34 final.
- Classification order: A, D, H; RPS is the headline proper score.
- Margin: home minus away, clipped to `[-5, 5]`.
- Snapshots: 0, 5, …, 90 minutes at exact timestamp boundaries.
- Synthetic sampling: training-only pre-match experiments; never snapshots.
- Calibration: held-out Platt/isotonic; Model 3 by early/middle/late phase.
- Uncertainty: match bootstrap; a sampled match keeps all its snapshots together.

## Repository structure

- `src/ml_project/` — P1, scratch FIGS, features, snapshots, metrics, calibration, models, CLI.
- `api/` — FastAPI application, Pydantic schemas, frozen model registry, small Python client.
- `scripts/` — deterministic research, evaluation, explanation, report, OpenAPI, and docs-data entry points.
- `tests/` — scientific unit tests plus genuine frozen-inference API parity tests.
- `data/normalized/` — relational data store and SHA-256 provenance.
- `models/reproducible/final/` — frozen final predictors, calibrators, and margin mappers.
- `outputs/features/`, `outputs/snapshots/` — target-free inference stores and ordered manifests.
- `outputs/final_evaluation/` — one-shot predictions, metrics, intervals, errors, and marker.
- `docs/`, `mkdocs.yml` — GitHub Pages documentation sources and small static replay assets.
- `report/final_report.pdf` — compiled 12-page project paper.

## P1 and P2 fidelity

P1 uses a 4 × 5 pitch grid, successful-pass intra-layer edges, boundary-aware possession-change inter-layer edges, a 40 × 40 supra-adjacency matrix, the dominant right eigenvector, and published leakage/recovery/switching formulas. It is a methodological StatsBomb replication, not a numerical Opta reproduction.

Scratch FIGS globally compares every current leaf with a fresh root, residualizes against other trees, revisits earlier trees, supports a total split budget, multiclass softmax probabilities, and optional backfitting. Production code never imports `imodels`; that package is optional test-reference evidence only.

## Report and references

- [Final report](report/final_report.pdf)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Football-Data Spain results and odds](https://www.football-data.co.uk/spainm.php)
- Novillo et al., *A multilayer network framework for soccer analysis*, 2024.
- Tan et al., *Fast Interpretable Greedy-Tree Sums*, 2025.

Retain StatsBomb attribution when redistributing or presenting derived work. The project does not currently declare a repository software licence; no licence is implied by this README.

## Honest remaining external items

The initially supplied coursework directory had no historical `.git` metadata, so earlier commits cannot be reconstructed honestly. No verifiable TA sign-off record was supplied; genuine P1/P2 approval remains a human academic action. These limitations are preserved in `docs/git_history_unavailable.txt` and `docs/ta_signoff_required.md`.
