# Repository structure

```text
forecasting-competitive-football/
├── api/                         # FastAPI app, schemas, registry, client
├── data/
│   ├── normalized/              # relational store and provenance
│   ├── processed/               # historical coursework artifacts
│   └── raw_sources/             # retained public odds inputs
├── docs/                        # MkDocs sources and small Pages assets
├── models/
│   └── reproducible/            # frozen development/final Joblib artifacts
├── notebooks/                   # seven received/reproducible notebooks
├── outputs/
│   ├── features/                # target-free pre-match store + manifest
│   ├── snapshots/               # target-free in-play store + manifest
│   ├── final_evaluation/        # one-shot predictions and metrics
│   ├── figures/                 # scientific plots
│   └── explainability/          # SHAP and error-analysis tables
├── report/                      # LaTeX source and final PDF
├── scripts/                     # deterministic research/docs entry points
├── src/ml_project/              # production ML and feature package
├── tests/                       # scientific and API tests
├── mkdocs.yml                   # site configuration and navigation
├── requirements-docs.txt        # documentation build dependencies
└── .github/workflows/docs.yml   # Pages build and deployment
```

## Generated versus source-controlled

`docs/api/openapi.json`, generated result tables, selected web figures, the report copy, and static replay JSON are generated from finished source artifacts. The generating scripts are cheap and never train models. Model files stay under `models/`; none are copied into Pages.

## Historical materials

The notebooks and older `data/processed/figs_results/` files remain useful provenance, but the public API and headline site use the later frozen final pipeline and 2016/17 one-shot evaluation. The 79-match tail in older notebooks is correctly labelled `pilot_holdout`, not final test.

## Start points

- Research orchestration: `football-ml-reproduce --stage validate`
- API: `uvicorn api.main:app --reload`
- Documentation: `mkdocs serve`
- OpenAPI: `python scripts/export_openapi.py`
- Static docs data: `python scripts/build_docs_data.py`
