# Installation

The tested project range is Python 3.11–3.13; the finished environment records Python 3.13.7.

## Conda

```bash
conda env create -f environment.yml
conda activate football-ml
python -m pip install -e .
python -m pytest
```

## pip

```bash
python -m venv .venv
```

=== "PowerShell"

    ```powershell
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python -m pytest
    ```

=== "bash"

    ```bash
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install -e .
    python -m pytest
    ```

## Focused environments

For a separately packaged API, `requirements-api.txt` excludes notebooks, plots, SHAP, and data-acquisition libraries while retaining everything required to unpickle the four public predictors and their custom calibrators/mappers.

For the site:

```bash
python -m pip install -r requirements-docs.txt
python scripts/export_openapi.py
python scripts/build_docs_data.py
mkdocs serve
```

`mkdocs serve` opens the local development site at <http://127.0.0.1:8000> by default. If the API is also running, choose a different documentation port such as `mkdocs serve -a 127.0.0.1:8001`.

## Dependency roles

- `requirements.txt`: complete research, API, and test environment.
- `requirements-api.txt`: deployable frozen-inference subset.
- `requirements-docs.txt`: MkDocs, OpenAPI export, and static docs-data generation.
- `environment.yml`: Conda equivalent for the full project.

Pins preserve compatibility with the serialized scikit-learn and XGBoost artifacts. Changing major versions requires a genuine load-and-prediction compatibility test.
