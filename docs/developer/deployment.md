# Deployment

## GitHub Pages

`.github/workflows/docs.yml` builds and deploys the static site whenever relevant files are pushed to `main`, and can be run manually. It:

1. checks out the repository;
2. sets up Python 3.13;
3. installs `requirements-docs.txt`;
4. regenerates static result/demo assets and OpenAPI;
5. verifies schema synchronization and runs `mkdocs build --strict`;
6. uploads `site/` as a Pages artifact;
7. deploys through the `github-pages` environment.

It does not run data acquisition, feature construction, training, development experiments, or the final evaluator.

In GitHub, choose **Repository → Settings → Pages → Build and deployment → Source: GitHub Actions**. No generated HTML needs to be copied or uploaded manually.

Expected project-site URL:

```text
https://mana-peiravian.github.io/forecasting-competitive-football/
```

`site_url` includes the repository path and site links are relative, so assets work below that subpath.

## Local API

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Environment variables are documented in `.env.example`. Important production settings are artifact/feature paths, `PUBLIC_API_BASE_URL`, and an explicit comma-separated `ALLOWED_ORIGINS` list.

## Container

```bash
docker build -t football-forecast-api .
docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://mana-peiravian.github.io" \
  football-forecast-api
```

The Dockerfile copies the four public predictors, four calibrators/mappers, ordered manifests, and target-free feature stores. It excludes raw events and experimental archives.

## Real hosted backend

A callable public API requires a Python-capable host separate from GitHub Pages. General requirements are:

- Python 3.11–3.13 compatible with frozen dependencies;
- sufficient memory for pandas feature stores and loaded estimators;
- the eight selected frozen artifacts and two target-free stores;
- HTTPS and a stable public URL;
- exact CORS origins, health checks, private logs, request limits, and restart policy;
- environment configuration rather than source edits.

No cloud vendor or public backend URL is invented here. After deployment, set `PUBLIC_API_BASE_URL`, regenerate `docs/api/openapi.json`, and redeploy Pages so Swagger points at the real service.

## CORS

The API does not use wildcard origins. The default includes the discovered GitHub Pages origin and local development origins; replace or extend it through `ALLOWED_ORIGINS`. Credentials are disabled in the current unauthenticated v1 service.
