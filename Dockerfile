FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src/ml_project ./src/ml_project
COPY outputs/features/feature_manifest.json outputs/features/prematch_features.parquet ./outputs/features/
COPY outputs/snapshots/snapshot_manifest.json outputs/snapshots/snapshot_features.parquet ./outputs/snapshots/

COPY models/reproducible/final/prematch_classifier_random_forest.joblib \
     models/reproducible/final/prematch_calibrator_random_forest_platt.joblib \
     models/reproducible/final/prematch_regressor_random_forest.joblib \
     models/reproducible/final/prematch_margin_mapper_random_forest.joblib \
     models/reproducible/final/snapshot_classifier_gradient_boosting.joblib \
     models/reproducible/final/snapshot_phase_calibrator_gradient_boosting_platt.joblib \
     models/reproducible/final/snapshot_regressor_xgboost.joblib \
     models/reproducible/final/snapshot_margin_mapper_xgboost.joblib \
     ./models/reproducible/final/

EXPOSE 8000

CMD ["python", "-m", "api"]
