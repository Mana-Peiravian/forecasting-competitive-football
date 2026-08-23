# Adding a new model

A new public model is more than another Joblib file. It must remain aligned with the scientific pipeline and API contract.

## 1. Define the task boundary

State what the prediction represents, which information exists at prediction time, its target, class order or units, and headline metric. For snapshots, specify the exact time predicate.

## 2. Add it to the research protocol

Use temporal development roles for selection. Fit calibration/mapping only on the calibration role. Do not use final labels for hyperparameters, features, thresholds, or public-model selection.

## 3. Freeze a complete inference bundle

Retain:

- estimator;
- imputer/scaler/preprocessor inside the fitted pipeline;
- exact ordered feature manifest;
- calibrator or mapper where applicable;
- safe model/version metadata;
- compatibility versions.

Do not implement a simplified parallel feature calculation inside the API.

## 4. Extend the registry

Load the artifact once in `api/model_registry.py`, validate its `feature_names_in_` count/order, and expose a repository-relative artifact identifier. Never return an absolute path.

## 5. Extend schemas and routes

Add typed request/response models in `api/schemas.py`, versioned routes in `api/main.py`, field descriptions, constraints, examples, response errors, and explicit calibration semantics. Avoid arbitrary uploaded feature vectors unless dimensions, names, finiteness, and security are rigorously controlled.

## 6. Add parity tests

At least one end-to-end test should compare API output with a saved frozen prediction. Test probability normalization, invalid inputs, missing matches, response models, and OpenAPI synchronization.

## 7. Regenerate public artifacts

```bash
python scripts/export_openapi.py
python scripts/build_docs_data.py
python -m pytest
mkdocs build --strict
```

Update scientific documentation and limitations. Do not overwrite the historical one-shot result or imply the model was evaluated under a protocol it did not follow.
