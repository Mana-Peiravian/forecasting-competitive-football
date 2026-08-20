"""Development-only Model 3 experiments with match-grouped snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_project.calibration import PhaseCalibrator
from ml_project.config import CLASS_ORDER, MODEL_DIR, NORMALIZED_DIR, OUTPUT_DIR, RANDOM_SEED
from ml_project.metrics import MarginProbabilityMapper, classification_metrics, regression_metrics, reliability_table
from ml_project.modeling import make_classifier, make_regressor, measured_fit, ordered_probabilities

CLASS_SPECS = {
    "dummy": {},
    "random_forest": {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 5},
    "gradient_boosting": {"n_estimators": 75, "max_depth": 2},
    "xgboost": {"n_estimators": 100, "max_depth": 3},
    "lightgbm": {"n_estimators": 100, "num_leaves": 15},
    "scratch_figs": {"max_rules": 12, "max_trees": 5},
}
REG_SPECS = {
    "dummy": {},
    "random_forest": {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 5},
    "gradient_boosting": {"n_estimators": 75, "max_depth": 2},
    "xgboost": {"n_estimators": 100, "max_depth": 3},
    "lightgbm": {"n_estimators": 100, "num_leaves": 15},
    "scratch_figs": {"max_rules": 12, "max_trees": 5},
}


def y_int(frame: pd.DataFrame) -> np.ndarray:
    return frame["outcome"].map({label: i for i, label in enumerate(CLASS_ORDER)}).to_numpy()


def minute_metrics(frame: pd.DataFrame, probabilities: np.ndarray, model: str, method: str) -> pd.DataFrame:
    rows = []
    for minute in sorted(frame["snapshot_minute"].unique()):
        mask = frame["snapshot_minute"].eq(minute).to_numpy()
        rows.append({"model": model, "method": method, "snapshot_minute": int(minute), "n_matches": int(mask.sum()), **classification_metrics(frame.loc[mask, "outcome"], probabilities[mask])})
    return pd.DataFrame(rows)


def minute_regression_metrics(frame: pd.DataFrame, prediction: np.ndarray, model: str) -> pd.DataFrame:
    rows = []
    for minute in sorted(frame["snapshot_minute"].unique()):
        mask = frame["snapshot_minute"].eq(minute).to_numpy()
        rows.append({"model": model, "snapshot_minute": int(minute), "n_matches": int(mask.sum()), **regression_metrics(frame.loc[mask, "goal_margin"], prediction[mask])})
    return pd.DataFrame(rows)


def main():
    output = OUTPUT_DIR / "experiments" / "snapshots"
    model_dir = MODEL_DIR / "development" / "snapshots"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshots = pd.read_parquet(OUTPUT_DIR / "snapshots" / "snapshot_features.parquet")
    snapshots = snapshots[snapshots["split_role"].ne("final_test")].copy()
    targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")
    data = snapshots.merge(targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="many_to_one")
    if data["split_role"].eq("final_test").any() or data["match_id"].nunique() != 380:
        raise AssertionError("Snapshot development boundary violated")
    manifest = json.loads((OUTPUT_DIR / "snapshots" / "snapshot_manifest.json").read_text())
    columns = manifest["groups"]["combined"]
    train = data[data.split_role.eq("train")]
    validation = data[data.split_role.eq("validation")]
    fit_rows = data[data.split_role.isin(["train", "validation"])]
    calibration = data[data.split_role.eq("calibration")]
    pilot = data[data.split_role.eq("pilot_holdout")]

    validation_class_rows, validation_reg_rows, pilot_class_rows, pilot_reg_rows = [], [], [], []
    pilot_prediction_rows, minute_class_frames, minute_reg_frames, reliability_frames, resources = [], [], [], [], []

    # Frozen market probabilities are the mandatory no-update comparator.
    for partition, frame in (("validation", validation), ("pilot_holdout", pilot)):
        p = frame[["prematch_market_prob_A", "prematch_market_prob_D", "prematch_market_prob_H"]].to_numpy()
        target_rows = validation_class_rows if partition == "validation" else pilot_class_rows
        target_rows.append({"model": "frozen_market", "method": "raw", "partition": partition, **classification_metrics(frame["outcome"], p)})
        if partition == "pilot_holdout":
            minute_class_frames.append(minute_metrics(frame, p, "frozen_market", "raw"))

    for name, params in CLASS_SPECS.items():
        selection_model = make_classifier(name, params)
        measurement = measured_fit(selection_model, train[columns].astype(float), y_int(train))
        resources.append({"stage": "snapshot_selection", "task": "classification", "model": name, **measurement.__dict__})
        p_validation = ordered_probabilities(selection_model, validation[columns].astype(float))
        validation_class_rows.append({"model": name, "method": "raw", "partition": "validation", **classification_metrics(validation["outcome"], p_validation)})

        model = make_classifier(name, params)
        measurement = measured_fit(model, fit_rows[columns].astype(float), y_int(fit_rows))
        resources.append({"stage": "snapshot_locked_fit", "task": "classification", "model": name, **measurement.__dict__})
        p_cal = ordered_probabilities(model, calibration[columns].astype(float))
        p_pilot = ordered_probabilities(model, pilot[columns].astype(float))
        calibrators = {"raw": None}
        for method in ("platt", "isotonic"):
            calibrators[method] = PhaseCalibrator(method, RANDOM_SEED).fit(
                p_cal, y_int(calibration), calibration["snapshot_minute"].to_numpy()
            )
            joblib.dump(calibrators[method], model_dir / f"phase_calibrator_{name}_{method}.joblib")
        for method, calibrator in calibrators.items():
            probabilities = p_pilot if calibrator is None else calibrator.predict_proba(
                p_pilot, pilot["snapshot_minute"].to_numpy()
            )
            pilot_class_rows.append({"model": name, "method": method, "partition": "pilot_holdout", **classification_metrics(pilot["outcome"], probabilities)})
            minute_class_frames.append(minute_metrics(pilot, probabilities, name, method))
            reliability = reliability_table(pilot["outcome"], probabilities)
            reliability["model"], reliability["method"] = name, method
            reliability_frames.append(reliability)
            for row_index, row in enumerate(pilot[["match_id", "snapshot_minute", "outcome"]].itertuples(index=False)):
                pilot_prediction_rows.append({"match_id": int(row.match_id), "snapshot_minute": int(row.snapshot_minute), "actual": row.outcome, "model": name, "method": method, "p_A": probabilities[row_index, 0], "p_D": probabilities[row_index, 1], "p_H": probabilities[row_index, 2]})
        joblib.dump(model, model_dir / f"classifier_{name}.joblib")

    for name, params in REG_SPECS.items():
        selection_model = make_regressor(name, params)
        measurement = measured_fit(selection_model, train[columns].astype(float), train["goal_margin"])
        resources.append({"stage": "snapshot_selection", "task": "regression", "model": name, **measurement.__dict__})
        prediction_validation = np.clip(selection_model.predict(validation[columns].astype(float)), -5, 5)
        validation_reg_rows.append({"model": name, "partition": "validation", **regression_metrics(validation["goal_margin"], prediction_validation)})

        model = make_regressor(name, params)
        measurement = measured_fit(model, fit_rows[columns].astype(float), fit_rows["goal_margin"])
        resources.append({"stage": "snapshot_locked_fit", "task": "regression", "model": name, **measurement.__dict__})
        pred_cal = np.clip(model.predict(calibration[columns].astype(float)), -5, 5)
        pred_pilot = np.clip(model.predict(pilot[columns].astype(float)), -5, 5)
        mapper = MarginProbabilityMapper(RANDOM_SEED).fit(pred_cal, calibration["outcome"])
        converted = mapper.predict_proba(pred_pilot)
        pilot_reg_rows.append({"model": name, "partition": "pilot_holdout", **regression_metrics(pilot["goal_margin"], pred_pilot), **{f"converted_{key}": value for key, value in classification_metrics(pilot["outcome"], converted).items()}})
        minute_reg_frames.append(minute_regression_metrics(pilot, pred_pilot, name))
        joblib.dump(model, model_dir / f"regressor_{name}.joblib")
        joblib.dump(mapper, model_dir / f"margin_mapper_{name}.joblib")

    # Snapshot imbalance comparison: class weights only. No snapshot is synthetic.
    imbalance_rows = []
    for variant, class_weight in (("vanilla", None), ("class_weight", "balanced")):
        params = {**CLASS_SPECS["random_forest"], "class_weight": class_weight}
        model = make_classifier("random_forest", params).fit(train[columns].astype(float), y_int(train))
        p = ordered_probabilities(model, validation[columns].astype(float))
        imbalance_rows.append({"variant": variant, "synthetic_snapshots": 0, **classification_metrics(validation["outcome"], p)})

    pd.DataFrame(validation_class_rows).to_csv(output / "classification_validation.csv", index=False)
    pd.DataFrame(validation_reg_rows).to_csv(output / "regression_validation.csv", index=False)
    pd.DataFrame(pilot_class_rows).to_csv(output / "classification_calibration_and_pilot.csv", index=False)
    pd.DataFrame(pilot_reg_rows).to_csv(output / "regression_pilot.csv", index=False)
    pd.DataFrame(pilot_prediction_rows).to_parquet(output / "pilot_classification_predictions.parquet", index=False)
    pd.concat(minute_class_frames, ignore_index=True).to_csv(output / "classification_by_minute.csv", index=False)
    pd.concat(minute_reg_frames, ignore_index=True).to_csv(output / "regression_by_minute.csv", index=False)
    pd.concat(reliability_frames, ignore_index=True).to_csv(output / "reliability_bins.csv", index=False)
    pd.DataFrame(imbalance_rows).to_csv(output / "imbalance_no_synthetic_snapshots.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "compute_profile.csv", index=False)
    specs = {
        "classification": CLASS_SPECS,
        "regression": REG_SPECS,
        "features": "combined",
        "calibration": "phase-specific; early <=30, middle <=60, late >60",
        "final_method_fixed_a_priori": "platt",
        "synthetic_snapshots": False,
    }
    (output / "locked_snapshot_specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    print("Snapshot development experiments complete; final targets were not opened.")


if __name__ == "__main__":
    main()

