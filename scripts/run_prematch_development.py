"""Development-only Model 1/2 selection, calibration, pilot, and ablations.

This script never opens ``final_targets_sealed.parquet``.  It consumes only the
380-row development target file and refuses any row labelled ``final_test``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml_project.calibration import MulticlassProbabilityCalibrator
from ml_project.config import CLASS_ORDER, MODEL_DIR, NORMALIZED_DIR, OUTPUT_DIR, RANDOM_SEED
from ml_project.metrics import (
    MarginProbabilityMapper,
    classification_metrics,
    regression_metrics,
    reliability_table,
)
from ml_project.modeling import (
    make_classifier,
    make_regressor,
    measured_fit,
    ordered_probabilities,
)


CLASSIFIER_CANDIDATES = {
    "dummy": [{}],
    "svc_rbf": [{"C": 0.3}, {"C": 1.0}],
    "random_forest": [{"max_depth": 5}, {"max_depth": None}],
    "gradient_boosting": [{"max_depth": 1}, {"max_depth": 2}],
    "xgboost": [{"max_depth": 2}, {"max_depth": 3}],
    "lightgbm": [{"num_leaves": 7}, {"num_leaves": 15}],
    "p1_logistic": [{"C": 0.1}, {"C": 1.0}],
    "scratch_figs": [{"max_rules": 8}, {"max_rules": 12}, {"max_rules": 20}],
}

REGRESSOR_CANDIDATES = {
    "dummy": [{}],
    "svr_rbf": [{"C": 0.3}, {"C": 1.0}],
    "kernel_ridge_exact": [{"alpha": 1.0, "gamma": 0.005}, {"alpha": 10.0, "gamma": 0.005}],
    "nystroem_ridge": [{"n_components": 75, "alpha": 1.0}, {"n_components": 125, "alpha": 10.0}],
    "random_forest": [{"max_depth": 5}, {"max_depth": None}],
    "gradient_boosting": [{"max_depth": 1}, {"max_depth": 2}],
    "xgboost": [{"max_depth": 2}, {"max_depth": 3}],
    "lightgbm": [{"num_leaves": 7}, {"num_leaves": 15}],
    "p1_ridge": [{"alpha": 3.0}, {"alpha": 30.0}],
    "scratch_figs": [{"max_rules": 8}, {"max_rules": 12}, {"max_rules": 20}],
}


def feature_group(model_name: str) -> str:
    if model_name.startswith("p1_"):
        return "network"
    return "combined"


def _xy(frame: pd.DataFrame, columns: list[str]):
    return frame[columns].astype(float), frame["outcome"].map({label: i for i, label in enumerate(CLASS_ORDER)}).astype(int)


def select_models(data: pd.DataFrame, groups: dict[str, list[str]], output: Path):
    train, validation = data[data.split_role.eq("train")], data[data.split_role.eq("validation")]
    class_rows, regression_rows, resources = [], [], []
    for name, candidates in CLASSIFIER_CANDIDATES.items():
        group = feature_group(name)
        columns = groups[group]
        X_train, y_train = _xy(train, columns)
        X_validation, _ = _xy(validation, columns)
        for index, params in enumerate(candidates):
            estimator = make_classifier(name, params)
            measurement = measured_fit(estimator, X_train, y_train)
            probabilities = ordered_probabilities(estimator, X_validation)
            metrics = classification_metrics(validation["outcome"], probabilities)
            class_rows.append({"model": name, "candidate": index, "feature_group": group, "params": json.dumps(params, sort_keys=True), **metrics})
            resources.append({"stage": "prematch_selection", "task": "classification", "model": name, "candidate": index, **measurement.__dict__})
    market = validation[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy()
    class_rows.append({"model": "market", "candidate": 0, "feature_group": "market", "params": "{}", **classification_metrics(validation["outcome"], market)})

    for name, candidates in REGRESSOR_CANDIDATES.items():
        group = feature_group(name)
        columns = groups[group]
        X_train = train[columns].astype(float)
        X_validation = validation[columns].astype(float)
        for index, params in enumerate(candidates):
            estimator = make_regressor(name, params)
            measurement = measured_fit(estimator, X_train, train["goal_margin"])
            prediction = np.clip(estimator.predict(X_validation), -5, 5)
            regression_rows.append({"model": name, "candidate": index, "feature_group": group, "params": json.dumps(params, sort_keys=True), **regression_metrics(validation["goal_margin"], prediction)})
            resources.append({"stage": "prematch_selection", "task": "regression", "model": name, "candidate": index, **measurement.__dict__})
    class_results = pd.DataFrame(class_rows)
    reg_results = pd.DataFrame(regression_rows)
    class_results.to_csv(output / "classification_model_selection.csv", index=False)
    reg_results.to_csv(output / "regression_model_selection.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "compute_profile_selection.csv", index=False)
    selected_class = class_results[class_results.model.ne("market")].loc[
        class_results[class_results.model.ne("market")].groupby("model")["rps"].idxmin()
    ]
    selected_reg = reg_results.loc[reg_results.groupby("model")["mae"].idxmin()]
    return selected_class, selected_reg


def fit_calibrate_and_pilot(
    data: pd.DataFrame,
    groups: dict[str, list[str]],
    selected_class: pd.DataFrame,
    selected_reg: pd.DataFrame,
    output: Path,
    model_dir: Path,
):
    fit_rows = data[data.split_role.isin(["train", "validation"])]
    calibration = data[data.split_role.eq("calibration")]
    pilot = data[data.split_role.eq("pilot_holdout")]
    class_metric_rows, class_prediction_rows, reliability_rows = [], [], []
    reg_metric_rows, reg_prediction_rows, conversion_rows, resources = [], [], [], []
    model_dir.mkdir(parents=True, exist_ok=True)

    class_specs = selected_class.to_dict("records")
    class_specs.append({"model": "market", "feature_group": "market", "params": "{}"})
    for spec in class_specs:
        name, group = spec["model"], spec["feature_group"]
        if name == "market":
            p_cal = calibration[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy()
            p_pilot = pilot[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy()
            estimator = None
        else:
            params = json.loads(spec["params"])
            columns = groups[group]
            estimator = make_classifier(name, params)
            X_fit, y_fit = _xy(fit_rows, columns)
            measurement = measured_fit(estimator, X_fit, y_fit)
            resources.append({"stage": "prematch_locked_fit", "task": "classification", "model": name, **measurement.__dict__})
            p_cal = ordered_probabilities(estimator, calibration[columns].astype(float))
            p_pilot = ordered_probabilities(estimator, pilot[columns].astype(float))
            joblib.dump(estimator, model_dir / f"classifier_{name}.joblib")
        calibrators = {"raw": None}
        y_cal_int = calibration["outcome"].map({label: i for i, label in enumerate(CLASS_ORDER)}).to_numpy()
        for method in ("platt", "isotonic"):
            calibrators[method] = MulticlassProbabilityCalibrator(method, RANDOM_SEED).fit(p_cal, y_cal_int)
            joblib.dump(calibrators[method], model_dir / f"calibrator_{name}_{method}.joblib")
        for method, calibrator in calibrators.items():
            for partition, frame, raw in (("calibration", calibration, p_cal), ("pilot_holdout", pilot, p_pilot)):
                probabilities = raw if calibrator is None else calibrator.predict_proba(raw)
                class_metric_rows.append({"model": name, "method": method, "partition": partition, **classification_metrics(frame["outcome"], probabilities)})
                reliability = reliability_table(frame["outcome"], probabilities)
                reliability["model"], reliability["method"], reliability["partition"] = name, method, partition
                reliability_rows.append(reliability)
                if partition == "pilot_holdout":
                    for row_index, match_id in enumerate(frame["match_id"]):
                        class_prediction_rows.append({"match_id": int(match_id), "model": name, "method": method, "actual": frame["outcome"].iloc[row_index], "p_A": probabilities[row_index, 0], "p_D": probabilities[row_index, 1], "p_H": probabilities[row_index, 2]})

    for spec in selected_reg.to_dict("records"):
        name, group, params = spec["model"], spec["feature_group"], json.loads(spec["params"])
        columns = groups[group]
        estimator = make_regressor(name, params)
        measurement = measured_fit(estimator, fit_rows[columns].astype(float), fit_rows["goal_margin"])
        resources.append({"stage": "prematch_locked_fit", "task": "regression", "model": name, **measurement.__dict__})
        pred_cal = np.clip(estimator.predict(calibration[columns].astype(float)), -5, 5)
        pred_pilot = np.clip(estimator.predict(pilot[columns].astype(float)), -5, 5)
        mapper = MarginProbabilityMapper(RANDOM_SEED).fit(pred_cal, calibration["outcome"])
        converted = mapper.predict_proba(pred_pilot)
        reg_metric_rows.append({"model": name, "partition": "pilot_holdout", **regression_metrics(pilot["goal_margin"], pred_pilot)})
        conversion_rows.append({"model": name, **classification_metrics(pilot["outcome"], converted)})
        for row_index, match_id in enumerate(pilot["match_id"]):
            reg_prediction_rows.append({"match_id": int(match_id), "model": name, "actual_margin": pilot["goal_margin"].iloc[row_index], "predicted_margin": pred_pilot[row_index], "p_A": converted[row_index, 0], "p_D": converted[row_index, 1], "p_H": converted[row_index, 2]})
        joblib.dump(estimator, model_dir / f"regressor_{name}.joblib")
        joblib.dump(mapper, model_dir / f"margin_mapper_{name}.joblib")

    pd.DataFrame(class_metric_rows).to_csv(output / "classification_calibration_and_pilot.csv", index=False)
    pd.DataFrame(class_prediction_rows).to_csv(output / "pilot_classification_predictions.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(output / "reliability_bins.csv", index=False)
    pd.DataFrame(reg_metric_rows).to_csv(output / "pilot_regression_metrics.csv", index=False)
    pd.DataFrame(reg_prediction_rows).to_csv(output / "pilot_regression_predictions.csv", index=False)
    pd.DataFrame(conversion_rows).to_csv(output / "pilot_margin_to_outcome_metrics.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "compute_profile_locked_fit.csv", index=False)


def run_ablations(data: pd.DataFrame, groups: dict[str, list[str]], output: Path):
    train, validation = data[data.split_role.eq("train")], data[data.split_role.eq("validation")]
    class_rows, reg_rows = [], []
    for group in ("conventional", "market", "network", "combined"):
        columns = groups[group]
        for model_name in ("random_forest", "scratch_figs"):
            classifier = make_classifier(model_name, {"max_rules": 12} if model_name == "scratch_figs" else {"max_depth": 5})
            _, y_train = _xy(train, columns)
            classifier.fit(train[columns].astype(float), y_train)
            probabilities = ordered_probabilities(classifier, validation[columns].astype(float))
            class_rows.append({"model": model_name, "feature_group": group, **classification_metrics(validation["outcome"], probabilities)})
            regressor = make_regressor(model_name, {"max_rules": 12} if model_name == "scratch_figs" else {"max_depth": 5})
            regressor.fit(train[columns].astype(float), train["goal_margin"])
            prediction = np.clip(regressor.predict(validation[columns].astype(float)), -5, 5)
            reg_rows.append({"model": model_name, "feature_group": group, **regression_metrics(validation["goal_margin"], prediction)})
    pd.DataFrame(class_rows).to_csv(output / "classification_feature_ablation.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(output / "regression_feature_ablation.csv", index=False)


def run_imbalance(data: pd.DataFrame, groups: dict[str, list[str]], output: Path):
    columns = groups["combined"]
    train, validation = data[data.split_role.eq("train")], data[data.split_role.eq("validation")]
    imputer, scaler = SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(train[columns].astype(float)))
    X_validation = scaler.transform(imputer.transform(validation[columns].astype(float)))
    y_train = train["outcome"].map({label: i for i, label in enumerate(CLASS_ORDER)}).to_numpy()
    variants = {
        "vanilla": (None, None),
        "class_weight": (None, "balanced"),
        "smote": (SMOTE(random_state=RANDOM_SEED, k_neighbors=3), None),
        "borderline_smote": (BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=3), None),
        "adasyn": (ADASYN(random_state=RANDOM_SEED, n_neighbors=3), None),
    }
    rows = []
    for name, (resampler, class_weight) in variants.items():
        X_fit, y_fit = (resampler.fit_resample(X_train, y_train) if resampler is not None else (X_train, y_train))
        model = LogisticRegression(max_iter=3000, class_weight=class_weight, random_state=RANDOM_SEED).fit(X_fit, y_fit)
        rows.append({"variant": name, "training_rows": len(y_fit), **classification_metrics(validation["outcome"], model.predict_proba(X_validation))})
    pd.DataFrame(rows).to_csv(output / "imbalance_training_only.csv", index=False)


def kernel_scaling(data: pd.DataFrame, groups: dict[str, list[str]], output: Path):
    train = data[data.split_role.eq("train")].sort_values(["clean_date", "match_id"])
    columns = groups["combined"]
    rows = []
    for n in (50, 100, len(train)):
        subset = train.iloc[:n]
        for name in ("kernel_ridge_exact", "nystroem_ridge"):
            model = make_regressor(name, {"n_components": min(75, n)} if name == "nystroem_ridge" else {})
            measurement = measured_fit(model, subset[columns].astype(float), subset["goal_margin"])
            rows.append({"model": name, "n_samples": n, "kernel_matrix_elements": n * n if name == "kernel_ridge_exact" else n * min(75, n), **measurement.__dict__})
    pd.DataFrame(rows).to_csv(output / "kernel_scaling.csv", index=False)


def main():
    output = OUTPUT_DIR / "experiments" / "prematch"
    output.mkdir(parents=True, exist_ok=True)
    model_dir = MODEL_DIR / "development" / "prematch"
    features = pd.read_parquet(OUTPUT_DIR / "features" / "prematch_features.parquet")
    if features["split_role"].eq("final_test").sum() == 0:
        raise AssertionError("Expected sealed final feature rows")
    development_features = features[features["split_role"].ne("final_test")].copy()
    targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")
    data = development_features.merge(targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="one_to_one")
    if data["split_role"].eq("final_test").any() or len(data) != 380:
        raise AssertionError("Development experiment boundary violated")
    manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    groups = manifest["groups"]
    selected_class, selected_reg = select_models(data, groups, output)
    selected_class.to_csv(output / "locked_classifier_specs.csv", index=False)
    selected_reg.to_csv(output / "locked_regressor_specs.csv", index=False)
    fit_calibrate_and_pilot(data, groups, selected_class, selected_reg, output, model_dir)
    run_ablations(data, groups, output)
    run_imbalance(data, groups, output)
    kernel_scaling(data, groups, output)
    protocol = {
        "class_order": list(CLASS_ORDER),
        "fit_roles": ["train", "validation"],
        "calibration_role": "calibration",
        "opened_benchmark_role": "pilot_holdout",
        "forbidden_role": "final_test",
        "selection_metric_classification": "RPS",
        "selection_metric_regression": "MAE",
        "final_calibration_method_fixed_a_priori": "platt",
        "synthetic_sampling_scope": "prematch training partition only",
    }
    (output / "development_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Prematch development experiments complete; final targets were not opened.")


if __name__ == "__main__":
    main()

