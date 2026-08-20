"""One-shot evaluation on the sealed 2016/17 StatsBomb subset.

The script validates the pre-final lock, fits every locked model without final
labels, writes an irreversible open marker, reads final labels exactly once,
and creates all final metrics/predictions. It refuses a second run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_project.calibration import MulticlassProbabilityCalibrator, PhaseCalibrator
from ml_project.config import CLASS_ORDER, MODEL_DIR, NORMALIZED_DIR, OUTPUT_DIR, RANDOM_SEED
from ml_project.metrics import MarginProbabilityMapper, classification_metrics, regression_metrics, reliability_table
from ml_project.modeling import make_classifier, make_regressor, measured_fit, ordered_probabilities

sns.set_theme(style="whitegrid", context="paper")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock(root: Path) -> dict:
    lock_path = OUTPUT_DIR / "model_lock_before_final.json"
    if not lock_path.exists():
        raise RuntimeError("Final evaluation requires outputs/model_lock_before_final.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in lock["locked_file_hashes"].items():
        path = root / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Locked file changed after protocol freeze: {relative}")
    return lock


def y_indices(labels: pd.Series) -> np.ndarray:
    return labels.map({label: index for index, label in enumerate(CLASS_ORDER)}).to_numpy()


def individual_rps(labels: pd.Series, probabilities: np.ndarray) -> np.ndarray:
    observed = np.eye(3)[y_indices(labels)]
    return np.sum((np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]) ** 2, axis=1) / 2


def bootstrap_interval(values_function, n: int, rng: np.random.Generator, draws: int = 2000):
    values = np.empty(draws)
    for draw in range(draws):
        indices = rng.integers(0, n, n)
        values[draw] = values_function(indices)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def match_group_bootstrap_rps(labels: pd.Series, probabilities: np.ndarray, match_ids: pd.Series, draws=1000):
    unique = np.asarray(pd.unique(match_ids))
    positions = {match_id: np.flatnonzero(match_ids.to_numpy() == match_id) for match_id in unique}
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[match_id] for match_id in sampled])
        values.append(classification_metrics(labels.iloc[indices], probabilities[indices])["rps"])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    final_dir = OUTPUT_DIR / "final_evaluation"
    marker_path = final_dir / "FINAL_TEST_OPENED.json"
    if marker_path.exists():
        raise RuntimeError(f"Final test was already opened; refusing re-run: {marker_path}")
    lock = validate_lock(project_root)
    final_dir.mkdir(parents=True, exist_ok=True)
    model_dir = MODEL_DIR / "final"
    model_dir.mkdir(parents=True, exist_ok=True)

    feature_manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    groups = feature_manifest["groups"]
    features = pd.read_parquet(OUTPUT_DIR / "features" / "prematch_features.parquet")
    development_features = features[features.split_role.ne("final_test")].copy()
    final_features = features[features.split_role.eq("final_test")].sort_values(["clean_date", "match_id"]).reset_index(drop=True)
    development_targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")
    development = development_features.merge(
        development_targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="one_to_one"
    )
    base = development[development.split_role.isin(["train", "validation", "pilot_holdout"])]
    calibration = development[development.split_role.eq("calibration")]
    class_specs = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "locked_classifier_specs.csv")
    reg_specs = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "locked_regressor_specs.csv")
    prematch_class_forecasts, prematch_reg_forecasts, resources = {}, {}, []

    for spec in class_specs.to_dict("records"):
        name, group, params = spec["model"], spec["feature_group"], json.loads(spec["params"])
        columns = groups[group]
        model = make_classifier(name, params)
        measurement = measured_fit(model, base[columns].astype(float), y_indices(base.outcome))
        resources.append({"stage": "final_fit", "task": "prematch_classification", "model": name, **measurement.__dict__})
        p_cal = ordered_probabilities(model, calibration[columns].astype(float))
        p_final = ordered_probabilities(model, final_features[columns].astype(float))
        calibrator = MulticlassProbabilityCalibrator("platt", RANDOM_SEED).fit(p_cal, y_indices(calibration.outcome))
        prematch_class_forecasts[name] = {"raw": p_final, "platt": calibrator.predict_proba(p_final)}
        joblib.dump(model, model_dir / f"prematch_classifier_{name}.joblib")
        joblib.dump(calibrator, model_dir / f"prematch_calibrator_{name}_platt.joblib")
    p_cal_market = calibration[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy()
    p_final_market = final_features[["market_prob_A", "market_prob_D", "market_prob_H"]].to_numpy()
    market_calibrator = MulticlassProbabilityCalibrator("platt", RANDOM_SEED).fit(p_cal_market, y_indices(calibration.outcome))
    prematch_class_forecasts["market"] = {"raw": p_final_market, "platt": market_calibrator.predict_proba(p_final_market)}

    for spec in reg_specs.to_dict("records"):
        name, group, params = spec["model"], spec["feature_group"], json.loads(spec["params"])
        columns = groups[group]
        model = make_regressor(name, params)
        measurement = measured_fit(model, base[columns].astype(float), base.goal_margin)
        resources.append({"stage": "final_fit", "task": "prematch_regression", "model": name, **measurement.__dict__})
        pred_cal = np.clip(model.predict(calibration[columns].astype(float)), -5, 5)
        pred_final = np.clip(model.predict(final_features[columns].astype(float)), -5, 5)
        mapper = MarginProbabilityMapper(RANDOM_SEED).fit(pred_cal, calibration.outcome)
        prematch_reg_forecasts[name] = {"margin": pred_final, "probabilities": mapper.predict_proba(pred_final)}
        joblib.dump(model, model_dir / f"prematch_regressor_{name}.joblib")
        joblib.dump(mapper, model_dir / f"prematch_margin_mapper_{name}.joblib")

    snapshot_manifest = json.loads((OUTPUT_DIR / "snapshots" / "snapshot_manifest.json").read_text())
    snapshot_columns = snapshot_manifest["groups"]["combined"]
    snapshots = pd.read_parquet(OUTPUT_DIR / "snapshots" / "snapshot_features.parquet")
    development_snapshots = snapshots[snapshots.split_role.ne("final_test")].merge(
        development_targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="many_to_one"
    )
    final_snapshots = snapshots[snapshots.split_role.eq("final_test")].sort_values(["clean_date", "match_id", "snapshot_minute"]).reset_index(drop=True)
    snapshot_base = development_snapshots[development_snapshots.split_role.isin(["train", "validation", "pilot_holdout"])]
    snapshot_calibration = development_snapshots[development_snapshots.split_role.eq("calibration")]
    snapshot_specs = json.loads((OUTPUT_DIR / "experiments" / "snapshots" / "locked_snapshot_specs.json").read_text())
    snapshot_class_forecasts, snapshot_reg_forecasts = {}, {}
    for name, params in snapshot_specs["classification"].items():
        model = make_classifier(name, params)
        measurement = measured_fit(model, snapshot_base[snapshot_columns].astype(float), y_indices(snapshot_base.outcome))
        resources.append({"stage": "final_fit", "task": "snapshot_classification", "model": name, **measurement.__dict__})
        p_cal = ordered_probabilities(model, snapshot_calibration[snapshot_columns].astype(float))
        p_final = ordered_probabilities(model, final_snapshots[snapshot_columns].astype(float))
        calibrator = PhaseCalibrator("platt", RANDOM_SEED).fit(
            p_cal, y_indices(snapshot_calibration.outcome), snapshot_calibration.snapshot_minute.to_numpy()
        )
        snapshot_class_forecasts[name] = {
            "raw": p_final,
            "platt": calibrator.predict_proba(p_final, final_snapshots.snapshot_minute.to_numpy()),
        }
        joblib.dump(model, model_dir / f"snapshot_classifier_{name}.joblib")
        joblib.dump(calibrator, model_dir / f"snapshot_phase_calibrator_{name}_platt.joblib")
    snapshot_class_forecasts["frozen_market"] = {
        "raw": final_snapshots[["prematch_market_prob_A", "prematch_market_prob_D", "prematch_market_prob_H"]].to_numpy()
    }
    best_learned = class_specs.sort_values("rps").iloc[0].model
    lookup = {
        int(match_id): prematch_class_forecasts[best_learned]["platt"][index]
        for index, match_id in enumerate(final_features.match_id)
    }
    snapshot_class_forecasts["frozen_best_prematch"] = {
        "platt": np.vstack([lookup[int(match_id)] for match_id in final_snapshots.match_id])
    }

    for name, params in snapshot_specs["regression"].items():
        model = make_regressor(name, params)
        measurement = measured_fit(model, snapshot_base[snapshot_columns].astype(float), snapshot_base.goal_margin)
        resources.append({"stage": "final_fit", "task": "snapshot_regression", "model": name, **measurement.__dict__})
        pred_cal = np.clip(model.predict(snapshot_calibration[snapshot_columns].astype(float)), -5, 5)
        pred_final = np.clip(model.predict(final_snapshots[snapshot_columns].astype(float)), -5, 5)
        mapper = MarginProbabilityMapper(RANDOM_SEED).fit(pred_cal, snapshot_calibration.outcome)
        snapshot_reg_forecasts[name] = {"margin": pred_final, "probabilities": mapper.predict_proba(pred_final)}
        joblib.dump(model, model_dir / f"snapshot_regressor_{name}.joblib")
        joblib.dump(mapper, model_dir / f"snapshot_margin_mapper_{name}.joblib")

    # This is the sole transition from sealed forecasts to opened final labels.
    marker = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "opened_evaluation_in_progress",
        "lock_sha256": sha256(OUTPUT_DIR / "model_lock_before_final.json"),
        "target_sha256": sha256(NORMALIZED_DIR / "final_targets_sealed.parquet"),
        "expected_matches": len(final_features),
    }
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    final_targets = pd.read_parquet(NORMALIZED_DIR / "final_targets_sealed.parquet")
    final_targets = final_targets.set_index("match_id").loc[final_features.match_id].reset_index()
    if len(final_targets) != 34 or set(final_targets.match_id) != set(final_features.match_id):
        raise AssertionError("Final target/feature identity mismatch")
    final_snapshot_targets = final_targets.set_index("match_id").loc[final_snapshots.match_id].reset_index()

    rng = np.random.default_rng(RANDOM_SEED)
    class_rows, class_predictions, reliability_frames = [], [], []
    for name, methods in prematch_class_forecasts.items():
        for method, probabilities in methods.items():
            metrics = classification_metrics(final_targets.outcome, probabilities)
            lower, upper = bootstrap_interval(
                lambda indices: classification_metrics(final_targets.outcome.iloc[indices], probabilities[indices])["rps"],
                len(final_targets), rng,
            )
            class_rows.append({"model": name, "method": method, "n_matches": len(final_targets), "rps_ci_lower": lower, "rps_ci_upper": upper, **metrics})
            reliability = reliability_table(final_targets.outcome, probabilities)
            reliability["model"], reliability["method"] = name, method
            reliability_frames.append(reliability)
            for index, match_id in enumerate(final_features.match_id):
                class_predictions.append({"match_id": int(match_id), "model": name, "method": method, "actual": final_targets.outcome.iloc[index], "p_A": probabilities[index, 0], "p_D": probabilities[index, 1], "p_H": probabilities[index, 2]})

    reg_rows, reg_predictions = [], []
    for name, forecast in prematch_reg_forecasts.items():
        metrics = regression_metrics(final_targets.goal_margin, forecast["margin"])
        lower, upper = bootstrap_interval(
            lambda indices: regression_metrics(final_targets.goal_margin.iloc[indices], forecast["margin"][indices])["mae"],
            len(final_targets), rng,
        )
        converted = classification_metrics(final_targets.outcome, forecast["probabilities"])
        reg_rows.append({"model": name, "n_matches": len(final_targets), "mae_ci_lower": lower, "mae_ci_upper": upper, **metrics, **{f"converted_{key}": value for key, value in converted.items()}})
        for index, match_id in enumerate(final_features.match_id):
            reg_predictions.append({"match_id": int(match_id), "model": name, "actual_margin": final_targets.goal_margin.iloc[index], "predicted_margin": forecast["margin"][index], "p_A": forecast["probabilities"][index, 0], "p_D": forecast["probabilities"][index, 1], "p_H": forecast["probabilities"][index, 2]})

    snapshot_class_rows, snapshot_class_predictions, minute_rows = [], [], []
    for name, methods in snapshot_class_forecasts.items():
        for method, probabilities in methods.items():
            metrics = classification_metrics(final_snapshot_targets.outcome, probabilities)
            lower, upper = match_group_bootstrap_rps(final_snapshot_targets.outcome, probabilities, final_snapshots.match_id)
            snapshot_class_rows.append({"model": name, "method": method, "n_matches": final_snapshots.match_id.nunique(), "n_snapshots": len(final_snapshots), "rps_ci_lower": lower, "rps_ci_upper": upper, **metrics})
            for minute in sorted(final_snapshots.snapshot_minute.unique()):
                mask = final_snapshots.snapshot_minute.eq(minute).to_numpy()
                minute_rows.append({"model": name, "method": method, "snapshot_minute": int(minute), "n_matches": int(mask.sum()), **classification_metrics(final_snapshot_targets.loc[mask, "outcome"], probabilities[mask])})
            for index, row in enumerate(final_snapshots[["match_id", "snapshot_minute"]].itertuples(index=False)):
                snapshot_class_predictions.append({"match_id": int(row.match_id), "snapshot_minute": int(row.snapshot_minute), "model": name, "method": method, "actual": final_snapshot_targets.outcome.iloc[index], "p_A": probabilities[index, 0], "p_D": probabilities[index, 1], "p_H": probabilities[index, 2]})

    snapshot_reg_rows, snapshot_reg_predictions, snapshot_reg_minute_rows = [], [], []
    for name, forecast in snapshot_reg_forecasts.items():
        snapshot_reg_rows.append({"model": name, "n_matches": final_snapshots.match_id.nunique(), "n_snapshots": len(final_snapshots), **regression_metrics(final_snapshot_targets.goal_margin, forecast["margin"]), **{f"converted_{key}": value for key, value in classification_metrics(final_snapshot_targets.outcome, forecast["probabilities"]).items()}})
        for minute in sorted(final_snapshots.snapshot_minute.unique()):
            mask = final_snapshots.snapshot_minute.eq(minute).to_numpy()
            snapshot_reg_minute_rows.append({"model": name, "snapshot_minute": int(minute), "n_matches": int(mask.sum()), **regression_metrics(final_snapshot_targets.loc[mask, "goal_margin"], forecast["margin"][mask])})
        for index, row in enumerate(final_snapshots[["match_id", "snapshot_minute"]].itertuples(index=False)):
            snapshot_reg_predictions.append({"match_id": int(row.match_id), "snapshot_minute": int(row.snapshot_minute), "model": name, "actual_margin": final_snapshot_targets.goal_margin.iloc[index], "predicted_margin": forecast["margin"][index]})

    pd.DataFrame(class_rows).to_csv(final_dir / "prematch_classification_metrics.csv", index=False)
    pd.DataFrame(class_predictions).to_csv(final_dir / "prematch_classification_predictions.csv", index=False)
    pd.concat(reliability_frames, ignore_index=True).to_csv(final_dir / "prematch_reliability_bins.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(final_dir / "prematch_regression_metrics.csv", index=False)
    pd.DataFrame(reg_predictions).to_csv(final_dir / "prematch_regression_predictions.csv", index=False)
    pd.DataFrame(snapshot_class_rows).to_csv(final_dir / "snapshot_classification_metrics.csv", index=False)
    pd.DataFrame(minute_rows).to_csv(final_dir / "snapshot_classification_by_minute.csv", index=False)
    pd.DataFrame(snapshot_class_predictions).to_parquet(final_dir / "snapshot_classification_predictions.parquet", index=False)
    pd.DataFrame(snapshot_reg_rows).to_csv(final_dir / "snapshot_regression_metrics.csv", index=False)
    pd.DataFrame(snapshot_reg_minute_rows).to_csv(final_dir / "snapshot_regression_by_minute.csv", index=False)
    pd.DataFrame(snapshot_reg_predictions).to_parquet(final_dir / "snapshot_regression_predictions.parquet", index=False)
    pd.DataFrame(resources).to_csv(final_dir / "compute_profile_final_fit.csv", index=False)

    # Paired final comparisons against the raw market.
    market_loss = individual_rps(final_targets.outcome, prematch_class_forecasts["market"]["raw"])
    paired_rows = []
    for name, methods in prematch_class_forecasts.items():
        for method, probabilities in methods.items():
            difference = individual_rps(final_targets.outcome, probabilities) - market_loss
            lower, upper = bootstrap_interval(lambda indices: difference[indices].mean(), len(difference), rng)
            paired_rows.append({"model": name, "method": method, "mean_rps_difference_vs_raw_market": difference.mean(), "ci_lower": lower, "ci_upper": upper})
    pd.DataFrame(paired_rows).to_csv(final_dir / "paired_rps_differences_vs_market.csv", index=False)

    # Final error slices for the best learned locked classifier.
    learned = class_specs.sort_values("rps").iloc[0].model
    learned_p = prematch_class_forecasts[learned]["platt"]
    market_p = prematch_class_forecasts["market"]["raw"]
    error = final_features[["match_id", "clean_date", "home_team", "away_team"]].copy()
    error["actual"] = final_targets.outcome
    error["goal_margin"] = final_targets.goal_margin
    error["learned_rps"] = individual_rps(final_targets.outcome, learned_p)
    error["market_rps"] = market_loss
    error["learned_prediction"] = np.asarray(CLASS_ORDER)[learned_p.argmax(axis=1)]
    error["market_prediction"] = np.asarray(CLASS_ORDER)[market_p.argmax(axis=1)]
    error["draw"] = error.actual.eq("D")
    error["large_margin"] = error.goal_margin.abs().ge(3)
    error["market_upset"] = error.market_prediction.ne(error.actual)
    error["model_market_disagree"] = error.learned_prediction.ne(error.market_prediction)
    error.sort_values("learned_rps", ascending=False).to_csv(final_dir / "final_error_cases.csv", index=False)

    figure_dir = OUTPUT_DIR / "figures" / "final"
    figure_dir.mkdir(parents=True, exist_ok=True)
    class_frame = pd.DataFrame(class_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=class_frame.sort_values("rps"), x="model", y="rps", hue="method", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Sealed final Model 1 RPS (34 available matches)")
    save_figure(fig, figure_dir / "final_model1_rps.png")
    reg_frame = pd.DataFrame(reg_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=reg_frame.sort_values("mae"), x="model", y="mae", hue="model", legend=False, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Sealed final Model 2 MAE")
    save_figure(fig, figure_dir / "final_model2_mae.png")
    minute_frame = pd.DataFrame(minute_rows)
    minute_frame = minute_frame[
        minute_frame.model.isin(["frozen_market", "frozen_best_prematch", "gradient_boosting", "random_forest", "scratch_figs"])
        & (minute_frame.method.eq("platt") | minute_frame.model.eq("frozen_market"))
    ]
    minute_frame["series"] = minute_frame.model + "/" + minute_frame.method
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=minute_frame, x="snapshot_minute", y="rps", hue="series", marker="o", ax=ax)
    ax.set_title("Sealed final Model 3 RPS by minute")
    save_figure(fig, figure_dir / "final_model3_rps_by_minute.png")

    marker["status"] = "completed"
    marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker["result_files"] = sorted(path.name for path in final_dir.iterdir() if path.is_file())
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print("SEALED FINAL EVALUATION COMPLETED EXACTLY ONCE.")


if __name__ == "__main__":
    main()

