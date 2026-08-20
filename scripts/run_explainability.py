"""SHAP global/local explanations, worst cases, and one full timeline."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from ml_project.config import CLASS_ORDER, MODEL_DIR, NORMALIZED_DIR, OUTPUT_DIR, RANDOM_SEED

sns.set_theme(style="whitegrid", context="paper")


def transformed(pipeline, X: pd.DataFrame):
    values = X
    for name, step in pipeline.steps[:-1]:
        values = step.transform(values)
    return np.asarray(values, dtype=float), pipeline.steps[-1][1]


def normalize_shap_values(explanation, n_features: int) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[1] == n_features:
        return values
    if values.ndim == 3 and values.shape[2] == n_features:
        return np.transpose(values, (0, 2, 1))
    raise ValueError(f"Unsupported SHAP shape {values.shape}")


def global_and_local(
    pipeline,
    X: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    task: str,
    output_dir: Path,
    local_indices: list[int],
) -> tuple[pd.DataFrame, list[pd.DataFrame], np.ndarray]:
    matrix, tree_model = transformed(pipeline, X)
    explainer = shap.TreeExplainer(tree_model)
    explanation = explainer(matrix)
    values = normalize_shap_values(explanation, len(feature_names))
    mean_abs = np.mean(np.abs(values), axis=(0, 2)) if values.ndim == 3 else np.mean(np.abs(values), axis=0)
    global_frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    global_frame["model"], global_frame["task"] = model_name, task
    top = global_frame.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.feature, top.mean_abs_shap)
    ax.set_title(f"{task}: {model_name} global SHAP")
    ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task}_{model_name}_global_shap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    locals_out = []
    for rank, index in enumerate(local_indices):
        local = values[index]
        if local.ndim == 2:
            class_index = int(np.argmax(np.abs(local).sum(axis=0)))
            local = local[:, class_index]
        local_frame = pd.DataFrame(
            {"feature": feature_names, "feature_value": matrix[index], "shap_value": local}
        ).assign(model=model_name, task=task, local_rank=rank)
        locals_out.append(local_frame.reindex(local_frame.shap_value.abs().sort_values(ascending=False).index).head(30))
    return global_frame, locals_out, values


def individual_rps(frame: pd.DataFrame) -> np.ndarray:
    p = frame[["p_A", "p_D", "p_H"]].to_numpy()
    y = np.eye(3)[frame.actual.map({label: i for i, label in enumerate(CLASS_ORDER)}).to_numpy()]
    return np.sum((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(y, axis=1)[:, :-1]) ** 2, axis=1) / 2


def error_analysis(output_dir: Path) -> None:
    prematch_predictions = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_classification_predictions.csv")
    rf = prematch_predictions[(prematch_predictions.model.eq("random_forest")) & prematch_predictions.method.eq("raw")].copy()
    market = prematch_predictions[(prematch_predictions.model.eq("market")) & prematch_predictions.method.eq("raw")].copy()
    merged = rf.merge(market, on=["match_id", "actual"], suffixes=("_rf", "_market"), validate="one_to_one")
    targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")[["match_id", "goal_margin"]]
    merged = merged.merge(targets, on="match_id", validate="one_to_one")
    for suffix in ("rf", "market"):
        temporary = merged[["actual", f"p_A_{suffix}", f"p_D_{suffix}", f"p_H_{suffix}"]].rename(
            columns={f"p_A_{suffix}": "p_A", f"p_D_{suffix}": "p_D", f"p_H_{suffix}": "p_H"}
        )
        merged[f"rps_{suffix}"] = individual_rps(temporary)
        merged[f"predicted_{suffix}"] = np.asarray(CLASS_ORDER)[temporary[["p_A", "p_D", "p_H"]].to_numpy().argmax(axis=1)]
    merged["slice_draw"] = merged.actual.eq("D")
    merged["slice_large_margin"] = merged.goal_margin.abs().ge(3)
    merged["slice_market_upset"] = merged.predicted_market.ne(merged.actual)
    merged["slice_model_market_disagree"] = merged.predicted_rf.ne(merged.predicted_market)
    slice_rows = []
    for column in ["slice_draw", "slice_large_margin", "slice_market_upset", "slice_model_market_disagree"]:
        for value, group in merged.groupby(column):
            slice_rows.append({"slice": column, "active": bool(value), "n": len(group), "rf_rps": group.rps_rf.mean(), "market_rps": group.rps_market.mean()})
    pd.DataFrame(slice_rows).to_csv(output_dir / "prematch_error_slices.csv", index=False)
    merged.sort_values("rps_rf", ascending=False).head(15).to_csv(output_dir / "prematch_worst_classification_cases.csv", index=False)

    reg = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_regression_predictions.csv")
    reg["absolute_error"] = (reg.actual_margin - reg.predicted_margin).abs()
    reg.sort_values("absolute_error", ascending=False).groupby("model", as_index=False).head(10).to_csv(output_dir / "prematch_worst_margin_cases.csv", index=False)


def main():
    output_dir = OUTPUT_DIR / "explainability"
    figure_dir = OUTPUT_DIR / "figures" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    prematch = pd.read_parquet(OUTPUT_DIR / "features" / "prematch_features.parquet")
    pilot_match = prematch[prematch.split_role.eq("pilot_holdout")].reset_index(drop=True)
    development_targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")
    pilot_match = pilot_match.merge(development_targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="one_to_one")
    prediction = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_classification_predictions.csv")
    global_frames, local_frames, status = [], [], []
    tree_models = ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]

    for task in ("prematch_classification", "prematch_regression"):
        is_class = task.endswith("classification")
        for model_name in tree_models:
            try:
                prefix = "classifier" if is_class else "regressor"
                pipeline = joblib.load(MODEL_DIR / "development" / "prematch" / f"{prefix}_{model_name}.joblib")
                columns = feature_manifest["groups"]["combined"]
                X = pilot_match[columns].astype(float)
                if is_class:
                    model_predictions = prediction[(prediction.model.eq(model_name)) & prediction.method.eq("raw")].copy()
                    model_predictions["loss"] = individual_rps(model_predictions)
                    worst_ids = model_predictions.nlargest(3, "loss").match_id.tolist()
                    local_indices = [int(pilot_match.index[pilot_match.match_id.eq(match_id)][0]) for match_id in worst_ids]
                else:
                    reg_pred = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_regression_predictions.csv")
                    reg_pred = reg_pred[reg_pred.model.eq(model_name)].copy()
                    reg_pred["loss"] = (reg_pred.actual_margin - reg_pred.predicted_margin).abs()
                    worst_ids = reg_pred.nlargest(3, "loss").match_id.tolist()
                    local_indices = [int(pilot_match.index[pilot_match.match_id.eq(match_id)][0]) for match_id in worst_ids]
                global_frame, locals_out, _ = global_and_local(
                    pipeline, X, columns, model_name, task, figure_dir, local_indices
                )
                global_frames.append(global_frame)
                local_frames.extend(locals_out)
                status.append({"task": task, "model": model_name, "status": "success", "rows_explained": len(X)})
            except Exception as error:
                status.append({"task": task, "model": model_name, "status": "unsupported_or_failed", "error": f"{type(error).__name__}: {error}"})

    snapshot_manifest = json.loads((OUTPUT_DIR / "snapshots" / "snapshot_manifest.json").read_text())
    snapshot_columns = snapshot_manifest["groups"]["combined"]
    snapshots = pd.read_parquet(OUTPUT_DIR / "snapshots" / "snapshot_features.parquet")
    pilot_snapshots = snapshots[snapshots.split_role.eq("pilot_holdout")].reset_index(drop=True)
    pilot_snapshots = pilot_snapshots.merge(development_targets[["match_id", "outcome", "goal_margin"]], on="match_id", validate="many_to_one")
    sample = pilot_snapshots.sample(n=min(300, len(pilot_snapshots)), random_state=RANDOM_SEED).sort_index()
    for task in ("snapshot_classification", "snapshot_regression"):
        is_class = task.endswith("classification")
        for model_name in tree_models:
            try:
                prefix = "classifier" if is_class else "regressor"
                pipeline = joblib.load(MODEL_DIR / "development" / "snapshots" / f"{prefix}_{model_name}.joblib")
                global_frame, _, _ = global_and_local(
                    pipeline, sample[snapshot_columns].astype(float), snapshot_columns,
                    model_name, task, figure_dir, [0, 1, 2],
                )
                global_frames.append(global_frame)
                status.append({"task": task, "model": model_name, "status": "success", "rows_explained": len(sample), "sampling": "seeded simple random pilot sample"})
            except Exception as error:
                status.append({"task": task, "model": model_name, "status": "unsupported_or_failed", "error": f"{type(error).__name__}: {error}"})

    if global_frames:
        pd.concat(global_frames, ignore_index=True).to_csv(output_dir / "global_shap_importance.csv", index=False)
    if local_frames:
        pd.concat(local_frames, ignore_index=True).to_csv(output_dir / "worst_case_local_shap.csv", index=False)
    pd.DataFrame(status).to_csv(output_dir / "shap_support_and_sampling.csv", index=False)
    error_analysis(output_dir)

    # One complete Model 3 explanation timeline, selected as the worst RF match
    # by mean snapshot RPS on the opened pilot.
    snapshot_predictions = pd.read_parquet(OUTPUT_DIR / "experiments" / "snapshots" / "pilot_classification_predictions.parquet")
    rf_predictions = snapshot_predictions[(snapshot_predictions.model.eq("random_forest")) & snapshot_predictions.method.eq("raw")].copy()
    rf_predictions["individual_rps"] = individual_rps(rf_predictions)
    worst_match = int(rf_predictions.groupby("match_id").individual_rps.mean().idxmax())
    timeline = pilot_snapshots[pilot_snapshots.match_id.eq(worst_match)].sort_values("snapshot_minute").reset_index(drop=True)
    pipeline = joblib.load(MODEL_DIR / "development" / "snapshots" / "classifier_random_forest.joblib")
    matrix, tree_model = transformed(pipeline, timeline[snapshot_columns].astype(float))
    explanation = shap.TreeExplainer(tree_model)(matrix)
    values = normalize_shap_values(explanation, len(snapshot_columns))
    actual_class = CLASS_ORDER.index(timeline.outcome.iloc[0])
    class_values = values[:, :, actual_class] if values.ndim == 3 else values
    global_timeline = np.mean(np.abs(class_values), axis=0)
    top_indices = np.argsort(global_timeline)[-10:][::-1]
    timeline_output = timeline[["match_id", "snapshot_minute", "home_team", "away_team", "home_score", "away_score", "score_diff", "outcome"]].copy()
    p_timeline = rf_predictions[rf_predictions.match_id.eq(worst_match)].sort_values("snapshot_minute")
    timeline_output[["p_A", "p_D", "p_H", "individual_rps"]] = p_timeline[["p_A", "p_D", "p_H", "individual_rps"]].to_numpy()
    for index in top_indices:
        timeline_output[f"shap_{snapshot_columns[index]}"] = class_values[:, index]
    top_each = np.argmax(np.abs(class_values), axis=1)
    timeline_output["top_feature"] = [snapshot_columns[index] for index in top_each]
    timeline_output["top_feature_value"] = matrix[np.arange(len(matrix)), top_each]
    timeline_output["top_feature_shap"] = class_values[np.arange(len(matrix)), top_each]
    timeline_output.to_csv(output_dir / "full_match_explanation_timeline.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    for label in CLASS_ORDER:
        axes[0].plot(timeline_output.snapshot_minute, timeline_output[f"p_{label}"], marker="o", label=label)
    axes[0].legend(title="outcome")
    axes[0].set_ylabel("probability")
    axes[0].set_title(f"Full explanation timeline: {timeline.home_team.iloc[0]} vs {timeline.away_team.iloc[0]}")
    sns.heatmap(class_values[:, top_indices].T, cmap="vlag", center=0, ax=axes[1],
                xticklabels=timeline_output.snapshot_minute, yticklabels=[snapshot_columns[index] for index in top_indices])
    axes[1].set_xlabel("snapshot minute")
    axes[1].set_ylabel(f"SHAP contribution to class {CLASS_ORDER[actual_class]}")
    fig.tight_layout()
    fig.savefig(figure_dir / "full_match_explanation_timeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP/error artifacts complete; full timeline match_id={worst_match}.")


if __name__ == "__main__":
    main()

