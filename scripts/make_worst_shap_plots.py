"""Create local SHAP evidence for the ten worst pilot rows in each task."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ml_project.config import CLASS_ORDER, MODEL_DIR, NORMALIZED_DIR, OUTPUT_DIR


def transform(pipeline, frame):
    values = frame
    for _, step in pipeline.steps[:-1]:
        values = step.transform(values)
    return np.asarray(values), pipeline.steps[-1][1]


def normalize(explanation, n_features):
    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[2] == n_features:
        values = np.transpose(values, (0, 2, 1))
    return values


def save_local(rows, task, figure_dir):
    first = rows[rows.error_rank.eq(1)].copy()
    first = first.reindex(first.shap_value.abs().sort_values().index).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#b44" if value > 0 else "#4676b8" for value in first.shap_value]
    ax.barh(first.feature, first.shap_value, color=colors)
    ax.set_title(f"Worst pilot prediction: {task}")
    ax.set_xlabel("local SHAP contribution")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{task}_worst_local_shap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    output = OUTPUT_DIR / "explainability"
    figures = OUTPUT_DIR / "figures" / "explainability"
    targets = pd.read_parquet(NORMALIZED_DIR / "development_targets.parquet")[
        ["match_id", "outcome", "goal_margin"]
    ]
    feature_manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    columns = feature_manifest["groups"]["combined"]
    prematch = pd.read_parquet(OUTPUT_DIR / "features" / "prematch_features.parquet")
    prematch = prematch[prematch.split_role.eq("pilot_holdout")].merge(targets, on="match_id", validate="one_to_one")
    all_rows = []

    class_predictions = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_classification_predictions.csv")
    class_predictions = class_predictions[(class_predictions.model.eq("random_forest")) & class_predictions.method.eq("raw")].copy()
    y = np.eye(3)[class_predictions.actual.map({label: i for i, label in enumerate(CLASS_ORDER)})]
    p = class_predictions[["p_A", "p_D", "p_H"]].to_numpy()
    class_predictions["error"] = np.sum((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(y, axis=1)[:, :-1]) ** 2, axis=1) / 2
    worst = class_predictions.nlargest(10, "error")
    selected = prematch.set_index("match_id").loc[worst.match_id].reset_index()
    pipeline = joblib.load(MODEL_DIR / "development" / "prematch" / "classifier_random_forest.joblib")
    matrix, tree = transform(pipeline, selected[columns].astype(float))
    values = normalize(shap.TreeExplainer(tree)(matrix), len(columns))
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        local = values[rank - 1, :, CLASS_ORDER.index(row.outcome)]
        frame = pd.DataFrame({"feature": columns, "feature_value": matrix[rank - 1], "shap_value": local})
        frame["task"], frame["error_rank"], frame["match_id"] = "model1_classification", rank, int(row.match_id)
        all_rows.append(frame)
    save_local(pd.concat(all_rows), "model1_classification", figures)

    regression_predictions = pd.read_csv(OUTPUT_DIR / "experiments" / "prematch" / "pilot_regression_predictions.csv")
    regression_predictions = regression_predictions[regression_predictions.model.eq("random_forest")].copy()
    regression_predictions["error"] = (regression_predictions.actual_margin - regression_predictions.predicted_margin).abs()
    worst = regression_predictions.nlargest(10, "error")
    selected = prematch.set_index("match_id").loc[worst.match_id].reset_index()
    pipeline = joblib.load(MODEL_DIR / "development" / "prematch" / "regressor_random_forest.joblib")
    matrix, tree = transform(pipeline, selected[columns].astype(float))
    values = normalize(shap.TreeExplainer(tree)(matrix), len(columns))
    task_rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        frame = pd.DataFrame({"feature": columns, "feature_value": matrix[rank - 1], "shap_value": values[rank - 1]})
        frame["task"], frame["error_rank"], frame["match_id"] = "model2_regression", rank, int(row.match_id)
        all_rows.append(frame); task_rows.append(frame)
    save_local(pd.concat(task_rows), "model2_regression", figures)

    snapshot_manifest = json.loads((OUTPUT_DIR / "snapshots" / "snapshot_manifest.json").read_text())
    snapshot_columns = snapshot_manifest["groups"]["combined"]
    snapshots = pd.read_parquet(OUTPUT_DIR / "snapshots" / "snapshot_features.parquet")
    snapshots = snapshots[snapshots.split_role.eq("pilot_holdout")].merge(targets, on="match_id", validate="many_to_one")
    snapshot_predictions = pd.read_parquet(OUTPUT_DIR / "experiments" / "snapshots" / "pilot_classification_predictions.parquet")
    snapshot_predictions = snapshot_predictions[(snapshot_predictions.model.eq("random_forest")) & snapshot_predictions.method.eq("raw")].copy()
    y = np.eye(3)[snapshot_predictions.actual.map({label: i for i, label in enumerate(CLASS_ORDER)})]
    p = snapshot_predictions[["p_A", "p_D", "p_H"]].to_numpy()
    snapshot_predictions["error"] = np.sum((np.cumsum(p, axis=1)[:, :-1] - np.cumsum(y, axis=1)[:, :-1]) ** 2, axis=1) / 2
    worst = snapshot_predictions.nlargest(10, "error")
    selected = worst[["match_id", "snapshot_minute"]].merge(snapshots, on=["match_id", "snapshot_minute"], validate="one_to_one")
    pipeline = joblib.load(MODEL_DIR / "development" / "snapshots" / "classifier_random_forest.joblib")
    matrix, tree = transform(pipeline, selected[snapshot_columns].astype(float))
    values = normalize(shap.TreeExplainer(tree)(matrix), len(snapshot_columns))
    task_rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        local = values[rank - 1, :, CLASS_ORDER.index(row.outcome)]
        frame = pd.DataFrame({"feature": snapshot_columns, "feature_value": matrix[rank - 1], "shap_value": local})
        frame["task"], frame["error_rank"], frame["match_id"], frame["snapshot_minute"] = "model3_classification", rank, int(row.match_id), int(row.snapshot_minute)
        all_rows.append(frame); task_rows.append(frame)
    save_local(pd.concat(task_rows), "model3_classification", figures)

    pipeline = joblib.load(MODEL_DIR / "development" / "snapshots" / "regressor_random_forest.joblib")
    snapshot_pred = pipeline.predict(snapshots[snapshot_columns].astype(float))
    candidate = snapshots[["match_id", "snapshot_minute", "goal_margin"]].copy()
    candidate["error"] = np.abs(candidate.goal_margin - snapshot_pred)
    worst = candidate.nlargest(10, "error")
    selected = worst[["match_id", "snapshot_minute"]].merge(snapshots, on=["match_id", "snapshot_minute"], validate="one_to_one")
    matrix, tree = transform(pipeline, selected[snapshot_columns].astype(float))
    values = normalize(shap.TreeExplainer(tree)(matrix), len(snapshot_columns))
    task_rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        frame = pd.DataFrame({"feature": snapshot_columns, "feature_value": matrix[rank - 1], "shap_value": values[rank - 1]})
        frame["task"], frame["error_rank"], frame["match_id"], frame["snapshot_minute"] = "model3_regression", rank, int(row.match_id), int(row.snapshot_minute)
        all_rows.append(frame); task_rows.append(frame)
    save_local(pd.concat(task_rows), "model3_regression", figures)
    pd.concat(all_rows, ignore_index=True).to_parquet(output / "worst10_local_shap.parquet", index=False)
    print("Created worst-10 local SHAP evidence for all three tasks.")


if __name__ == "__main__":
    main()
