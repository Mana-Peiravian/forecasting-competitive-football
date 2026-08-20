"""Create development/pilot model figures, FIGS rules, and compute tables."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ml_project.config import MODEL_DIR, OUTPUT_DIR

sns.set_theme(style="whitegrid", context="paper")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_dir = OUTPUT_DIR / "figures" / "models"
    table_dir = OUTPUT_DIR / "tables" / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    prematch = OUTPUT_DIR / "experiments" / "prematch"
    snapshots = OUTPUT_DIR / "experiments" / "snapshots"

    selection = pd.read_csv(prematch / "classification_model_selection.csv")
    best = selection.loc[selection.groupby("model")["rps"].idxmin()].sort_values("rps")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=best, x="rps", y="model", hue="model", legend=False, ax=ax)
    ax.set_title("Model 1 validation RPS (lower is better)")
    save(fig, fig_dir / "model1_validation_rps.png")

    reg = pd.read_csv(prematch / "regression_model_selection.csv")
    best_reg = reg.loc[reg.groupby("model")["mae"].idxmin()].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=best_reg, x="mae", y="model", hue="model", legend=False, ax=ax)
    ax.set_title("Model 2 validation MAE (lower is better)")
    save(fig, fig_dir / "model2_validation_mae.png")

    calibration = pd.read_csv(prematch / "classification_calibration_and_pilot.csv")
    calibration = calibration[calibration.partition.eq("pilot_holdout")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=calibration, x="model", y="rps", hue="method", ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Opened pilot: pre/post calibration RPS")
    save(fig, fig_dir / "model1_pilot_calibration.png")

    reliability = pd.read_csv(prematch / "reliability_bins.csv")
    reliability = reliability[
        reliability.partition.eq("pilot_holdout")
        & reliability.series.eq("top_label")
        & reliability.model.isin(["market", "random_forest", "scratch_figs"])
        & reliability.method.isin(["raw", "platt"])
        & reliability["count"].gt(0)
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    for (model, method), group in reliability.groupby(["model", "method"]):
        ax.plot(group.mean_confidence, group.observed_frequency, marker="o", label=f"{model}/{method}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="mean confidence", ylabel="observed accuracy", title="Opened-pilot reliability")
    ax.legend(fontsize=7)
    save(fig, fig_dir / "model1_reliability.png")

    class_ablation = pd.read_csv(prematch / "classification_feature_ablation.csv")
    reg_ablation = pd.read_csv(prematch / "regression_feature_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=class_ablation, x="feature_group", y="rps", hue="model", ax=axes[0])
    sns.barplot(data=reg_ablation, x="feature_group", y="mae", hue="model", ax=axes[1])
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_title("Model 1 feature ablation")
    axes[1].set_title("Model 2 feature ablation")
    save(fig, fig_dir / "prematch_feature_ablations.png")

    figs_curve = selection[selection.model.eq("scratch_figs")].copy()
    figs_curve["max_rules"] = figs_curve.params.map(lambda value: json.loads(value)["max_rules"])
    figs_reg_curve = reg[reg.model.eq("scratch_figs")].copy()
    figs_reg_curve["max_rules"] = figs_reg_curve.params.map(lambda value: json.loads(value)["max_rules"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    sns.lineplot(data=figs_curve, x="max_rules", y="rps", marker="o", ax=axes[0])
    sns.lineplot(data=figs_reg_curve, x="max_rules", y="mae", marker="o", ax=axes[1])
    axes[0].set_title("Scratch FIGS classification")
    axes[1].set_title("Scratch FIGS regression")
    save(fig, fig_dir / "figs_split_budget.png")

    minute = pd.read_csv(snapshots / "classification_by_minute.csv")
    minute = minute[
        minute.model.isin(["frozen_market", "gradient_boosting", "random_forest", "scratch_figs"])
        & ((minute.method.eq("raw")) | (minute.method.eq("platt") & minute.model.eq("gradient_boosting")))
    ]
    minute["series"] = minute.model + "/" + minute.method
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=minute, x="snapshot_minute", y="rps", hue="series", marker="o", ax=ax)
    ax.set_title("Model 3 opened-pilot RPS by match minute")
    save(fig, fig_dir / "model3_rps_by_minute.png")

    minute_reg = pd.read_csv(snapshots / "regression_by_minute.csv")
    minute_reg = minute_reg[minute_reg.model.isin(["random_forest", "gradient_boosting", "scratch_figs", "dummy"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=minute_reg, x="snapshot_minute", y="mae", hue="model", marker="o", ax=ax)
    ax.set_title("Model 3 opened-pilot margin MAE by match minute")
    save(fig, fig_dir / "model3_mae_by_minute.png")

    imbalance = pd.read_csv(prematch / "imbalance_training_only.csv")
    snapshot_imbalance = pd.read_csv(snapshots / "imbalance_no_synthetic_snapshots.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.barplot(data=imbalance.sort_values("rps"), x="variant", y="rps", ax=axes[0], hue="variant", legend=False)
    sns.barplot(data=snapshot_imbalance.sort_values("rps"), x="variant", y="rps", ax=axes[1], hue="variant", legend=False)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Prematch training-only imbalance methods")
    axes[1].set_title("Snapshots: no synthetic rows")
    save(fig, fig_dir / "imbalance_comparison.png")

    kernel = pd.read_csv(prematch / "kernel_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sns.lineplot(data=kernel, x="n_samples", y="elapsed_seconds", hue="model", marker="o", ax=axes[0])
    sns.lineplot(data=kernel, x="n_samples", y="kernel_matrix_elements", hue="model", marker="o", ax=axes[1])
    axes[0].set_title("Kernel wall time")
    axes[1].set_title("Stored feature/kernel elements")
    save(fig, fig_dir / "kernel_scaling.png")

    compute = pd.concat(
        [
            pd.read_csv(prematch / "compute_profile_selection.csv"),
            pd.read_csv(prematch / "compute_profile_locked_fit.csv"),
            pd.read_csv(snapshots / "compute_profile.csv"),
        ],
        ignore_index=True,
    )
    compute.to_csv(table_dir / "compute_profile_all_models.csv", index=False)
    summary = compute.groupby(["stage", "task", "model"], as_index=False).agg(
        elapsed_seconds=("elapsed_seconds", "sum"), peak_rss_mb=("peak_rss_mb", "max")
    )
    summary.to_csv(table_dir / "compute_profile_summary.csv", index=False)

    feature_manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    for task, filename, spec_file in (
        ("classification", "classifier_scratch_figs.joblib", prematch / "locked_classifier_specs.csv"),
        ("regression", "regressor_scratch_figs.joblib", prematch / "locked_regressor_specs.csv"),
    ):
        pipeline = joblib.load(MODEL_DIR / "development" / "prematch" / filename)
        specs = pd.read_csv(spec_file)
        group = specs.loc[specs.model.eq("scratch_figs"), "feature_group"].iloc[0]
        rules = pipeline.named_steps["model"].export_rules(feature_manifest["groups"][group])
        pd.DataFrame(rules).to_json(table_dir / f"scratch_figs_{task}_rules.json", orient="records", indent=2)
    print("Created model figures, compute tables, and scratch-FIGS rule exports.")


if __name__ == "__main__":
    main()

