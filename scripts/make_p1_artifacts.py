"""Create publication-ready P1 figures and descriptive tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml_project.config import NORMALIZED_DIR, OUTPUT_DIR

sns.set_theme(style="whitegrid", context="paper")


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap_grid(matrices, titles, path, cmap="mako"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, matrix, title in zip(axes.flat, matrices, titles):
        sns.heatmap(matrix, cmap=cmap, ax=ax, cbar_kws={"label": "edge count"})
        ax.set_title(title)
        ax.set_xlabel("destination zone")
        ax.set_ylabel("source zone")
    save(fig, path)


def main():
    figure_dir = OUTPUT_DIR / "figures" / "p1"
    table_dir = OUTPUT_DIR / "tables" / "p1"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(NORMALIZED_DIR / "integrated_matches.parquet")
    dev_matches = matches[matches.season_name.eq("2015/2016")].copy()
    metrics = pd.read_parquet(OUTPUT_DIR / "p1" / "team_match_metrics.parquet")
    metrics = metrics[metrics.season_name.eq("2015/2016")].copy()
    zones = pd.read_parquet(OUTPUT_DIR / "p1" / "team_zone_metrics.parquet")
    zones = zones[zones.season_name.eq("2015/2016")].copy()
    matrices = np.load(OUTPUT_DIR / "p1" / "match_matrices.npz")
    match_ids = matrices["match_ids"]
    dev_indices = np.flatnonzero(np.isin(match_ids, dev_matches.match_id))

    example_row = dev_matches[
        dev_matches.home_team.str.contains("Barcelona", case=False)
        & dev_matches.away_team.str.contains("Real Madrid", case=False)
    ]
    if example_row.empty:
        example_row = dev_matches.iloc[[0]]
    example = example_row.iloc[0]
    matrix_index = int(np.flatnonzero(match_ids == example.match_id)[0])
    heatmap_grid(
        [
            matrices["home_passes"][matrix_index], matrices["home_to_away"][matrix_index],
            matrices["away_to_home"][matrix_index], matrices["away_passes"][matrix_index],
        ],
        [
            f"{example.home_team}: completed passes", f"{example.home_team} → {example.away_team}",
            f"{example.away_team} → {example.home_team}", f"{example.away_team}: completed passes",
        ],
        figure_dir / "p1_example_multilayer_network.png",
    )
    heatmap_grid(
        [
            matrices["home_passes"][dev_indices].mean(axis=0), matrices["home_to_away"][dev_indices].mean(axis=0),
            matrices["away_to_home"][dev_indices].mean(axis=0), matrices["away_passes"][dev_indices].mean(axis=0),
        ],
        ["League-average home passes", "League-average home→away", "League-average away→home", "League-average away passes"],
        figure_dir / "p1_league_average_network.png",
    )

    zone_average = zones.groupby("zone")[["leakage", "recovery", "supra_centrality"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, column, title in zip(axes, zone_average.columns, ["Leakage", "Recovery", "Supra centrality"]):
        sns.heatmap(zone_average[column].to_numpy().reshape(5, 4), cmap="vlag", center=zone_average[column].mean(), annot=True, fmt=".3f", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("attacking x bin →")
        ax.set_ylabel("y bin")
    save(fig, figure_dir / "p1_league_zone_heatmaps.png")

    teams = [team for team in ("Barcelona", "Atlético Madrid", "Getafe") if team in set(zones.team)]
    fig, axes = plt.subplots(len(teams), 3, figsize=(11, 3.2 * len(teams)), squeeze=False)
    for row, team in enumerate(teams):
        team_zone = zones[zones.team.eq(team)].groupby("zone")[["leakage_z", "recovery_z", "supra_centrality_z"]].mean()
        for col, metric in enumerate(team_zone.columns):
            sns.heatmap(team_zone[metric].to_numpy().reshape(5, 4), cmap="vlag", center=0, vmin=-1.5, vmax=1.5, ax=axes[row, col], cbar=row == 0)
            axes[row, col].set_title(f"{team}: {metric.replace('_z', '')}")
            axes[row, col].set_xlabel("attacking x bin →")
            axes[row, col].set_ylabel("y bin")
    save(fig, figure_dir / "p1_team_zone_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, column in zip(axes, ["accumulated_eigenvector_centrality", "mean_zone_leakage", "switching_factor"]):
        sns.histplot(metrics[column], kde=True, ax=ax)
        ax.set_title(column.replace("_", " ").title())
    save(fig, figure_dir / "p1_metric_distributions.png")

    home_points = dev_matches[["match_id", "home_team", "home_score", "away_score"]].copy()
    home_points["points"] = np.where(home_points.home_score > home_points.away_score, 3, np.where(home_points.home_score == home_points.away_score, 1, 0))
    home_points = home_points.rename(columns={"home_team": "team"})[["match_id", "team", "points"]]
    away_points = dev_matches[["match_id", "away_team", "home_score", "away_score"]].copy()
    away_points["points"] = np.where(away_points.away_score > away_points.home_score, 3, np.where(away_points.home_score == away_points.away_score, 1, 0))
    away_points = away_points.rename(columns={"away_team": "team"})[["match_id", "team", "points"]]
    points = pd.concat([home_points, away_points], ignore_index=True).groupby("team", as_index=False).points.sum()
    summary = metrics.groupby("team", as_index=False).agg(
        mean_supra_centrality=("accumulated_eigenvector_centrality", "mean"),
        mean_leakage=("mean_zone_leakage", "mean"),
        mean_recovery=("mean_zone_recovery", "mean"),
        mean_switching=("switching_factor", "mean"),
        mean_completed_passes=("completed_passes", "mean"),
    ).merge(points, on="team", validate="one_to_one")
    summary["centrality_rank"] = summary.mean_supra_centrality.rank(ascending=False, method="min").astype(int)
    summary["points_rank"] = summary.points.rank(ascending=False, method="min").astype(int)
    summary.sort_values("points", ascending=False).to_csv(table_dir / "p1_team_comparison.csv", index=False)
    correlation = summary[["mean_supra_centrality", "points"]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=summary, x="mean_supra_centrality", y="points", ax=ax, scatter_kws={"s": 45})
    for row in summary.itertuples(index=False):
        if row.team in teams or row.points_rank <= 3:
            ax.annotate(row.team, (row.mean_supra_centrality, row.points), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"Team centrality versus points (descriptive r={correlation:.2f})")
    save(fig, figure_dir / "p1_centrality_vs_points.png")
    pd.DataFrame([{"pearson_r": correlation, "n_teams": len(summary), "claim": "descriptive association; not causal"}]).to_csv(table_dir / "p1_centrality_points_association.csv", index=False)

    orientation = pd.read_csv(OUTPUT_DIR / "p1" / "attacking_orientation_validation.csv")
    orientation.to_csv(table_dir / "p1_attacking_orientation_validation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    sns.barplot(data=orientation, x="side", y="mean_shot_x", ax=axes[0], hue="side", legend=False)
    axes[0].axhline(60, ls="--", color="black", lw=1)
    axes[0].set_title("Mean shot x")
    sns.barplot(data=orientation, x="side", y="mean_pass_dx", ax=axes[1], hue="side", legend=False)
    axes[1].axhline(0, ls="--", color="black", lw=1)
    axes[1].set_title("Mean pass x displacement")
    save(fig, figure_dir / "p1_orientation_validation.png")
    print(f"Created P1 artifacts for {len(dev_matches)} matches and {len(summary)} teams.")


if __name__ == "__main__":
    main()

