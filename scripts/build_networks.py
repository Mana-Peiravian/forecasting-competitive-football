"""Recompute P1 networks for every available match using tested package code."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml_project.config import NORMALIZED_DIR, OUTPUT_DIR
from ml_project.network import (
    build_match_network,
    summarize_match_network,
    validate_attacking_orientation,
)


def main() -> None:
    output_dir = OUTPUT_DIR / "p1"
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(NORMALIZED_DIR / "events.parquet")
    matches = pd.read_parquet(NORMALIZED_DIR / "integrated_matches.parquet")
    metric_frames, zone_frames, logs = [], [], []
    supra_matrices, home_matrices, away_matrices, h2a_matrices, a2h_matrices, ids = [], [], [], [], [], []
    for number, match in enumerate(matches.itertuples(index=False), start=1):
        started = time.perf_counter()
        match_events = events[events["match_id"].eq(match.match_id)]
        network = build_match_network(match_events, match.home_team, match.away_team)
        metrics, zones = summarize_match_network(network)
        common = {
            "match_id": int(match.match_id),
            "match_date": pd.Timestamp(match.clean_date),
            "season_name": match.season_name,
            "home_team": match.home_team,
            "away_team": match.away_team,
        }
        for key, value in common.items():
            metrics[key] = value
            zones[key] = value
        metric_frames.append(metrics)
        zone_frames.append(zones)
        ids.append(int(match.match_id))
        supra_matrices.append(network.supra)
        home_matrices.append(network.home_passes)
        away_matrices.append(network.away_passes)
        h2a_matrices.append(network.home_to_away)
        a2h_matrices.append(network.away_to_home)
        logs.append(
            {
                **common,
                "events": len(match_events),
                "elapsed_seconds": time.perf_counter() - started,
                "status": "success",
            }
        )
        if number % 50 == 0:
            print(f"P1 networks: {number}/{len(matches)}")
    metrics_all = pd.concat(metric_frames, ignore_index=True)
    zones_all = pd.concat(zone_frames, ignore_index=True)
    for column in ("leakage", "recovery", "supra_centrality", "individual_layer_centrality"):
        zones_all[f"{column}_z"] = zones_all.groupby(["season_name", "zone"])[column].transform(
            lambda series: (series - series.mean()) / series.std(ddof=0) if series.std(ddof=0) > 0 else 0.0
        )
    metrics_all.to_parquet(output_dir / "team_match_metrics.parquet", index=False)
    zones_all.to_parquet(output_dir / "team_zone_metrics.parquet", index=False)
    pd.DataFrame(logs).to_csv(output_dir / "processing_log.csv", index=False)
    np.savez_compressed(
        output_dir / "match_matrices.npz",
        match_ids=np.asarray(ids, dtype=np.int64),
        supra=np.stack(supra_matrices),
        home_passes=np.stack(home_matrices),
        away_passes=np.stack(away_matrices),
        home_to_away=np.stack(h2a_matrices),
        away_to_home=np.stack(a2h_matrices),
    )
    orientation = validate_attacking_orientation(events[events["match_date"].dt.year.le(2016)])
    orientation.to_csv(output_dir / "attacking_orientation_validation.csv", index=False)
    assert orientation["orientation_supported"].all()
    assert len(metrics_all) == 2 * len(matches)
    assert len(zones_all) == 40 * len(matches)
    print(f"Built {len(matches)} P1 networks; all orientation checks passed.")


if __name__ == "__main__":
    main()

