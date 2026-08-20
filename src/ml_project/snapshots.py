"""Exact-boundary in-play snapshot construction with match-grouped roles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import SNAPSHOT_MINUTES

COUNT_METRICS = (
    "shots",
    "passes",
    "completed_passes",
    "final_third_entries",
    "pressures",
    "recoveries",
    "interceptions",
    "fouls",
    "yellow_cards",
    "red_cards",
    "carries",
)
WEIGHTED_METRICS = ("xg",)


def _event_indicators(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["shots"] = frame["type"].eq("Shot").astype(float)
    frame["passes"] = frame["type"].eq("Pass").astype(float)
    frame["completed_passes"] = (frame["type"].eq("Pass") & frame["pass_outcome"].isna()).astype(float)
    end_x = frame["pass_end_x"].where(frame["type"].eq("Pass"), frame["carry_end_x"])
    frame["final_third_entries"] = (
        frame["type"].isin(["Pass", "Carry"])
        & frame["location_x"].lt(80)
        & end_x.ge(80)
    ).astype(float)
    frame["pressures"] = frame["type"].eq("Pressure").astype(float)
    frame["recoveries"] = (
        frame["type"].eq("Ball Recovery")
        & ~frame["ball_recovery_recovery_failure"].fillna(False).astype(bool)
    ).astype(float)
    frame["interceptions"] = frame["type"].eq("Interception").astype(float)
    frame["fouls"] = frame["type"].eq("Foul Committed").astype(float)
    card = frame["foul_committed_card"].fillna(frame["bad_behaviour_card"]).astype("string")
    frame["yellow_cards"] = card.isin(["Yellow Card", "Second Yellow"]).astype(float)
    frame["red_cards"] = card.isin(["Red Card", "Second Yellow"]).astype(float)
    frame["carries"] = frame["type"].eq("Carry").astype(float)
    frame["xg"] = pd.to_numeric(frame["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
    return frame


def _window_values(
    times: np.ndarray, values: np.ndarray, cutoffs: np.ndarray, window_seconds: float | None
) -> np.ndarray:
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    cumulative = np.concatenate([[0.0], np.cumsum(values[order], dtype=float)])
    right = np.searchsorted(sorted_times, cutoffs, side="right")
    if window_seconds is None:
        left = np.zeros_like(right)
    else:
        # (t-window, t], so events exactly on the old boundary are excluded.
        left = np.searchsorted(sorted_times, cutoffs - window_seconds, side="right")
    return cumulative[right] - cumulative[left]


def build_match_snapshots(
    match_events: pd.DataFrame,
    match_row: pd.Series,
    prematch_row: pd.Series,
    frozen_columns: Sequence[str],
    snapshot_minutes: Sequence[int] = SNAPSHOT_MINUTES,
) -> pd.DataFrame:
    if match_events.empty:
        raise ValueError("Cannot build snapshots from an empty event stream")
    frame = _event_indicators(match_events)
    if "event_seconds_exact" not in frame:
        raise ValueError("Exact event seconds are required")
    times = frame["event_seconds_exact"].to_numpy(dtype=float)
    cutoffs = np.asarray(snapshot_minutes, dtype=float) * 60.0
    home_team, away_team = str(match_row["home_team"]), str(match_row["away_team"])
    output = pd.DataFrame(
        {
            "match_id": int(match_row["match_id"]),
            "clean_date": pd.Timestamp(match_row["clean_date"]),
            "season_name": match_row["season_name"],
            "home_team": home_team,
            "away_team": away_team,
            "split_role": prematch_row["split_role"],
            "snapshot_minute": np.asarray(snapshot_minutes, dtype=int),
            "snapshot_seconds": cutoffs,
        }
    )
    dynamic: dict[str, np.ndarray | list[float]] = {}
    for side, team in (("home", home_team), ("away", away_team)):
        team_mask = frame["team"].eq(team).to_numpy()
        team_times = times[team_mask]
        for metric in (*COUNT_METRICS, *WEIGHTED_METRICS):
            values = frame.loc[team_mask, metric].to_numpy(dtype=float)
            dynamic[f"{side}_cum_{metric}"] = _window_values(team_times, values, cutoffs, None)
            dynamic[f"{side}_recent5_{metric}"] = _window_values(team_times, values, cutoffs, 5 * 60.0)
            dynamic[f"{side}_recent10_{metric}"] = _window_values(team_times, values, cutoffs, 10 * 60.0)

    shot_goal = frame["type"].eq("Shot") & frame["shot_outcome"].eq("Goal")
    own_goal = frame["type"].eq("Own Goal Against")
    for side, team, opponent in (("home", home_team, away_team), ("away", away_team, home_team)):
        goal_mask = (shot_goal & frame["team"].eq(team)) | (own_goal & frame["team"].eq(opponent))
        goal_times = times[goal_mask.to_numpy()]
        dynamic[f"{side}_score"] = _window_values(
            goal_times, np.ones(len(goal_times), dtype=float), cutoffs, None
        )

    latest_indices = np.searchsorted(np.sort(times), cutoffs, side="right") - 1
    ordered_team = frame.iloc[np.argsort(times, kind="mergesort")]["team"].to_numpy()
    dynamic["home_in_possession"] = [
        float(ordered_team[index] == home_team) if index >= 0 else 0.0 for index in latest_indices
    ]
    dynamic["feature_source_max_seconds"] = [
        float(np.max(times[times <= cutoff])) if np.any(times <= cutoff) else np.nan for cutoff in cutoffs
    ]
    dynamic_frame = pd.DataFrame(dynamic, index=output.index)
    output = pd.concat([output, dynamic_frame], axis=1)
    if (
        output["feature_source_max_seconds"].notna()
        & output["feature_source_max_seconds"].gt(output["snapshot_seconds"])
    ).any():
        raise AssertionError("Snapshot contains a post-boundary event")
    derived: dict[str, np.ndarray | pd.Series] = {
        "score_diff": output["home_score"] - output["away_score"],
        "match_fraction": output["snapshot_minute"] / 90.0,
        "phase_first_half": output["snapshot_minute"].lt(45).astype(float),
        "phase_halftime": output["snapshot_minute"].eq(45).astype(float),
        "phase_late": output["snapshot_minute"].ge(75).astype(float),
    }
    for timing in ("cum", "recent5", "recent10"):
        for metric in (*COUNT_METRICS, *WEIGHTED_METRICS):
            derived[f"diff_{timing}_{metric}"] = output[f"home_{timing}_{metric}"] - output[f"away_{timing}_{metric}"]
    output = pd.concat([output, pd.DataFrame(derived, index=output.index)], axis=1)
    frozen_frame = pd.DataFrame(
        {
            f"prematch_{column}": np.repeat(prematch_row[column], len(output))
            for column in frozen_columns
        },
        index=output.index,
    )
    return pd.concat([output, frozen_frame], axis=1)


def build_snapshot_dataset(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    prematch: pd.DataFrame,
    frozen_columns: Sequence[str],
    snapshot_minutes: Sequence[int] = SNAPSHOT_MINUTES,
) -> pd.DataFrame:
    event_groups = {int(key): group for key, group in events.groupby("match_id", sort=False)}
    prematch_index = prematch.set_index("match_id", verify_integrity=True)
    rows = []
    for match in matches.sort_values(["clean_date", "match_id"]).to_dict("records"):
        match_id = int(match["match_id"])
        rows.append(
            build_match_snapshots(
                event_groups[match_id],
                pd.Series(match),
                prematch_index.loc[match_id],
                frozen_columns,
                snapshot_minutes,
            )
        )
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["match_id", "snapshot_minute"]).any():
        raise AssertionError("Duplicate match snapshots")
    if output.groupby("match_id")["split_role"].nunique().max() != 1:
        raise AssertionError("Snapshots from a match crossed split roles")
    return output


def write_snapshot_artifacts(
    snapshots: pd.DataFrame,
    frozen_columns: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "snapshot_features.parquet"
    snapshots.to_parquet(path, index=False)
    identifiers = {
        "match_id", "clean_date", "season_name", "home_team", "away_team", "split_role",
        "snapshot_minute", "snapshot_seconds", "feature_source_max_seconds",
    }
    frozen = [f"prematch_{column}" for column in frozen_columns]
    inplay = [column for column in snapshots if column not in identifiers and column not in frozen]
    payload = {
        "snapshot_minutes": sorted(snapshots["snapshot_minute"].unique().tolist()),
        "rows": len(snapshots),
        "matches": int(snapshots["match_id"].nunique()),
        "groups": {"inplay": inplay, "frozen_prematch": frozen, "combined": [*inplay, *frozen]},
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "boundary": "event_seconds_exact <= snapshot_seconds; recent windows are (t-w, t].",
    }
    (output_dir / "snapshot_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
