"""Leakage-safe historical event and P1 feature engineering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CALIBRATION_END,
    CALIBRATION_START,
    CLASS_ORDER,
    FINAL_SEASON,
    PILOT_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
)

EVENT_METRICS = (
    "goals_for",
    "goals_against",
    "points",
    "shots",
    "shots_on_target",
    "xg",
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
    "duels_won",
    "possessions",
)

NETWORK_METRICS = (
    "accumulated_eigenvector_centrality",
    "maximum_zone_centrality",
    "mean_zone_centrality",
    "mean_zone_leakage",
    "median_zone_leakage",
    "maximum_zone_leakage",
    "mean_zone_recovery",
    "median_zone_recovery",
    "maximum_zone_recovery",
    "completed_passes",
    "possession_losses",
    "possession_recoveries",
    "switching_factor",
)


def assign_split_role(matches: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(matches["clean_date"])
    role = pd.Series("unassigned", index=matches.index, dtype="string")
    final = matches["season_name"].eq(FINAL_SEASON)
    role.loc[final] = "final_test"
    development = ~final
    role.loc[development & dates.le(TRAIN_END)] = "train"
    role.loc[development & dates.between(VALIDATION_START, VALIDATION_END)] = "validation"
    role.loc[development & dates.between(CALIBRATION_START, CALIBRATION_END)] = "calibration"
    role.loc[development & dates.ge(PILOT_START)] = "pilot_holdout"
    if role.eq("unassigned").any():
        raise ValueError(f"Unassigned matches: {int(role.eq('unassigned').sum())}")
    return role


def aggregate_event_team_matches(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    frame = events[events["team"].notna()].copy()
    frame["shots"] = frame["type"].eq("Shot").astype(int)
    frame["shots_on_target"] = (
        frame["type"].eq("Shot")
        & frame["shot_outcome"].isin(["Goal", "Saved", "Saved to Post"])
    ).astype(int)
    frame["xg"] = pd.to_numeric(frame["shot_statsbomb_xg"], errors="coerce").fillna(0.0)
    frame["passes"] = frame["type"].eq("Pass").astype(int)
    frame["completed_passes"] = (frame["type"].eq("Pass") & frame["pass_outcome"].isna()).astype(int)
    end_x = frame["pass_end_x"].where(frame["type"].eq("Pass"), frame["carry_end_x"])
    frame["final_third_entries"] = (
        frame["type"].isin(["Pass", "Carry"])
        & frame["location_x"].lt(80)
        & end_x.ge(80)
    ).astype(int)
    frame["pressures"] = frame["type"].eq("Pressure").astype(int)
    frame["recoveries"] = (
        frame["type"].eq("Ball Recovery")
        & ~frame["ball_recovery_recovery_failure"].fillna(False).astype(bool)
    ).astype(int)
    frame["interceptions"] = frame["type"].eq("Interception").astype(int)
    frame["fouls"] = frame["type"].eq("Foul Committed").astype(int)
    card = frame["foul_committed_card"].fillna(frame["bad_behaviour_card"]).astype("string")
    frame["yellow_cards"] = card.isin(["Yellow Card", "Second Yellow"]).astype(int)
    frame["red_cards"] = card.isin(["Red Card", "Second Yellow"]).astype(int)
    frame["carries"] = frame["type"].eq("Carry").astype(int)
    frame["duels_won"] = frame["duel_outcome"].isin(["Won", "Success", "Success In Play", "Success Out"]).astype(int)
    aggregate_columns = [
        "shots", "shots_on_target", "xg", "passes", "completed_passes", "final_third_entries",
        "pressures", "recoveries", "interceptions", "fouls", "yellow_cards", "red_cards",
        "carries", "duels_won",
    ]
    aggregate = frame.groupby(["match_id", "team"], as_index=False)[aggregate_columns].sum()
    possessions = (
        frame.dropna(subset=["possession"])
        .groupby(["match_id", "team"])["possession"]
        .nunique()
        .rename("possessions")
        .reset_index()
    )
    aggregate = aggregate.merge(possessions, on=["match_id", "team"], how="left")
    home = matches[
        ["match_id", "clean_date", "season_name", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    home = home.assign(
        team=home["home_team"], opponent=home["away_team"], is_home=1,
        goals_for=home["home_score"], goals_against=home["away_score"],
    )
    away = matches[
        ["match_id", "clean_date", "season_name", "home_team", "away_team", "home_score", "away_score"]
    ].copy()
    away = away.assign(
        team=away["away_team"], opponent=away["home_team"], is_home=0,
        goals_for=away["away_score"], goals_against=away["home_score"],
    )
    rows = pd.concat([home, away], ignore_index=True)
    rows["points"] = np.where(rows["goals_for"] > rows["goals_against"], 3, np.where(rows["goals_for"] == rows["goals_against"], 1, 0))
    rows = rows.merge(aggregate, on=["match_id", "team"], how="left", validate="one_to_one")
    rows[[*aggregate_columns, "possessions"]] = rows[[*aggregate_columns, "possessions"]].fillna(0)
    return rows.sort_values(["team", "clean_date", "match_id"], kind="mergesort").reset_index(drop=True)


def _historical_features(
    team_rows: pd.DataFrame, raw_metrics: tuple[str, ...], prefix: str
) -> tuple[pd.DataFrame, list[str]]:
    frame = team_rows.sort_values(["team", "clean_date", "match_id"], kind="mergesort").copy()
    groups = frame.groupby("team", sort=False)
    frame[f"{prefix}source_match_time"] = groups["clean_date"].shift(1)
    frame[f"{prefix}matches_played_prior"] = groups.cumcount().astype(float)
    frame[f"{prefix}rest_days"] = (
        frame["clean_date"] - groups["clean_date"].shift(1)
    ).dt.days.clip(lower=0, upper=60)
    features = [
        f"{prefix}source_match_time",
        f"{prefix}matches_played_prior",
        f"{prefix}rest_days",
    ]
    for metric in raw_metrics:
        lagged = groups[metric].shift(1)
        for window in (3, 5, 10):
            name = f"{prefix}{metric}_roll{window}"
            frame[name] = (
                lagged.groupby(frame["team"], sort=False)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            features.append(name)
        name = f"{prefix}{metric}_expanding"
        frame[name] = (
            lagged.groupby(frame["team"], sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        features.append(name)
    source = frame[f"{prefix}source_match_time"]
    if (source.notna() & source.ge(frame["clean_date"])).any():
        raise AssertionError(f"{prefix} historical feature leakage detected")
    return frame, features


def _pivot_home_away(
    team_features: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    home = team_features[team_features["is_home"].eq(1)][["match_id", *feature_columns]].copy()
    away = team_features[team_features["is_home"].eq(0)][["match_id", *feature_columns]].copy()
    home = home.rename(columns={column: f"home_{column}" for column in feature_columns})
    away = away.rename(columns={column: f"away_{column}" for column in feature_columns})
    merged = home.merge(away, on="match_id", validate="one_to_one")
    for column in feature_columns:
        if column.endswith("source_match_time"):
            continue
        merged[f"diff_{column}"] = merged[f"home_{column}"] - merged[f"away_{column}"]
    return merged


def build_prematch_features(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    p1_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    matches = matches.sort_values(["clean_date", "match_id"]).copy()
    event_rows = aggregate_event_team_matches(events, matches)
    event_history, conventional_team_features = _historical_features(event_rows, EVENT_METRICS, "event_")
    conventional = _pivot_home_away(event_history, conventional_team_features)

    network_rows = event_rows[
        ["match_id", "clean_date", "season_name", "home_team", "away_team", "team", "opponent", "is_home"]
    ].merge(
        p1_metrics[["match_id", "team", *NETWORK_METRICS]],
        on=["match_id", "team"],
        how="left",
        validate="one_to_one",
    )
    network_history, network_team_features = _historical_features(network_rows, NETWORK_METRICS, "p1_")
    network = _pivot_home_away(network_history, network_team_features)
    base = matches[
        [
            "match_id", "clean_date", "season_name", "match_week", "home_team", "away_team",
            "market_prob_H", "market_prob_D", "market_prob_A", "bookie_margin",
        ]
    ].copy()
    base["season_progress"] = pd.to_numeric(base["match_week"], errors="coerce") / 38.0
    day_of_year = base["clean_date"].dt.dayofyear
    base["calendar_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    base["calendar_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    base["split_role"] = assign_split_role(base)
    output = base.merge(conventional, on="match_id", validate="one_to_one").merge(network, on="match_id", validate="one_to_one")
    source_columns = [column for column in output if column.endswith("source_match_time")]
    for source_column in source_columns:
        if (output[source_column].notna() & output[source_column].ge(output["clean_date"])).any():
            raise AssertionError(f"Leakage in {source_column}")
    market = ["market_prob_A", "market_prob_D", "market_prob_H"]
    context = ["season_progress", "calendar_sin", "calendar_cos", "bookie_margin"]
    conventional_columns = [
        column
        for column in output
        if ("event_" in column and not column.endswith("source_match_time"))
    ] + context
    network_columns = [
        column
        for column in output
        if ("p1_" in column and not column.endswith("source_match_time"))
    ]
    groups = {
        "market": market,
        "conventional": conventional_columns,
        "network": network_columns,
        "conventional_market": [*conventional_columns, *market],
        "conventional_network": [*conventional_columns, *network_columns],
        "combined": [*conventional_columns, *network_columns, *market],
    }
    for name, columns in groups.items():
        if len(columns) != len(set(columns)):
            raise AssertionError(f"Duplicate columns in feature group {name}")
    return output, groups


def write_feature_artifacts(
    features: pd.DataFrame,
    groups: dict[str, list[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "prematch_features.parquet"
    features.to_parquet(feature_path, index=False)
    split_manifest = features[
        ["match_id", "clean_date", "season_name", "home_team", "away_team", "split_role"]
    ].copy()
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    schema_payload = {
        "class_order": list(CLASS_ORDER),
        "groups": groups,
        "split_counts": split_manifest["split_role"].value_counts().sort_index().to_dict(),
        "feature_file_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "leakage_rule": "Every historical source_match_time is strictly earlier than clean_date.",
        "final_test_use": "2016/17 rows are forbidden in fit, selection, calibration, and ablation code.",
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
