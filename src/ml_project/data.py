"""Data acquisition, normalization, odds integration, and provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from statsbombpy import sb
from statsbombpy.helpers import flatten_event

from .config import CLASS_ORDER, DATA_DIR, NORMALIZED_DIR, PROCESSED_DIR, ensure_directories

STATSBOMB_RAW_ROOT = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
FOOTBALL_DATA_URLS = {
    "2015/2016": "https://www.football-data.co.uk/mmz4281/1516/SP1.csv",
    "2016/2017": "https://www.football-data.co.uk/mmz4281/1617/SP1.csv",
}

TEAM_ALIAS_MAP = {
    "Athletic Club": "Ath Bilbao",
    "Atlético Madrid": "Ath Madrid",
    "Celta Vigo": "Celta",
    "RC Deportivo La Coruña": "La Coruna",
    "Eibar": "Eibar",
    "Espanyol": "Espanol",
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Getafe": "Getafe",
    "Granada": "Granada",
    "Las Palmas": "Las Palmas",
    "Levante UD": "Levante",
    "Málaga": "Malaga",
    "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Sociedad",
    "Sevilla": "Sevilla",
    "Sporting Gijón": "Sp Gijon",
    "Valencia": "Valencia",
    "Villarreal": "Villarreal",
    "Deportivo Alavés": "Alaves",
    "Leganés": "Leganes",
    "Osasuna": "Osasuna",
}

EVENT_COLUMNS = [
    "match_id", "match_date", "home_team", "away_team", "id", "index", "period",
    "timestamp", "minute", "second", "event_seconds", "event_seconds_exact",
    "event_minute_decimal", "type", "team", "team_id", "player", "player_id",
    "position", "possession", "possession_team", "possession_team_id", "play_pattern",
    "under_pressure", "counterpress", "duration", "location_x", "location_y",
    "pass_recipient", "pass_recipient_id", "pass_length", "pass_angle", "pass_height",
    "pass_type", "pass_outcome", "pass_body_part", "pass_cross", "pass_cut_back",
    "pass_switch", "pass_through_ball", "pass_shot_assist", "pass_goal_assist",
    "pass_end_x", "pass_end_y", "carry_end_x", "carry_end_y", "shot_statsbomb_xg",
    "shot_outcome", "shot_type", "shot_body_part", "shot_technique", "shot_first_time",
    "shot_one_on_one", "shot_end_x", "shot_end_y", "shot_end_z", "duel_type",
    "duel_outcome", "interception_outcome", "ball_recovery_recovery_failure",
    "clearance_body_part", "block_deflection", "dribble_outcome", "foul_committed_card",
    "foul_won_defensive", "goalkeeper_type", "goalkeeper_outcome", "goalkeeper_position",
    "goalkeeper_technique", "goalkeeper_body_part", "goalkeeper_end_x",
    "goalkeeper_end_y", "substitution_replacement", "substitution_outcome",
    "bad_behaviour_card",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str, retries: int = 4) -> object:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # pragma: no cover - network-specific branch
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to acquire {url}") from last_error


def acquire_matches(competition_id: int, season_id: int, season_name: str) -> pd.DataFrame:
    frame = sb.matches(competition_id=competition_id, season_id=season_id).copy()
    if frame.empty or frame["match_id"].duplicated().any():
        raise ValueError(f"Invalid match table for {competition_id}/{season_id}")
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    frame["season_name"] = season_name
    stable_columns = [
        "match_id", "match_date", "kick_off", "home_score", "away_score", "match_status",
        "match_status_360", "last_updated", "last_updated_360", "match_week",
        "competition_id", "competition_country_name", "competition_name", "competition",
        "season_id", "season", "season_name", "home_team_id", "home_team",
        "home_team_gender", "home_team_country_id", "home_team_country_name",
        "away_team_id", "away_team", "away_team_gender", "away_team_country_id",
        "away_team_country_name", "competition_stage_id", "competition_stage", "stadium_id",
        "stadium", "stadium_country_id", "stadium_country_name", "referee_id", "referee",
        "referee_country_id", "referee_country_name", "data_version", "shot_fidelity_version",
        "xy_fidelity_version",
    ]
    frame = frame[[column for column in stable_columns if column in frame]].copy()
    frame = frame.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return frame


def _extract_coordinate(value, index: int) -> float:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > index:
        try:
            return float(value[index])
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def normalize_events(raw_events: list[dict], match_row: pd.Series) -> pd.DataFrame:
    flattened = [flatten_event(copy.deepcopy(event), True) for event in raw_events]
    frame = pd.DataFrame(flattened)
    frame["match_id"] = int(match_row["match_id"])
    frame["match_date"] = pd.Timestamp(match_row["match_date"])
    frame["home_team"] = str(match_row["home_team"])
    frame["away_team"] = str(match_row["away_team"])
    coordinate_specs = {
        "location": ("location_x", "location_y"),
        "pass_end_location": ("pass_end_x", "pass_end_y"),
        "carry_end_location": ("carry_end_x", "carry_end_y"),
        "shot_end_location": ("shot_end_x", "shot_end_y", "shot_end_z"),
        "goalkeeper_end_location": ("goalkeeper_end_x", "goalkeeper_end_y"),
    }
    for source, destinations in coordinate_specs.items():
        values = frame[source] if source in frame else pd.Series([None] * len(frame))
        for index, destination in enumerate(destinations):
            frame[destination] = values.map(lambda value, i=index: _extract_coordinate(value, i))
    frame["minute"] = pd.to_numeric(frame["minute"], errors="raise").astype(int)
    frame["second"] = pd.to_numeric(frame["second"], errors="raise").astype(int)
    clock = pd.to_timedelta(frame["timestamp"].astype(str), errors="coerce").dt.total_seconds()
    fractional = (clock - np.floor(clock)).fillna(0.0)
    frame["event_seconds"] = frame["minute"] * 60.0 + frame["second"]
    frame["event_seconds_exact"] = frame["event_seconds"] + fractional
    frame["event_minute_decimal"] = frame["event_seconds_exact"] / 60.0
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[EVENT_COLUMNS].sort_values(["period", "index"], kind="mergesort")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"Invalid event identity for match {match_row['match_id']}")
    return frame.reset_index(drop=True)


def acquire_event_season(
    matches: pd.DataFrame,
    output_path: Path,
    cache_dir: Path,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch raw open-data JSON concurrently, normalize, validate, and persist."""

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        if existing["match_id"].nunique() == matches["match_id"].nunique():
            return existing
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(row_dict: dict) -> tuple[int, pd.DataFrame, str]:
        match_id = int(row_dict["match_id"])
        cache_path = cache_dir / f"{match_id}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            source = "cache"
        else:
            raw = _request_json(f"{STATSBOMB_RAW_ROOT}/events/{match_id}.json")
            cache_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            source = "statsbomb-open-data"
        return match_id, normalize_events(raw, pd.Series(row_dict)), source

    frames: dict[int, pd.DataFrame] = {}
    logs: list[dict[str, object]] = []
    records = matches[["match_id", "match_date", "home_team", "away_team"]].to_dict("records")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, row): row for row in records}
        for future in as_completed(futures):
            row = futures[future]
            started = time.perf_counter()
            try:
                match_id, frame, source = future.result()
                frames[match_id] = frame
                logs.append({"match_id": match_id, "status": "success", "source": source, "events": len(frame)})
            except Exception as error:
                logs.append({"match_id": int(row["match_id"]), "status": "failed", "error": repr(error)})
    log = pd.DataFrame(logs).sort_values("match_id")
    log_path = output_path.with_name(output_path.stem + "_ingestion_log.csv")
    log.to_csv(log_path, index=False)
    failures = log[log["status"].eq("failed")]
    if not failures.empty:
        raise RuntimeError(f"Event ingestion failed for {len(failures)} matches; see {log_path}")
    combined = pd.concat([frames[int(mid)] for mid in matches["match_id"]], ignore_index=True)
    if combined["match_id"].nunique() != len(matches):
        raise ValueError("Event coverage does not equal match coverage")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return combined


def acquire_lineups(matches: pd.DataFrame, output_path: Path, max_workers: int = 12) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_parquet(output_path)

    def fetch(match_id: int) -> list[dict[str, object]]:
        payload = _request_json(f"{STATSBOMB_RAW_ROOT}/lineups/{match_id}.json")
        rows = []
        for team in payload:
            for player in team.get("lineup", []):
                positions = player.get("positions", [])
                rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team.get("team_id"),
                        "team": team.get("team_name"),
                        "player_id": player.get("player_id"),
                        "player": player.get("player_name"),
                        "player_nickname": player.get("player_nickname"),
                        "jersey_number": player.get("jersey_number"),
                        "country_id": (player.get("country") or {}).get("id"),
                        "country": (player.get("country") or {}).get("name"),
                        "starter": bool(any(position.get("start_reason") == "Starting XI" for position in positions)),
                        "positions_json": json.dumps(positions, ensure_ascii=False, sort_keys=True),
                        "cards_json": json.dumps(player.get("cards", []), ensure_ascii=False, sort_keys=True),
                    }
                )
        return rows

    all_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, int(mid)): int(mid) for mid in matches["match_id"]}
        for future in as_completed(futures):
            all_rows.extend(future.result())
    frame = pd.DataFrame(all_rows).sort_values(["match_id", "team_id", "jersey_number"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def acquire_odds(season_name: str, raw_dir: Path) -> pd.DataFrame:
    url = FOOTBALL_DATA_URLS[season_name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = f"football_data_sp1_{season_name.replace('/', '_')}.csv"
    path = raw_dir / filename
    if not path.exists():
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    odds = pd.read_csv(path)
    required = ["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
    missing = set(required) - set(odds)
    if missing:
        raise ValueError(f"Odds source is missing {sorted(missing)}")
    odds = odds[required].copy()
    odds["clean_date"] = pd.to_datetime(odds["Date"], dayfirst=True, errors="raise")
    odds["season_name"] = season_name
    odds["source_url"] = url
    return odds


def integrate_odds(matches: pd.DataFrame, odds: pd.DataFrame, season_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = matches.copy()
    left["clean_date"] = pd.to_datetime(left["match_date"])
    left["odds_home_team"] = left["home_team"].map(TEAM_ALIAS_MAP)
    left["odds_away_team"] = left["away_team"].map(TEAM_ALIAS_MAP)
    if left[["odds_home_team", "odds_away_team"]].isna().any().any():
        missing = set(left.loc[left["odds_home_team"].isna(), "home_team"]) | set(
            left.loc[left["odds_away_team"].isna(), "away_team"]
        )
        raise ValueError(f"Missing explicit team aliases: {sorted(missing)}")
    keys_left = ["clean_date", "odds_home_team", "odds_away_team"]
    keys_right = ["clean_date", "HomeTeam", "AwayTeam"]
    if odds.duplicated(keys_right).any():
        raise ValueError("Duplicate odds join keys detected")
    merged = left.merge(odds, how="left", left_on=keys_left, right_on=keys_right, indicator=True)
    diagnostic = merged[
        ["match_id", "clean_date", "home_team", "away_team", "odds_home_team", "odds_away_team", "_merge"]
    ].copy()
    if merged["_merge"].ne("both").any():
        raise ValueError(f"Unmatched StatsBomb fixtures: {int(merged['_merge'].ne('both').sum())}")
    raw = merged[["B365H", "B365D", "B365A"]].astype(float)
    implied = 1.0 / raw.to_numpy()
    margin = implied.sum(axis=1)
    market = implied / margin[:, None]
    merged["bookie_margin"] = margin
    for index, label in enumerate(("H", "D", "A")):
        merged[f"implied_{label}"] = implied[:, index]
        merged[f"market_prob_{label}"] = market[:, index]
    merged["season_name"] = season_name
    keep = [
        "match_id", "clean_date", "season_name", "competition_id", "season_id", "match_week",
        "home_team_id", "home_team", "away_team_id", "away_team", "home_score", "away_score",
        "B365H", "B365D", "B365A", "implied_H", "implied_D", "implied_A", "bookie_margin",
        "market_prob_H", "market_prob_D", "market_prob_A",
    ]
    return merged[keep].sort_values(["clean_date", "match_id"]).reset_index(drop=True), diagnostic


def outcome_and_margin(matches: pd.DataFrame) -> pd.DataFrame:
    frame = matches[["match_id", "season_name", "clean_date", "home_score", "away_score"]].copy()
    raw_margin = frame["home_score"] - frame["away_score"]
    frame["goal_margin"] = raw_margin.clip(-5, 5).astype(int)
    frame["outcome"] = np.where(raw_margin > 0, "H", np.where(raw_margin < 0, "A", "D"))
    if set(frame["outcome"].unique()) - set(CLASS_ORDER):
        raise ValueError("Invalid outcome target")
    return frame


def build_relational_tables(
    season_specs: Iterable[tuple[int, int, str]],
    include_final_events: bool = True,
) -> dict[str, Path]:
    ensure_directories()
    raw_dir = DATA_DIR / "raw_sources"
    cache_dir = DATA_DIR / "cache" / "statsbomb_events"
    match_frames, integrated_frames, event_frames, lineup_frames = [], [], [], []
    diagnostics = []
    for competition_id, season_id, season_name in season_specs:
        matches = acquire_matches(competition_id, season_id, season_name)
        match_frames.append(matches)
        odds = acquire_odds(season_name, raw_dir)
        integrated, diagnostic = integrate_odds(matches, odds, season_name)
        integrated_frames.append(integrated)
        diagnostic["season_name"] = season_name
        diagnostics.append(diagnostic)
        season_tag = season_name.replace("/", "_")
        if season_name == "2015/2016":
            old_path = PROCESSED_DIR / "events_laliga_2015_2016.parquet"
            events = pd.read_parquet(old_path)
            if "event_seconds_exact" not in events:
                clock = pd.to_timedelta(events["timestamp"].astype(str), errors="coerce").dt.total_seconds()
                events["event_seconds_exact"] = events["event_seconds"] + (clock - np.floor(clock)).fillna(0.0)
        elif include_final_events:
            events = acquire_event_season(
                matches,
                NORMALIZED_DIR / f"events_laliga_{season_tag}.parquet",
                cache_dir / season_tag,
            )
        else:
            events = pd.DataFrame(columns=EVENT_COLUMNS)
        event_frames.append(events)
        lineup_frames.append(
            acquire_lineups(matches, NORMALIZED_DIR / f"lineups_laliga_{season_tag}.parquet")
        )

    competitions = sb.competitions()
    wanted = {(competition_id, season_id) for competition_id, season_id, _ in season_specs}
    competitions = competitions[
        competitions.apply(lambda row: (int(row["competition_id"]), int(row["season_id"])) in wanted, axis=1)
    ].copy()
    matches_all = pd.concat(match_frames, ignore_index=True)
    integrated_all = pd.concat(integrated_frames, ignore_index=True)
    events_all = pd.concat(event_frames, ignore_index=True)
    lineups_all = pd.concat(lineup_frames, ignore_index=True)
    teams = pd.concat(
        [
            matches_all[["home_team_id", "home_team"]].rename(columns={"home_team_id": "team_id", "home_team": "team"}),
            matches_all[["away_team_id", "away_team"]].rename(columns={"away_team_id": "team_id", "away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates().sort_values("team_id")
    players = events_all[["player_id", "player"]].dropna().drop_duplicates().sort_values("player_id")
    odds_all = integrated_all[
        [column for column in integrated_all if column.startswith(("B365", "implied_", "market_prob_"))]
        + ["match_id", "season_name", "clean_date"]
    ]
    # Selected La Liga seasons have no StatsBomb 360 files. The empty table and
    # availability manifest make the absence explicit and machine-readable.
    three_sixty = pd.DataFrame(columns=["match_id", "event_uuid", "visible_area", "freeze_frame"])
    availability = competitions[
        ["competition_id", "season_id", "season_name", "match_available_360", "match_available"]
    ].copy()
    aliases = pd.DataFrame(sorted(TEAM_ALIAS_MAP.items()), columns=["statsbomb_team", "football_data_team"])
    tables = {
        "competitions": competitions,
        "matches": matches_all,
        "teams": teams,
        "players": players,
        "lineups": lineups_all,
        "events": events_all,
        "three_sixty": three_sixty,
        "three_sixty_availability": availability,
        "odds": odds_all,
        "integrated_matches": integrated_all,
        "odds_join_diagnostics": pd.concat(diagnostics, ignore_index=True),
        "team_aliases": aliases,
    }
    paths = {}
    for name, table in tables.items():
        path = NORMALIZED_DIR / f"{name}.parquet"
        table.to_parquet(path, index=False)
        paths[name] = path
    labels = outcome_and_margin(integrated_all)
    labels_path = NORMALIZED_DIR / "sealed_targets.parquet"
    labels.to_parquet(labels_path, index=False)
    paths["sealed_targets"] = labels_path
    development_labels = labels[labels["season_name"].ne("2016/2017")].copy()
    final_labels = labels[labels["season_name"].eq("2016/2017")].copy()
    development_path = NORMALIZED_DIR / "development_targets.parquet"
    final_path = NORMALIZED_DIR / "final_targets_sealed.parquet"
    development_labels.to_parquet(development_path, index=False)
    final_labels.to_parquet(final_path, index=False)
    paths["development_targets"] = development_path
    paths["final_targets_sealed"] = final_path
    manifest = {
        "sources": {
            "statsbomb": "https://github.com/statsbomb/open-data",
            **{f"football_data_{season}": url for season, url in FOOTBALL_DATA_URLS.items()},
        },
        "tables": {
            name: {
                "path": str(path.relative_to(DATA_DIR.parent)),
                "sha256": sha256_file(path),
                "rows": (
                    len(development_labels) if name == "development_targets" else
                    len(final_labels) if name == "final_targets_sealed" else
                    len(tables.get(name, labels))
                ),
            }
            for name, path in paths.items()
        },
        "notes": [
            "Odds joins use only date and explicit home/away team aliases.",
            "Scores and match statistics are not odds-join keys.",
            "2016/17 open StatsBomb La Liga contains 34 available matches, not the full league season.",
            "The sealed target file is excluded from all selection, calibration, and feature-building functions.",
        ],
    }
    manifest_path = NORMALIZED_DIR / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths
