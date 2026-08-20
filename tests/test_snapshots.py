import numpy as np
import pandas as pd

from ml_project.snapshots import build_match_snapshots


def _snapshot_event(event_id, seconds, event_type="Pass", team="Home"):
    return {
        "id": event_id,
        "event_seconds_exact": seconds,
        "type": event_type,
        "team": team,
        "pass_outcome": np.nan,
        "pass_end_x": 90.0,
        "carry_end_x": np.nan,
        "location_x": 70.0,
        "ball_recovery_recovery_failure": np.nan,
        "foul_committed_card": np.nan,
        "bad_behaviour_card": np.nan,
        "shot_statsbomb_xg": np.nan,
        "shot_outcome": np.nan,
        "duel_outcome": np.nan,
    }


def test_exact_snapshot_boundary_excludes_one_millisecond_after_cutoff():
    events = pd.DataFrame(
        [
            _snapshot_event("before", 1799.999),
            _snapshot_event("at", 1800.0),
            _snapshot_event("after", 1800.001),
        ]
    )
    match = pd.Series(
        {"match_id": 1, "clean_date": pd.Timestamp("2020-01-01"), "season_name": "x", "home_team": "Home", "away_team": "Away"}
    )
    prematch = pd.Series({"split_role": "train", "f": 1.0})
    snapshot = build_match_snapshots(events, match, prematch, ["f"], [30]).iloc[0]
    assert snapshot["home_cum_passes"] == 2
    assert snapshot["feature_source_max_seconds"] == 1800.0
    assert snapshot["prematch_f"] == 1.0

