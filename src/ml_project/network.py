"""Paper-faithful multilayer passing and possession-transition networks.

The implementation follows P1's 4 x 5 (20-zone) two-layer construction. Rows
are source zones and columns are destination zones. Consequently the paper's
specified *right* dominant eigenvector is used even though other network
conventions sometimes use the left eigenvector for incoming prestige.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import GRID_X_BINS, GRID_Y_BINS, N_ZONES, PITCH_X_MAX, PITCH_Y_MAX


def zone_id(x: float, y: float) -> int:
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError("Zone coordinates must be finite")
    if x < 0 or x > PITCH_X_MAX or y < 0 or y > PITCH_Y_MAX:
        raise ValueError(f"Coordinate ({x}, {y}) is outside the StatsBomb pitch")
    x_bin = min(int(x / (PITCH_X_MAX / GRID_X_BINS)), GRID_X_BINS - 1)
    y_bin = min(int(y / (PITCH_Y_MAX / GRID_Y_BINS)), GRID_Y_BINS - 1)
    return y_bin * GRID_X_BINS + x_bin


def add_zones(events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with start and event-end zone columns."""

    frame = events.copy()
    valid_start = frame["location_x"].notna() & frame["location_y"].notna()
    frame["start_zone"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[valid_start, "start_zone"] = [
        zone_id(x, y)
        for x, y in frame.loc[valid_start, ["location_x", "location_y"]].itertuples(index=False, name=None)
    ]

    end_x = frame["location_x"].copy()
    end_y = frame["location_y"].copy()
    for prefix in ("pass", "carry", "shot", "goalkeeper"):
        x_col, y_col = f"{prefix}_end_x", f"{prefix}_end_y"
        if x_col in frame and y_col in frame:
            mask = frame[x_col].notna() & frame[y_col].notna()
            end_x.loc[mask] = frame.loc[mask, x_col]
            end_y.loc[mask] = frame.loc[mask, y_col]
    valid_end = end_x.notna() & end_y.notna()
    frame["end_zone"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[valid_end, "end_zone"] = [
        zone_id(x, y)
        for x, y in zip(end_x.loc[valid_end], end_y.loc[valid_end])
    ]
    return frame


@dataclass
class MatchNetwork:
    home_team: str
    away_team: str
    home_passes: np.ndarray
    away_passes: np.ndarray
    home_to_away: np.ndarray
    away_to_home: np.ndarray
    supra: np.ndarray


def _completed_pass_matrix(events: pd.DataFrame, team: str) -> np.ndarray:
    matrix = np.zeros((N_ZONES, N_ZONES), dtype=float)
    mask = (
        events["team"].eq(team)
        & events["type"].eq("Pass")
        & events["pass_outcome"].isna()
        & events["start_zone"].notna()
        & events["end_zone"].notna()
    )
    for source, target in events.loc[mask, ["start_zone", "end_zone"]].itertuples(index=False, name=None):
        matrix[int(source), int(target)] += 1.0
    return matrix


def _possession_segments(events: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = (
        events.dropna(subset=["period", "possession", "possession_team", "index"])
        .sort_values(["period", "index"], kind="mergesort")
        .copy()
    )
    boundaries = (
        ordered.groupby(["period", "possession"], as_index=False)
        .agg(first_event_index=("index", "min"))
        .sort_values(["period", "first_event_index", "possession"], kind="mergesort")
        .reset_index(drop=True)
    )
    boundaries["next_start"] = boundaries.groupby("period")["first_event_index"].shift(-1)
    segments: list[pd.DataFrame] = []
    for boundary in boundaries.itertuples(index=False):
        segment = ordered[
            ordered["period"].eq(boundary.period)
            & ordered["possession"].eq(boundary.possession)
        ].copy()
        # StatsBomb can tag a late event to the possession that has already
        # ended. P1 transitions use only spatial events before the next
        # possession's first chronological event.
        if pd.notna(boundary.next_start):
            segment = segment[segment["index"] < int(boundary.next_start)].copy()
        if segment.empty:
            segment = ordered[
                ordered["period"].eq(boundary.period)
                & ordered["index"].eq(int(boundary.first_event_index))
            ].copy()
        segments.append(segment)
    return segments


def _transition_matrices(
    events: pd.DataFrame, home_team: str, away_team: str
) -> tuple[np.ndarray, np.ndarray]:
    h2a = np.zeros((N_ZONES, N_ZONES), dtype=float)
    a2h = np.zeros((N_ZONES, N_ZONES), dtype=float)
    segments = _possession_segments(events)
    for current, following in zip(segments[:-1], segments[1:]):
        if int(current["period"].iloc[-1]) != int(following["period"].iloc[0]):
            continue
        current_team = current["possession_team"].dropna()
        next_team = following["possession_team"].dropna()
        if current_team.empty or next_team.empty:
            continue
        source_team, target_team = str(current_team.iloc[-1]), str(next_team.iloc[0])
        if source_team == target_team or {source_team, target_team} != {home_team, away_team}:
            continue
        # The original P1-style notebook defines a possession endpoint as the
        # location of its final spatial event (not a pass/carry end coordinate).
        source_rows = current[current["start_zone"].notna()]
        target_rows = following[following["start_zone"].notna()]
        if source_rows.empty or target_rows.empty:
            continue
        source = int(source_rows["start_zone"].iloc[-1])
        target = int(target_rows["start_zone"].iloc[0])
        (h2a if source_team == home_team else a2h)[source, target] += 1.0
    return h2a, a2h


def build_match_network(
    events: pd.DataFrame, home_team: str, away_team: str
) -> MatchNetwork:
    needed = {"type", "team", "pass_outcome", "possession", "possession_team", "period"}
    missing = needed - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    zoned = add_zones(events) if "start_zone" not in events else events.copy()
    home_passes = _completed_pass_matrix(zoned, home_team)
    away_passes = _completed_pass_matrix(zoned, away_team)
    h2a, a2h = _transition_matrices(zoned, home_team, away_team)
    supra = np.block([[home_passes, h2a], [a2h, away_passes]])
    return MatchNetwork(home_team, away_team, home_passes, away_passes, h2a, a2h, supra)


@dataclass
class CentralityResult:
    vector: np.ndarray
    eigenvalue: float
    residual: float


def dominant_right_eigenvector(adjacency: np.ndarray) -> CentralityResult:
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be square")
    if (adjacency < 0).any() or not np.isfinite(adjacency).all():
        raise ValueError("Adjacency must be finite and nonnegative")
    if not adjacency.any():
        vector = np.repeat(1.0 / len(adjacency), len(adjacency))
        return CentralityResult(vector, 0.0, 0.0)
    values, vectors = np.linalg.eig(adjacency)
    spectral_radius = np.max(np.abs(values))
    candidate_indices = np.flatnonzero(np.isclose(np.abs(values), spectral_radius, rtol=1e-9, atol=1e-12))
    # Stable choice for reducible/periodic graphs: prefer the candidate with the
    # largest real part, then convert the Perron vector to a nonnegative scale.
    selected = int(candidate_indices[np.argmax(values[candidate_indices].real)])
    eigenvalue = float(values[selected].real)
    vector = np.real_if_close(vectors[:, selected]).real
    if vector.sum() < 0:
        vector = -vector
    vector = np.clip(vector, 0.0, None)
    if vector.sum() <= 1e-15:
        vector = np.abs(np.real(vectors[:, selected]))
    vector = vector / vector.sum()
    residual = float(np.linalg.norm(adjacency @ vector - eigenvalue * vector))
    return CentralityResult(vector, eigenvalue, residual)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def layer_zone_metrics(
    intra: np.ndarray, outgoing_inter: np.ndarray, incoming_inter: np.ndarray
) -> pd.DataFrame:
    pass_in = intra.sum(axis=0)
    inter_out = outgoing_inter.sum(axis=1)
    inter_in = incoming_inter.sum(axis=0)
    total_in = pass_in + inter_in
    leakage = _safe_ratio(inter_out, total_in)
    recovery = _safe_ratio(inter_in, total_in)
    switching = _safe_ratio(inter_in + inter_out, pass_in)
    return pd.DataFrame(
        {
            "zone": np.arange(N_ZONES),
            "pass_in_degree": pass_in,
            "inter_out_degree": inter_out,
            "inter_in_degree": inter_in,
            "total_in_degree": total_in,
            "leakage": leakage,
            "recovery": recovery,
            "switching_zone_ratio": switching,
        }
    )


def summarize_match_network(network: MatchNetwork) -> tuple[pd.DataFrame, pd.DataFrame]:
    supra_c = dominant_right_eigenvector(network.supra)
    rows, zones = [], []
    specifications = (
        ("home", network.home_team, network.home_passes, network.home_to_away, network.away_to_home, slice(0, N_ZONES)),
        ("away", network.away_team, network.away_passes, network.away_to_home, network.home_to_away, slice(N_ZONES, 2 * N_ZONES)),
    )
    for layer, team, intra, outgoing, incoming, centrality_slice in specifications:
        zone_frame = layer_zone_metrics(intra, outgoing, incoming)
        zone_frame["supra_centrality"] = supra_c.vector[centrality_slice]
        individual = dominant_right_eigenvector(intra)
        zone_frame["individual_layer_centrality"] = individual.vector
        zone_frame["layer"] = layer
        zone_frame["team"] = team
        zones.append(zone_frame)
        finite_switch = zone_frame["switching_zone_ratio"].dropna()
        rows.append(
            {
                "layer": layer,
                "team": team,
                "accumulated_eigenvector_centrality": float(zone_frame["supra_centrality"].sum()),
                "maximum_zone_centrality": float(zone_frame["supra_centrality"].max()),
                "mean_zone_centrality": float(zone_frame["supra_centrality"].mean()),
                "individual_layer_eigenvalue": individual.eigenvalue,
                "individual_layer_eigen_residual": individual.residual,
                "supra_eigenvalue": supra_c.eigenvalue,
                "supra_eigen_residual": supra_c.residual,
                "mean_zone_leakage": float(zone_frame["leakage"].mean()),
                "median_zone_leakage": float(zone_frame["leakage"].median()),
                "maximum_zone_leakage": float(zone_frame["leakage"].max()),
                "mean_zone_recovery": float(zone_frame["recovery"].mean()),
                "median_zone_recovery": float(zone_frame["recovery"].median()),
                "maximum_zone_recovery": float(zone_frame["recovery"].max()),
                "defined_leakage_zones": int(zone_frame["leakage"].notna().sum()),
                "defined_recovery_zones": int(zone_frame["recovery"].notna().sum()),
                "completed_passes": int(intra.sum()),
                "possession_losses": int(outgoing.sum()),
                "possession_recoveries": int(incoming.sum()),
                "finite_switching_zones": int(len(finite_switch)),
                "undefined_switching_zones": int(N_ZONES - len(finite_switch)),
                "switching_factor": float(0.5 * finite_switch.sum()),
            }
        )
    return pd.DataFrame(rows), pd.concat(zones, ignore_index=True)


def validate_attacking_orientation(events: pd.DataFrame) -> pd.DataFrame:
    """Empirically verify StatsBomb's team-in-possession left-to-right frame.

    Both home and away shots should originate near x=120, and completed passes
    should on average have positive x displacement. This diagnostic avoids an
    unnecessary home/away half flip that would corrupt the common zone frame.
    """

    rows = []
    home_flag = events["team"].eq(events["home_team"])
    for side, side_mask in (("home", home_flag), ("away", ~home_flag)):
        shots = events[side_mask & events["type"].eq("Shot") & events["location_x"].notna()]
        passes = events[
            side_mask
            & events["type"].eq("Pass")
            & events["location_x"].notna()
            & events["pass_end_x"].notna()
        ]
        rows.append(
            {
                "side": side,
                "shot_count": len(shots),
                "mean_shot_x": float(shots["location_x"].mean()),
                "median_shot_x": float(shots["location_x"].median()),
                "pass_count": len(passes),
                "mean_pass_dx": float((passes["pass_end_x"] - passes["location_x"]).mean()),
                "orientation_supported": bool(
                    shots["location_x"].mean() > PITCH_X_MAX / 2
                    and (passes["pass_end_x"] - passes["location_x"]).mean() > 0
                ),
            }
        )
    return pd.DataFrame(rows)
