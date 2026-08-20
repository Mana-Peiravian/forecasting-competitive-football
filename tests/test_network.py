import numpy as np
import pandas as pd

from ml_project.network import (
    build_match_network,
    dominant_right_eigenvector,
    layer_zone_metrics,
    summarize_match_network,
    zone_id,
)


def _event(**overrides):
    row = {
        "id": "e",
        "index": 1,
        "period": 1,
        "event_seconds": 0.0,
        "type": "Pass",
        "team": "Home",
        "possession": 1,
        "possession_team": "Home",
        "pass_outcome": np.nan,
        "location_x": 1.0,
        "location_y": 1.0,
        "pass_end_x": 31.0,
        "pass_end_y": 1.0,
        "carry_end_x": np.nan,
        "carry_end_y": np.nan,
        "shot_end_x": np.nan,
        "shot_end_y": np.nan,
        "goalkeeper_end_x": np.nan,
        "goalkeeper_end_y": np.nan,
    }
    row.update(overrides)
    return row


def test_pitch_zone_boundaries_are_clipped_to_twenty_zones():
    assert zone_id(0, 0) == 0
    assert zone_id(29.999, 15.999) == 0
    assert zone_id(30, 16) == 5
    assert zone_id(120, 80) == 19


def test_supra_blocks_and_possession_transition_direction():
    events = pd.DataFrame(
        [
            _event(id="h1", index=1),
            _event(
                id="a1",
                index=2,
                event_seconds=2.0,
                type="Ball Recovery",
                team="Away",
                possession=2,
                possession_team="Away",
                location_x=61.0,
                location_y=1.0,
                pass_end_x=np.nan,
                pass_end_y=np.nan,
            ),
        ]
    )
    network = build_match_network(events, "Home", "Away")
    assert network.supra.shape == (40, 40)
    assert network.home_passes[0, 1] == 1
    # P1 uses the location of the final spatial event, not its pass endpoint.
    assert network.home_to_away[0, 2] == 1
    assert network.supra[0, 20 + 2] == 1
    assert network.away_to_home.sum() == 0


def test_right_eigenvector_satisfies_eigen_equation():
    adjacency = np.array([[1.0, 2.0], [1.0, 0.0]])
    result = dominant_right_eigenvector(adjacency)
    assert np.isclose(result.vector.sum(), 1.0)
    assert result.residual < 1e-10
    assert np.all(result.vector >= 0)


def test_p1_zone_ratios_and_switching_formula():
    intra = np.zeros((20, 20))
    outgoing = np.zeros((20, 20))
    incoming = np.zeros((20, 20))
    intra[0, 1] = 2
    outgoing[1, 3] = 1
    incoming[4, 1] = 1
    zone = layer_zone_metrics(intra, outgoing, incoming).set_index("zone")
    assert np.isclose(zone.loc[1, "leakage"], 1 / 3)
    assert np.isclose(zone.loc[1, "recovery"], 1 / 3)
    assert np.isclose(zone.loc[1, "switching_zone_ratio"], 1.0)
