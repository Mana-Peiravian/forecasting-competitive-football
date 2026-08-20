import numpy as np

from ml_project.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
    ranked_probability_score,
)


def test_perfect_probabilities_have_zero_proper_scores():
    y = np.array(["A", "D", "H"])
    p = np.eye(3)
    assert np.isclose(ranked_probability_score(y, p), 0.0)
    assert np.isclose(multiclass_brier_score(y, p), 0.0)
    assert np.isclose(expected_calibration_error(y, p), 0.0)


def test_rps_hand_calculation_for_certain_opposite_outcome():
    # True A, predicted H: both cumulative boundaries are wrong, then divide by 2.
    assert np.isclose(ranked_probability_score(["A"], np.array([[0.0, 0.0, 1.0]])), 1.0)


def test_probability_rows_are_normalized_centrally():
    metrics = classification_metrics(["A"], np.array([[2.0, 0.0, 0.0]]))
    assert metrics["rps"] < 1e-20
