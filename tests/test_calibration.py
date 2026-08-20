import numpy as np

from ml_project.calibration import MulticlassProbabilityCalibrator, PhaseCalibrator


def test_calibrators_return_normalized_multiclass_probabilities():
    rng = np.random.default_rng(2)
    p = rng.dirichlet([2, 2, 2], size=120)
    y = rng.integers(0, 3, size=120)
    for method in ("platt", "isotonic"):
        calibrated = MulticlassProbabilityCalibrator(method).fit(p, y).predict_proba(p)
        assert calibrated.shape == p.shape
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert np.all(calibrated >= 0)


def test_phase_calibration_preserves_row_order():
    rng = np.random.default_rng(3)
    minutes = np.tile(np.arange(0, 91, 5), 12)
    p = rng.dirichlet([2, 2, 2], size=len(minutes))
    y = rng.integers(0, 3, size=len(minutes))
    model = PhaseCalibrator("platt").fit(p, y, minutes)
    output = model.predict_proba(p, minutes)
    assert output.shape == p.shape
    assert np.allclose(output.sum(axis=1), 1.0)

