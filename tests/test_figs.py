import numpy as np
import pytest
from sklearn.datasets import make_classification

from ml_project.figs import ScratchFIGSClassifier, ScratchFIGSRegressor


def test_regression_paper_toy_uses_three_features_with_three_splits():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] > 0).astype(float) + ((X[:, 1] > 0) & (X[:, 2] > 0)).astype(float)
    model = ScratchFIGSRegressor(max_rules=3, random_state=7).fit(X, y)
    used = {int(item["feature"]) for item in model.split_history_}
    assert model.complexity_ == 3
    assert used == {0, 1, 2}
    assert np.mean((model.predict(X) - y) ** 2) < 0.03


def test_existing_tree_can_be_revisited_after_another_tree_is_started():
    rng = np.random.default_rng(9)
    X = rng.uniform(-1, 1, size=(3000, 3))
    y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 0).astype(float) + (
        (X[:, 0] > 0) & (X[:, 2] > 0)
    ).astype(float)
    model = ScratchFIGSRegressor(max_rules=4, random_state=9).fit(X, y)
    tree_sequence = [int(item["tree"]) for item in model.split_history_]
    assert len(set(tree_sequence)) >= 2
    assert any(
        tree_sequence[i] == tree_sequence[0]
        for i in range(2, len(tree_sequence))
        if tree_sequence[i - 1] != tree_sequence[0]
    )


def test_multiclass_probabilities_are_valid_and_deterministic():
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        random_state=11,
    )
    first = ScratchFIGSClassifier(max_rules=8, random_state=11).fit(X, y)
    second = ScratchFIGSClassifier(max_rules=8, random_state=11).fit(X, y)
    p1, p2 = first.predict_proba(X), second.predict_proba(X)
    assert p1.shape == (500, 3)
    assert np.allclose(p1.sum(axis=1), 1.0)
    assert np.all((p1 > 0) & (p1 < 1))
    assert np.allclose(p1, p2)
    assert first.split_history_ == second.split_history_


def test_zero_rule_model_is_a_valid_constant_estimator():
    X = np.arange(10.0).reshape(-1, 1)
    y = np.arange(10.0)
    model = ScratchFIGSRegressor(max_rules=0).fit(X, y)
    assert model.complexity_ == 0
    assert np.allclose(model.predict(X), y.mean())


def test_backfitting_does_not_increase_training_squared_error():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(500, 4))
    y = X[:, 0] + (X[:, 1] > 0) * X[:, 2] + rng.normal(scale=0.1, size=500)
    plain = ScratchFIGSRegressor(max_rules=6, random_state=5, backfit=False).fit(X, y)
    fitted = ScratchFIGSRegressor(
        max_rules=6, random_state=5, backfit=True, backfit_iterations=5
    ).fit(X, y)
    plain_mse = np.mean((plain.predict(X) - y) ** 2)
    fitted_mse = np.mean((fitted.predict(X) - y) ** 2)
    assert fitted_mse <= plain_mse + 1e-12


def test_independent_imodels_reference_parity_without_production_import():
    """The independent reference is an optional test dependency only."""

    reference = pytest.importorskip("imodels")
    rng = np.random.default_rng(123)
    X = rng.normal(size=(300, 5))
    y_reg = X[:, 0] + (X[:, 1] > 0) * X[:, 2]
    scratch_reg = ScratchFIGSRegressor(max_rules=8, random_state=42).fit(X, y_reg)
    reference_reg = reference.FIGSRegressor(max_rules=8, random_state=42).fit(X, y_reg)
    assert np.allclose(scratch_reg.predict(X), reference_reg.predict(X), atol=1e-12)

    y_cls = np.where(X[:, 0] + 0.5 * X[:, 1] > 0.8, "H", np.where(X[:, 0] < -0.7, "A", "D"))
    scratch_cls = ScratchFIGSClassifier(max_rules=8, random_state=42).fit(X, y_cls)
    reference_cls = reference.FIGSClassifier(max_rules=8, random_state=42).fit(X, y_cls)
    assert np.array_equal(scratch_cls.classes_, reference_cls.classes_)
    assert np.allclose(
        scratch_cls.predict_proba(X), reference_cls.predict_proba(X), atol=1e-12
    )
