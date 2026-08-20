"""Model factories, probability alignment, and lightweight resource telemetry."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import psutil
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .config import RANDOM_SEED
from .figs import ScratchFIGSClassifier, ScratchFIGSRegressor


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_classifier(name: str, params: dict | None = None):
    params = dict(params or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "svc_rbf":
        return Pipeline(
            [
                ("imputer", _imputer()),
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=RANDOM_SEED, **params)),
            ]
        )
    if name == "random_forest":
        defaults = dict(n_estimators=200, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", RandomForestClassifier(**defaults))])
    if name == "gradient_boosting":
        defaults = dict(n_estimators=100, learning_rate=0.04, max_depth=2, random_state=RANDOM_SEED)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", GradientBoostingClassifier(**defaults))])
    if name == "xgboost":
        defaults = dict(
            n_estimators=150, learning_rate=0.04, max_depth=3, subsample=0.85,
            colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=2,
        )
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", XGBClassifier(**defaults))])
    if name == "lightgbm":
        defaults = dict(
            n_estimators=150, learning_rate=0.04, num_leaves=15, max_depth=4,
            min_child_samples=15, random_state=RANDOM_SEED, n_jobs=2, verbosity=-1,
        )
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", LGBMClassifier(**defaults))])
    if name == "p1_logistic":
        defaults = dict(C=0.3, max_iter=3000, random_state=RANDOM_SEED)
        defaults.update(params)
        return Pipeline(
            [("imputer", _imputer()), ("scaler", StandardScaler()), ("model", LogisticRegression(**defaults))]
        )
    if name == "scratch_figs":
        defaults = dict(max_rules=12, max_trees=5, random_state=RANDOM_SEED, backfit=True)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", ScratchFIGSClassifier(**defaults))])
    raise KeyError(name)


def make_regressor(name: str, params: dict | None = None):
    params = dict(params or {})
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "svr_rbf":
        defaults = dict(C=1.0, epsilon=0.2, gamma="scale")
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", SVR(**defaults))])
    if name == "kernel_ridge_exact":
        defaults = dict(alpha=1.0, kernel="rbf", gamma=0.01)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", KernelRidge(**defaults))])
    if name == "nystroem_ridge":
        n_components = params.pop("n_components", 100)
        gamma = params.pop("gamma", 0.01)
        alpha = params.pop("alpha", 1.0)
        return Pipeline(
            [
                ("imputer", _imputer()),
                ("scaler", StandardScaler()),
                ("nystroem", Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=RANDOM_SEED)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
    if name == "random_forest":
        defaults = dict(n_estimators=200, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", RandomForestRegressor(**defaults))])
    if name == "gradient_boosting":
        defaults = dict(n_estimators=100, learning_rate=0.04, max_depth=2, random_state=RANDOM_SEED, loss="huber")
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", GradientBoostingRegressor(**defaults))])
    if name == "xgboost":
        defaults = dict(
            n_estimators=150, learning_rate=0.04, max_depth=3, subsample=0.85,
            colsample_bytree=0.8, objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=2,
        )
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", XGBRegressor(**defaults))])
    if name == "lightgbm":
        defaults = dict(
            n_estimators=150, learning_rate=0.04, num_leaves=15, max_depth=4,
            min_child_samples=15, random_state=RANDOM_SEED, n_jobs=2, verbosity=-1,
        )
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", LGBMRegressor(**defaults))])
    if name == "p1_ridge":
        defaults = dict(alpha=10.0)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("scaler", StandardScaler()), ("model", Ridge(**defaults))])
    if name == "scratch_figs":
        defaults = dict(max_rules=12, max_trees=5, random_state=RANDOM_SEED, backfit=True)
        defaults.update(params)
        return Pipeline([("imputer", _imputer()), ("model", ScratchFIGSRegressor(**defaults))])
    raise KeyError(name)


def ordered_probabilities(estimator, X, n_classes: int = 3) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = np.asarray(estimator.classes_ if hasattr(estimator, "classes_") else estimator[-1].classes_)
    aligned = np.zeros((len(probabilities), n_classes), dtype=float)
    for source, class_index in enumerate(classes):
        aligned[:, int(class_index)] = probabilities[:, source]
    return aligned / aligned.sum(axis=1, keepdims=True)


@dataclass
class ResourceMeasurement:
    elapsed_seconds: float
    peak_rss_mb: float


def measured_fit(estimator, X, y) -> ResourceMeasurement:
    """Fit while sampling process RSS; returns wall time and observed peak."""

    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def sample() -> None:
        while not stop.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        estimator.fit(X, y)
    finally:
        elapsed = time.perf_counter() - started
        peak[0] = max(peak[0], process.memory_info().rss)
        stop.set()
        monitor.join(timeout=1.0)
    return ResourceMeasurement(elapsed, peak[0] / (1024**2))

