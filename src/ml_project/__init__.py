"""Reproducible football forecasting and network-analysis package."""

from .config import CLASS_ORDER, RANDOM_SEED
from .figs import ScratchFIGSClassifier, ScratchFIGSRegressor

__all__ = [
    "CLASS_ORDER",
    "RANDOM_SEED",
    "ScratchFIGSClassifier",
    "ScratchFIGSRegressor",
]

__version__ = "1.0.0"

