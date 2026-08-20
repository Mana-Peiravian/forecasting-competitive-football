"""Fixed scientific configuration.

These constants were frozen before the 2016/17 final-season outcomes were used
by any model selection or evaluation code.  The 2015/16 tail beginning on
2016-04-02 is deliberately named ``pilot_holdout`` because prior notebooks had
already inspected it repeatedly.
"""

from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
CLASS_ORDER = ("A", "D", "H")
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASS_ORDER)}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
NORMALIZED_DIR = DATA_DIR / "normalized"
FINAL_DIR = DATA_DIR / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models" / "reproducible"

DEVELOPMENT_COMPETITION_ID = 11
DEVELOPMENT_SEASON_ID = 27
DEVELOPMENT_SEASON = "2015/2016"
FINAL_COMPETITION_ID = 11
FINAL_SEASON_ID = 2
FINAL_SEASON = "2016/2017"

# Non-overlapping development roles. The pilot boundary is inherited from the
# opened historical experiment and is not used for tuning or calibration.
TRAIN_END = "2016-01-10"
VALIDATION_START = "2016-01-11"
VALIDATION_END = "2016-02-14"
CALIBRATION_START = "2016-02-15"
CALIBRATION_END = "2016-04-01"
PILOT_START = "2016-04-02"

SNAPSHOT_MINUTES = tuple(range(0, 91, 5))
PITCH_X_MAX = 120.0
PITCH_Y_MAX = 80.0
GRID_X_BINS = 4
GRID_Y_BINS = 5
N_ZONES = GRID_X_BINS * GRID_Y_BINS


def ensure_directories() -> None:
    """Create generated-artifact directories without touching source data."""

    for path in (NORMALIZED_DIR, FINAL_DIR, OUTPUT_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)

