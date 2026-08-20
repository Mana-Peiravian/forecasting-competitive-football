"""Build normalized 2015/16 development and 2016/17 sealed-final data."""

from ml_project.config import (
    DEVELOPMENT_COMPETITION_ID,
    DEVELOPMENT_SEASON,
    DEVELOPMENT_SEASON_ID,
    FINAL_COMPETITION_ID,
    FINAL_SEASON,
    FINAL_SEASON_ID,
)
from ml_project.data import build_relational_tables


if __name__ == "__main__":
    artifacts = build_relational_tables(
        [
            (DEVELOPMENT_COMPETITION_ID, DEVELOPMENT_SEASON_ID, DEVELOPMENT_SEASON),
            (FINAL_COMPETITION_ID, FINAL_SEASON_ID, FINAL_SEASON),
        ]
    )
    print(f"Built {len(artifacts)} normalized artifacts.")

