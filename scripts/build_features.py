"""Build the frozen prematch feature table and temporal split manifest."""

import pandas as pd

from ml_project.config import NORMALIZED_DIR, OUTPUT_DIR
from ml_project.features import build_prematch_features, write_feature_artifacts


if __name__ == "__main__":
    events = pd.read_parquet(NORMALIZED_DIR / "events.parquet")
    matches = pd.read_parquet(NORMALIZED_DIR / "integrated_matches.parquet")
    p1 = pd.read_parquet(OUTPUT_DIR / "p1" / "team_match_metrics.parquet")
    features, groups = build_prematch_features(events, matches, p1)
    write_feature_artifacts(features, groups, OUTPUT_DIR / "features")
    print(f"Built {len(features)} prematch rows and {len(groups)} feature groups.")

