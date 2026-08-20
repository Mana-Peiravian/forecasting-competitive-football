"""Build exact five-minute snapshots with frozen prematch predictors."""

import json

import pandas as pd

from ml_project.config import NORMALIZED_DIR, OUTPUT_DIR
from ml_project.snapshots import build_snapshot_dataset, write_snapshot_artifacts


if __name__ == "__main__":
    events = pd.read_parquet(NORMALIZED_DIR / "events.parquet")
    matches = pd.read_parquet(NORMALIZED_DIR / "integrated_matches.parquet")
    prematch = pd.read_parquet(OUTPUT_DIR / "features" / "prematch_features.parquet")
    feature_manifest = json.loads((OUTPUT_DIR / "features" / "feature_manifest.json").read_text())
    frozen = feature_manifest["groups"]["combined"]
    snapshots = build_snapshot_dataset(events, matches, prematch, frozen)
    write_snapshot_artifacts(snapshots, frozen, OUTPUT_DIR / "snapshots")
    print(f"Built {len(snapshots)} exact-boundary snapshots from {snapshots.match_id.nunique()} matches.")

