"""Freeze code, features, splits, and model specifications before final labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml_project.config import OUTPUT_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock_path = OUTPUT_DIR / "model_lock_before_final.json"
    opened = OUTPUT_DIR / "final_evaluation" / "FINAL_TEST_OPENED.json"
    if lock_path.exists() or opened.exists():
        raise RuntimeError("A final lock/open marker already exists; refusing replacement")
    relative_files = [
        "pyproject.toml",
        "environment.yml",
        "src/ml_project/config.py",
        "src/ml_project/metrics.py",
        "src/ml_project/calibration.py",
        "src/ml_project/figs.py",
        "src/ml_project/network.py",
        "src/ml_project/features.py",
        "src/ml_project/snapshots.py",
        "src/ml_project/modeling.py",
        "scripts/run_final_evaluation.py",
        "outputs/features/prematch_features.parquet",
        "outputs/features/feature_manifest.json",
        "outputs/features/split_manifest.csv",
        "outputs/snapshots/snapshot_features.parquet",
        "outputs/snapshots/snapshot_manifest.json",
        "outputs/experiments/prematch/locked_classifier_specs.csv",
        "outputs/experiments/prematch/locked_regressor_specs.csv",
        "outputs/experiments/prematch/development_protocol.json",
        "outputs/experiments/snapshots/locked_snapshot_specs.json",
        "data/normalized/final_targets_sealed.parquet",
    ]
    missing = [relative for relative in relative_files if not (root / relative).exists()]
    if missing:
        raise FileNotFoundError(missing)
    split = pd.read_csv(root / "outputs/features/split_manifest.csv")
    final_rows = split[split.split_role.eq("final_test")]
    if len(final_rows) != 34:
        raise AssertionError(f"Expected 34 sealed final matches, found {len(final_rows)}")
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze every model/feature/calibration choice before opening 2016/17 final labels.",
        "final_test": {
            "season": "2016/2017",
            "available_matches": len(final_rows),
            "match_id_manifest_sha256": sha256(root / "outputs/features/split_manifest.csv"),
            "known_limitation": "StatsBomb open data exposes 34 Barcelona-centered matches, not the full 380-match season.",
        },
        "protocol": {
            "classification_headline": "RPS, class order A-D-H",
            "regression_headline": "MAE with margin clipped to [-5,5]",
            "calibration": "Platt fixed a priori; fit only on calibration role; phase-specific for Model 3",
            "base_refit_roles": ["train", "validation", "pilot_holdout"],
            "calibration_role": "calibration",
            "final_role": "final_test",
            "final_evaluation_runs_allowed": 1,
        },
        "locked_file_hashes": {relative: sha256(root / relative) for relative in relative_files},
        "git_history": "unavailable in supplied directory; no history fabricated",
        "final_targets_opened": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(f"Final protocol locked across {len(relative_files)} files; final targets remain sealed.")

