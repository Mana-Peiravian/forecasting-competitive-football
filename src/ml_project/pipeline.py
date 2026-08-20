"""Small command-line orchestrator for reproducible project stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["data", "features", "development", "artifacts", "validate", "all"],
        default="validate",
    )
    args = parser.parse_args()
    stages = {
        "data": ["build_data.py", "build_networks.py"],
        "features": ["build_features.py", "build_snapshots.py"],
        "development": ["run_prematch_development.py", "run_snapshot_development.py"],
        "artifacts": ["make_p1_artifacts.py", "make_model_artifacts.py", "run_explainability.py"],
    }
    if args.stage == "validate":
        subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT, check=True)
        return
    selected = [args.stage] if args.stage != "all" else ["data", "features", "development", "artifacts"]
    for stage in selected:
        for script in stages[stage]:
            run(script)
    marker = ROOT / "outputs" / "final_evaluation" / "FINAL_TEST_OPENED.json"
    if marker.exists():
        print("Final evaluation is preserved and intentionally not rerun.")


if __name__ == "__main__":
    main()

