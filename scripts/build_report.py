"""Generate result tables and compile the final LaTeX report."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
GENERATED = REPORT / "generated"


def escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    path: Path,
    minimize: set[str] = frozenset(),
    maximize: set[str] = frozenset(),
) -> None:
    best = {}
    for column in minimize:
        best[column] = frame[column].min()
    for column in maximize:
        best[column] = frame[column].max()
    alignment = "l" + "r" * (len(columns) - 1)
    rows = [r"\begin{table}[H]", r"\centering\scriptsize", r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{alignment}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                text = "--" if not np.isfinite(value) else f"{value:.3f}"
                if column in best and np.isfinite(value) and np.isclose(value, best[column]):
                    text = rf"\textbf{{{text}}}"
            elif isinstance(value, (int, np.integer)):
                text = str(value)
            else:
                text = escape(value)
            cells.append(text)
        rows.append(" & ".join(cells) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    final = ROOT / "outputs" / "final_evaluation"
    class_frame = pd.read_csv(final / "prematch_classification_metrics.csv").sort_values("rps")
    table(
        class_frame,
        ["model", "method", "rps", "log_loss", "brier", "ece", "accuracy", "rps_ci_lower", "rps_ci_upper"],
        ["Model", "Cal.", "RPS", "Log loss", "Brier", "ECE", "Acc.", "CI low", "CI high"],
        GENERATED / "prematch_classification.tex",
        minimize={"rps", "log_loss", "brier", "ece"}, maximize={"accuracy"},
    )
    reg_frame = pd.read_csv(final / "prematch_regression_metrics.csv").sort_values("mae")
    table(
        reg_frame,
        ["model", "mae", "rmse", "correlation", "converted_rps", "converted_log_loss", "mae_ci_lower", "mae_ci_upper"],
        ["Model", "MAE", "RMSE", "$r$", "H/D/A RPS", "H/D/A LL", "CI low", "CI high"],
        GENERATED / "prematch_regression.tex",
        minimize={"mae", "rmse", "converted_rps", "converted_log_loss"}, maximize={"correlation"},
    )
    snapshot_class = pd.read_csv(final / "snapshot_classification_metrics.csv").sort_values("rps")
    table(
        snapshot_class,
        ["model", "method", "rps", "log_loss", "brier", "ece", "accuracy", "rps_ci_lower", "rps_ci_upper"],
        ["Model", "Cal.", "RPS", "Log loss", "Brier", "ECE", "Acc.", "CI low", "CI high"],
        GENERATED / "snapshot_classification.tex",
        minimize={"rps", "log_loss", "brier", "ece"}, maximize={"accuracy"},
    )
    snapshot_reg = pd.read_csv(final / "snapshot_regression_metrics.csv").sort_values("mae")
    table(
        snapshot_reg,
        ["model", "mae", "rmse", "correlation", "converted_rps", "converted_log_loss", "converted_accuracy"],
        ["Model", "MAE", "RMSE", "$r$", "H/D/A RPS", "H/D/A LL", "H/D/A Acc."],
        GENERATED / "snapshot_regression.tex",
        minimize={"mae", "rmse", "converted_rps", "converted_log_loss"}, maximize={"correlation", "converted_accuracy"},
    )
    p1 = pd.read_csv(ROOT / "outputs" / "tables" / "p1" / "p1_team_comparison.csv").head(10)
    table(
        p1,
        ["team", "points", "mean_supra_centrality", "mean_leakage", "mean_recovery", "mean_switching", "centrality_rank"],
        ["Team", "Pts", "Supra cent.", "Leakage", "Recovery", "Switching", "Cent. rank"],
        GENERATED / "p1_teams.tex",
    )
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=REPORT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    source = REPORT / "report.pdf"
    target = REPORT / "final_report.pdf"
    target.write_bytes(source.read_bytes())
    print(f"Compiled {target}")


if __name__ == "__main__":
    main()

