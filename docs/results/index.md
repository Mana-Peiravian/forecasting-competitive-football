# Final results

The final evaluation contains 34 chronologically later StatsBomb 2016/17 matches and 646 five-minute snapshots. All tables below are generated from the final CSV artifacts by `scripts/build_docs_data.py`; they should not be edited manually.

<div class="metric-grid">
  <div class="metric-card"><span>Model 1 RF + Platt</span><strong>0.120</strong><span>RPS</span></div>
  <div class="metric-card"><span>Raw market</span><strong>0.131</strong><span>RPS</span></div>
  <div class="metric-card"><span>Model 2 RF</span><strong>1.479</strong><span>MAE goals</span></div>
  <div class="metric-card"><span>Model 3 GB raw</span><strong>0.116</strong><span>overall RPS</span></div>
</div>

!!! warning "Historical point estimates, not a deployment guarantee"

    The 34 final matches are Barcelona-centred and confidence intervals are wide. The paired Model 1 improvement over the raw market crosses zero. Results describe one historical open-data backtest; they do not demonstrate profitability or stable future superiority.

--8<-- "docs/results/generated_metrics.md"

## Reading the tables

- Lower is better for RPS, log loss, Brier, ECE, MAE, and RMSE.
- `converted_rps` evaluates probabilities derived from a margin model’s scalar output.
- Model 3 intervals resample whole matches, not individual snapshots.
- `platt` for Model 3 is phase-specific; it can improve ECE while worsening RPS.

![Final Model 1 RPS](../assets/images/results/model1_rps.png)

![Final Model 2 MAE](../assets/images/results/model2_mae.png)

![Final Model 3 RPS by minute](../assets/images/results/model3_rps_by_minute.png)

Machine-readable sources live in `outputs/final_evaluation/`; the permanent `FINAL_TEST_OPENED.json` marker prevents rerunning the one-shot evaluator.
