# Static held-out match replay

!!! info "Static demonstration using saved held-out predictions"

    This page replays stored predictions from the once-opened 34-match final cohort. It does not call FastAPI, consume a live feed, or generate new inference in the browser. Use the [API quickstart](api/quickstart.md) for real local frozen-artifact inference.

Choose a match and snapshot minute. The probability chart uses the raw gradient-boosting Model 3 headline; the margin uses the frozen XGBoost Model 3 regressor. Every point was built with `event_seconds_exact <= snapshot_seconds`.

<div id="demo-app" data-url="../assets/data/demo-matches.json">
  <div class="demo-controls">
    <label>Held-out match
      <select id="demo-match" aria-label="Held-out match"></select>
    </label>
    <label>Snapshot: <span id="demo-minute-label">0 min</span>
      <input id="demo-minute" type="range" min="0" max="90" step="5" value="0">
    </label>
  </div>
  <div id="demo-scoreline" class="demo-scoreline">Loading…</div>
  <div class="probability-row">
    <span>Home win</span>
    <div class="probability-track"><div class="probability-fill home" data-bar="home_win"></div></div>
    <strong data-probability="home_win">—</strong>
  </div>
  <div class="probability-row">
    <span>Draw</span>
    <div class="probability-track"><div class="probability-fill draw" data-bar="draw"></div></div>
    <strong data-probability="draw">—</strong>
  </div>
  <div class="probability-row">
    <span>Away win</span>
    <div class="probability-track"><div class="probability-fill away" data-bar="away_win"></div></div>
    <strong data-probability="away_win">—</strong>
  </div>
  <p>Expected final signed margin: <strong id="demo-margin">—</strong></p>
  <div class="chart-legend">
    <span style="color:#00796b">● Home win</span>
    <span style="color:#ffb300">● Draw</span>
    <span style="color:#5c6bc0">● Away win</span>
  </div>
  <div id="demo-chart" class="timeline-chart"></div>
</div>

<script src="../javascripts/demo.js"></script>

## Interpretation

The three probabilities describe the fitted model’s historical estimate of the **final** result at the selected match time. They are not the probability of the next goal. Expected margin is home goals minus away goals at full time, clipped to `[-5, 5]` during modeling.

Late-match probabilities generally become sharper as score and event evidence accumulates. Individual paths can remain wrong or overconfident; the [explainability timeline](models/explainability.md#full-match-timeline) preserves one such failure.

## Reproducing the data

```bash
python scripts/build_docs_data.py
```

The script reads existing final predictions and target-free snapshot scores, writes a small Pages-safe JSON file, and never trains a model.
