<div class="ffc-hero" markdown>

<div class="ffc-kicker">Machine learning · event data · calibrated probabilities</div>

# Forecasting Competitive Football

## Pre-Match & In-Play Prediction from Raw Event Data

<p class="ffc-subtitle">A leakage-safe research and engineering pipeline that turns StatsBomb events, Football-Data.co.uk odds, P1 multilayer network structure, and a from-scratch P2 FIGS implementation into reproducible pre-match and five-minute in-play forecasts.</p>

<div class="ffc-actions">
  <a class="ffc-button primary" href="api/quickstart/">API quickstart</a>
  <a class="ffc-button" href="api/swagger/">Swagger UI</a>
  <a class="ffc-button" href="api/redoc/">ReDoc</a>
  <a class="ffc-button" href="results/">Results</a>
  <a class="ffc-button" href="project/architecture/">Architecture</a>
  <a class="ffc-button" href="https://github.com/Mana-Peiravian/forecasting-competitive-football">GitHub repository</a>
</div>

</div>

<div class="model-grid">
  <div class="model-card">
    <h3>Model 1</h3>
    <strong>Pre-match outcome</strong>
    <p>Platt-calibrated probabilities for home win, draw, and away win from information available at kickoff.</p>
  </div>
  <div class="model-card">
    <h3>Model 2</h3>
    <strong>Signed goal margin</strong>
    <p>Expected final <code>home goals − away goals</code>, clipped to the project’s fixed range of −5 to +5.</p>
  </div>
  <div class="model-card">
    <h3>Model 3</h3>
    <strong>In-play updates</strong>
    <p>Outcome probabilities and expected final margin at 0, 5, …, 90 minutes using no event after the snapshot.</p>
  </div>
</div>

## What makes the project defensible

- **StatsBomb Open Data** supplies 414 match event streams; **Football-Data.co.uk** supplies the B365 market baseline.
- **P1** contributes a 40-node, two-layer football network representation built from a 4 × 5 pitch grid.
- **P2** contributes a tested scratch implementation of Fast Interpretable Greedy-Tree Sums (FIGS).
- Calibration uses disjoint holdout data; evaluation uses Ranked Probability Score, calibration error, and match-level uncertainty—not argmax accuracy alone.
- Historical features are shifted before aggregation, and every in-play feature obeys `event_seconds_exact <= snapshot_seconds`.
- SHAP evidence, reliability analysis, ablations, market comparisons, and compute profiles are retained as machine-readable outputs.

!!! success "Headline result—with the uncertainty attached"

    On the 34 available 2016/17 final matches, Model 1 random-forest/Platt reached RPS **0.120**, versus **0.131** for the raw market. The paired 95% interval crosses zero, so this is **not** evidence that the model beats bookmakers. Raw Model 3 gradient boosting reached overall RPS **0.116** and improved late in matches.

## Static site, separate Python service

This GitHub Pages site hosts documentation, generated OpenAPI reference, scientific plots, and a [stored held-out replay](demo.md). It does **not** execute FastAPI or perform live inference. The repository includes a [separately deployable backend](developer/deployment.md) that loads frozen artifacts once and can be run locally or on a Python-capable host.

Start with the [project overview](project/overview.md), inspect the [final results](results/index.md), or make a real local request through the [API quickstart](api/quickstart.md).
