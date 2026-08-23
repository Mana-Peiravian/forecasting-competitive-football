# Model 3 — in-play prediction

Model 3 combines a frozen kickoff prior with match evidence observed up to snapshot time `t`:

```text
frozen pre-match features + events observed at or before t
```

It predicts both the final A/D/H distribution and the clipped final signed margin at 19 snapshots: 0, 5, …, 90 minutes.

## Strict time boundary

No event after `t` may contribute:

```text
event_seconds_exact <= snapshot_seconds
```

Fractional timestamp seconds are retained. Recent five- and ten-minute windows are `(t − window, t]`. The public API serves the already validated frozen grid, so `second` must be zero and the minute must be divisible by five.

## Live state

The production snapshot contains:

- current home and away score and score difference;
- cumulative, recent-five, and recent-ten counts for shots, xG, passes, completed passes, final-third entries, pressures, recoveries, interceptions, fouls, cards, and carries;
- home-away differences for each timing window;
- current possession indicator and match-phase features;
- all 391 frozen pre-match predictors.

The implementation retains red-card counts but does not publish a separate named “man advantage” feature; documentation follows the actual schema.

## Public model pair

- **Outcome:** raw gradient boosting, final overall RPS 0.116. Its raw output is the headline because phase-Platt improves ECE but worsens final RPS.
- **Margin:** XGBoost, final MAE 1.276, essentially tied with gradient boosting.

The in-play response returns raw probabilities in `probabilities` and the phase-Platt alternative in `calibrated_probabilities`, with `calibrated: false` to identify which one is headline.

## Evolution through the match

At minute 0, raw gradient-boosting RPS is 0.136, slightly worse than the frozen raw market at 0.131. It reaches 0.104 at halftime, 0.087 at minute 75, and 0.082 at minute 90. Event state becomes valuable only after enough evidence accumulates; a strong pre-match prior remains important early.

![Final Model 3 RPS by minute](../assets/images/results/model3_rps_by_minute.png)

## Timeline API

`GET /api/v1/matches/{match_id}/timeline` vectorizes all 19 snapshots and returns score, raw/calibrated probabilities, and expected final margin at every point. It is a historical replay endpoint over known StatsBomb matches, not a live data feed.
