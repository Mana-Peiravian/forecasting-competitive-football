# Data pipeline

## 1. Acquisition and normalization

Official StatsBomb JSON is flattened into stable event columns with event identity, period, timestamp, team, possession, locations, passes, shots, defensive actions, cards, and exact fractional match seconds. Football-Data odds are retained at source and integrated through explicit aliases.

Every normalized artifact receives row-count and SHA-256 provenance in `data/normalized/data_manifest.json`.

## 2. P1 multilayer construction

Each team’s pitch is divided into 20 zones. Directed completed passes form within-team edges; boundary-aware possession changes form cross-team edges. Two 20-node layers become a 40 × 40 supra-adjacency matrix. Right-eigenvector centrality, leakage, recovery, switching, and flow totals are exported at zone and team-match grain.

## 3. Pre-match histories

Event and P1 team-match metrics are sorted by team, date, and match ID. The pipeline applies `shift(1)` **before** rolling or expanding aggregation. Each measure then receives 3-, 5-, and 10-match rolling means plus an expanding mean. Home, away, and difference views are retained.

The final groups contain:

| Group | Predictors |
|---|---:|
| Market | 3 |
| Conventional | 226 |
| Network | 162 |
| Combined | 391 |

The feature file contains no score, outcome, or goal-margin target.

## 4. Exact in-play snapshots

Nineteen snapshots are created per match at minutes 0, 5, …, 90. For cutoff `t`, only events satisfying:

```text
event_seconds_exact <= snapshot_seconds
```

are eligible. Recent windows are `(t − window, t]`, so the old boundary is excluded and the current boundary is included.

The live state includes score, shots, xG, passes, completed passes, final-third entries, pressures, recoveries, interceptions, fouls, cards, carries, current possession, and cumulative/recent home-away differences. All 391 frozen pre-match predictors are copied unchanged, producing 507 Model 3 predictors.

## 5. Modeling, calibration, and evaluation

Candidate families are selected on temporal development roles. Base models refit on train + validation + pilot; calibration remains disjoint. The final 34-match cohort is never used for training, tuning, calibration, or feature definition. Bootstrap intervals sample matches, keeping all snapshots from a sampled match together.

## 6. Public inference

The API uses the ordered manifests and already materialized rows. This is important: it does not rebuild a looser “API feature set,” retrain, or load raw events for every call. New fixtures would require running the upstream pipeline with legitimate history and event coverage before they become queryable.
