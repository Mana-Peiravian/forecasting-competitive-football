# Data preparation

Normalized and feature artifacts are already delivered. Ordinary validation does not require network access or rebuilding them.

## Data stage

```bash
football-ml-reproduce --stage data
```

This orchestrates normalized source acquisition and P1 network reconstruction. It is the slowest stage and requires network access for absent caches. StatsBomb attribution and Football-Data source URLs are retained in manifests.

## Feature stage

```bash
football-ml-reproduce --stage features
```

This rebuilds shifted pre-match histories and exact in-play snapshots from normalized events and P1 metrics. Expected identities include:

- 414 pre-match rows;
- 391 combined pre-match predictors;
- 7,866 snapshot rows;
- 19 snapshots per match;
- 507 combined snapshot predictors.

## Validation invariants

After rebuilding:

- match and event IDs must remain unique at their declared grains;
- odds coverage must remain 414/414 with exact joins;
- source times must precede kickoff;
- `feature_source_max_seconds <= snapshot_seconds` for every snapshot;
- each match must have exactly one split role;
- final target files must not appear in target-free feature outputs.

## Source-coverage warning

StatsBomb Open Data exposes only 34 selected 2016/17 La Liga matches, largely involving Barcelona. Rebuilding does not create the missing league fixtures. Later-season rolling histories remain incomplete relative to the real competition schedule.

## Provenance

`data/normalized/data_manifest.json`, feature/snapshot manifests, split manifests, and the pre-final lock contain row counts, schemas, source context, and SHA-256 identities. Do not silently replace source files while retaining old manifests.
