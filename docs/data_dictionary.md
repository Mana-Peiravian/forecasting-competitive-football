# Data dictionary and provenance

All normalized tables live under `data/normalized/`; hashes and row counts are
in `data_manifest.json`.

| Table | Grain | Purpose |
|---|---|---|
| `competitions` | competition-season | StatsBomb competition metadata and 360 availability |
| `matches` | match | Stable fixture/team/stadium/referee metadata; unstable serialized manager IDs are deliberately omitted |
| `teams` | team | Stable StatsBomb team ID/name pairs |
| `players` | player | Player ID/name pairs observed in events |
| `lineups` | match-team-player | Jersey, starter flag, country, positions and cards |
| `events` | event | 1,420,167 flat events, including exact fractional match seconds and selected pass/shot/defensive fields |
| `three_sixty` | event frame | Empty by design: selected seasons have no StatsBomb 360 files |
| `three_sixty_availability` | competition-season | Explicit evidence of the above non-availability |
| `odds` | match | B365 decimal odds, raw implied probabilities, overround, and de-vigged H/D/A probabilities |
| `integrated_matches` | match | Strict date/home/away alias join between StatsBomb and Football-Data |
| `odds_join_diagnostics` | match | Every join key and `_merge` result; coverage is 414/414 |
| `team_aliases` | alias | Explicit source-name mapping |
| `development_targets` | development match | H/D/A and clipped goal margin for the 380-match development season |
| `final_targets_sealed` | final match | The 34 chronologically later outcomes, opened exactly once after protocol lock |

StatsBomb coordinates use a team-in-possession attacking frame. Both home and
away shots start near x=104 and passes have positive mean x displacement, so no
home/away flipping is applied. Event inclusion at time t uses
`event_seconds_exact <= t`; recent windows use `(t-w, t]`.

The odds join uses only match date and explicit home/away aliases. Scores and
match statistics are neither keys nor fuzzy-match inputs. Source URLs are the
StatsBomb open-data repository and Football-Data SP1 season CSVs.

