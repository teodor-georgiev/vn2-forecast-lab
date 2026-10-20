# VN2 model arena

Eight free models, one identical task, one frozen scorer. Built from the full VN2
inventory-planning dataset (599 store-product series, 157 weeks of history, 8-week
decision window), anonymised so no contestant can look up the published answers.

## Why anonymised

The original VN2 files sit on this machine and the competition is public. The arena
panel was transformed before any model saw it:

- `(Store, Product)` → salted SHA-256 series id (`s...`), mapping kept private;
- demand multiplied by a per-series random factor in [0.60, 1.40];
- real dates replaced with synthetic weekly Mondays starting 2019-01-07.

The transform preserves each series' shape and the 1.00/0.20 cost ratio, and breaks
any join back to the raw files by id or by date. The mapping and answer panel live
outside the arena at `%TEMP%\opencode\vn2-arena-private\`.

## Rounds

**Round 1 — forecast.** 599 series × 8 horizons, scored on WAPE (plus MAE, PBIAS).
See `TASK.md`. Submissions: `runs\<model>\predictions.csv`.

**Round 2 — ordering game.** Six weekly order decisions per series, arrivals two weeks
later, €1.00 per missed unit, €0.20 per held unit-week, scored over weeks 3-8. See
`TASK2.md`. Contestants get the proven simulator at `tools\inventory_sim.py`.
Submissions: `runs\<model>\orders.csv`.

**Round 3 — the adaptive game.** Same costs and window, but it replicates how the real
VN2 competition was played (datasource.ai VN2 description): one order per week for six
rounds, two-week lead time, and — the key difference — **weekly feedback**: each
decision is made with the actual demand of the weeks already played. Workers submit a
`policy.py` defining `decide(week_index, state, revealed)`; the manager replays the
module against the hidden demand with `scripts\score_adaptive.py`, which also
combines the policies into consensus decision rules (pre-registered median of all).
Work directories: `runs3\w01..w20`. Reference bars in arena units: decent adaptive
rule ~3,983; historical competition replay equivalent to ~3,481.

Both rounds: same data, same prompt, one attempt, isolated directories, self-reported
metrics ignored.

## Frozen graders

| Script | Purpose |
|---|---|
| `scripts\prepare_arena.py` | builds the anonymised panel from the raw VN2 files |
| `scripts\score_submission.py` | round 1 scorer (WAPE, MAE, RMSE, PBIAS, per-horizon) |
| `scripts\score_all.py` | round 1 leaderboard → `results\leaderboard.md` |
| `scripts\make_baselines.py` | naive forecast baselines |
| `scripts\prepare_round2.py` | starting inventory + reference order plans |
| `scripts\score_orders.py` | round 2 scorer (simulator replay) |
| `scripts\score_adaptive.py` | round 3 adaptive replay scorer + consensus combos |
| `tools\helper_backtest.py` | per-worker backtest helper on known historical windows |
| `tools\selfcheck.py` | policy interface check (synthetic game, no truth) |
| `scripts\score_all_round2.py` | round 2 leaderboard → `results\round2_leaderboard.md` |
| `scripts\ensemble_round1.py` | all round-1 forecast combinations → `results\ensemble_results.csv` |
| `scripts\ensemble_round2.py` | all round-2 order-plan combinations → `results\ensemble2_results.csv` |
| `scripts\parity_check_simulator.py` | proves the arena simulator matches the proven research one |

Sanity checks built in: a zero forecast must score exactly WAPE 1.0; the extracted
simulator must reproduce the research simulator's weekly costs and order totals to 1e-9.

## Ensembles and combined policies

Each round's submissions were also combined per series (mean, median, trimmed mean) and
scored through the same frozen graders. The **all-member median is the pre-registered
rule** — it can be stated before any score is seen. Subset searches are reported for
description only: choosing the best of N combinations on the holdout is selection, not
skill, and the spread between combinations shows how much of the gain is noise.

Round 1 (WAPE, lower better): best single 0.5378 (DeepSeek V4.1); median of all eight
0.5344; mean 0.5349; best 3-model subset 0.5311; naive 13-week median baseline 0.5238.
The combine improves on every individual but still loses to the naive baseline.

Round 2 (EUR, lower better): singles span 4,430.07 to 5,327.92; median of all eight
**4,210.22**; trimmed 4,231.18; mean 4,289.10 (no better than the fair reference);
best 3-model subset 4,157.85 (hindsight-selected; only 3 of 130 combinations
break 4,200). Fair static reference 4,285.51; true-feedback headroom 3,982.63.

Weeks 1-2 are excluded from the round-2 score (the `setup` term) and are identical for
every plan to the cent (EUR 940.01) because orders placed in week w arrive in week w+2:
no decision can affect weeks 1-2. Including them would add a constant to every entry.

## Provenance

Raw source: `C:\Users\Toni\DemandAhead-VN2Game\raw` (`Week 0` history, `Week 8` answers,
`Week 0 - In Stock`, `Week 0 - Master`, `Week 0 - Initial State`). Simulation logic
extracted from `C:\Users\Toni\vn2-direct-lgbm-20261008\run_vn2_lgbm.py`. Raw file
SHA-256 digests are recorded in the private `meta.json`.
