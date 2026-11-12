# VN2 Inventory Planning Challenge — solution from scratch

A complete forecasting + inventory-ordering solution for the VN2 challenge,
built from the raw CSVs up. No code is reused from any earlier attempt: the
stock-flow engine, the causal forecast layer, the policies, the cross-validation
harness and the independent verifier were all derived here and checked against
the published competition numbers.

## Headline result

**Validity: retrospective competition replay.** The game weeks 1–8 labels were
published before this project existed, so every number produced from them is a
replay. None of it is blind and none of it is an official score.

| metric | value |
| --- | ---: |
| Organizer benchmark, weeks 3–8 | EUR 4,334.00 |
| **This solution, weeks 3–8** | **EUR 3,899.00** |
| &nbsp;&nbsp;holding | EUR 2,510.00 |
| &nbsp;&nbsp;shortage | EUR 1,389.00 |
| Fill rate, scored weeks | 86.79% |

For context only, and not comparable to the above: the published winning entry
scored EUR 3,763.20.

## The stock-flow equations

Game weeks run 1–8. Orders are placed at the end of weeks 1–6. An order placed
at the end of week *X* is received at the start of week *X+3* and is sellable
that week, so round *d* (1-based) arrives in game week *d+2*.

Writing `arr[t]` for arrivals into 0-indexed game week `t`:

```
arr[0] = In Transit W+1        # already in flight when the game opens
arr[1] = In Transit W+2
arr[t] = orders[t - 2]         # for t = 2..7; orders has shape (6, 599)
```

Per game week `t`:

```
available  = on_hand[t-1] + arr[t]
lost[t]    = max(0, demand[t] - available)
on_hand[t] = max(0, available - demand[t])
holding[t]  = 0.20 * on_hand[t]          # EUR per unit of ENDING inventory
shortage[t] = 1.00 * lost[t]             # EUR per LOST sale; no backorders
```

Only game weeks 3–8 are scored.

**Consequence worth stating plainly:** game weeks 1 and 2 costs depend *only*
on the opening state and on demand, never on any order, because the earliest a
player order can arrive is week 3. That gives a free invariant, and it is
asserted in the test suite: any two policies must produce identical week-1 and
week-2 costs.

The opening state is `End Inventory` (on-hand) plus `In Transit W+2` (pipeline).
`Start Inventory` and `In Transit W+1` are all-zero columns in the supplied
file and are *not* the game's starting position.

## Reproduction gates

The engine is trusted only because it reproduces the published competition
numbers exactly, before any modelling work is counted:

| gate | expected | measured |
| --- | ---: | ---: |
| Week-1 cost, any policy | 380.60 | 380.60 |
| Week-2 cost, any policy | 533.20 | 533.20 |
| Organizer benchmark, weeks 3–8 | 4,334.00 | 4,334.00 |
| Organizer holding, all 8 weeks | 3,618.80 | 3,618.80 |
| Organizer shortage, all 8 weeks | 1,629.00 | 1,629.00 |
| Organizer weekly profile | 6 values | all match |

The weekly profile is the strongest of these: six independent numbers pin the
policy down far tighter than a single total does, and a wrong holding/shortage
trade-off could match the total by coincidence while missing every week.

`verify.py` is an independent second stock-flow implementation, deliberately
written in a different style from `simulator.py`, plus a closed-form oracle for
the no-stockout regime where the recursion collapses to a cumulative sum.
`tests/test_simulator.py` reconciles the two to under 1e-9 on randomized and
degenerate inputs.

## What the organizer benchmark actually is

The published description — "a 13-week seasonal moving average combined with a
forecast-driven policy with 4 weeks coverage" — is not enough on its own to
reproduce EUR 4,334.00. Three details decide whether the number comes out, and
all three are easy to miss:

1. **Stockout censoring.** Weeks flagged out of stock must be treated as
   *missing*, not as zero demand. Averaging them as zeros understates the
   demand level badly.
2. **Multiplicative week-of-year seasonality.** The 13-week mean is taken on
   *deseasonalised* demand and then re-seasonalised onto each of the next four
   target weeks. It is not a plain trailing mean.
3. **Order up to the full position.** The target is four weeks of seasonal
   demand minus on-hand minus both committed pipeline slots, with no lead-time
   demand subtraction.

A naive reconstruction of the description lands near EUR 4,295 — close enough
to look plausible and wrong enough to be useless. Matching the holding and
shortage components *separately* is what exposes it.

## Solution architecture

| module | role |
| --- | --- |
| `config.py` | every constant, anchor and gate threshold in one place |
| `data.py` | loading, hash verification, contract assertion |
| `simulator.py` | the stock flow above |
| `organizer.py` | the official benchmark policy, reproduced exactly |
| `forecast.py` | causal features and two forecast sources |
| `policy.py` | order-up-to policies parameterised by coverage and buffer |
| `cv.py` | chronological, non-overlapping cross-validation |
| `verify.py` | independent second simulator and cross-check |
| `submit.py` | submission construction and format validation |
| `experiment.py` | the frozen end-to-end runner |

### Cross-validation

Eight disjoint 8-week windows, each evaluated from three synthetic opening
states under three demand levels (+0/+10/+20%), scoring only weeks 3–8 of each
window. Selection uses a robust statistic — `mean + 0.45·std + 0.15·(worst − mean)`
— so a configuration that wins on average but fails badly in one season loses.

Forecast models are fit **once per origin**, then cached, and the policy sweep
reuses that cache. Tuning a policy parameter therefore cannot silently refit a
model and contaminate the comparison.

Fold costs are measured on a deliberately harsher regime than the replay
(synthetic openings, stressed demand), so they select the configuration but do
not *estimate* the replay score. They are not comparable in level to the
€3,899.00 figure.

### Anti-leakage

Every forecast function takes an explicit cutoff and may read only columns
`[0:t]`. `tests/test_causality.py` enforces this by perturbing all columns at
and after the cutoff and requiring byte-identical output.

Two details of that test are load-bearing and were both found the hard way:

- The mutation happens **in place on a single allocation**. Two arrays holding
  equal values but living in different buffers disagree in the last few ULPs,
  because numpy's SIMD reductions choose their unrolling order from the base
  pointer's alignment. Comparing two separately-allocated panels reports a
  spurious ~1e-14 "leak" for a perfectly causal function.
- The mutation cutoff must **equal the decision time** the function is called
  with. Columns between the mutation cutoff and the decision time are
  legitimately in the past; changing them *should* change the output.

A harness self-check perturbs the past and requires the output *to* change, so
a passing assertion evidences causality rather than a no-op function.

## The LightGBM branch does not earn its place

The forecast layer offers a blend between the seasonal reference and a
three-horizon LightGBM model. Sweeping the blend weight over the CV folds:

| seasonal weight | best robust CV cost | vs pure seasonal |
| ---: | ---: | ---: |
| 1.0 (pure seasonal) | 6,209.19 | — |
| 0.7 | 6,234.18 | +24.98 |
| 0.4 | 6,670.07 | +460.87 |

Cost rises monotonically as the gradient-boosted head is given more weight, and
the optimum sits at the edge of the grid. The defensible reading is that on
this panel — 599 intermittent series, heavy stockout censoring — the boosted
head adds noise rather than signal, and the frozen configuration discards it.
That is the result the CV selected; it is reported here rather than hidden
behind the more flattering architecture.

## Running it

Dependencies are already present: Python 3.12, numpy, pandas, LightGBM, pytest.

```
$env:PYTHONPATH = "<project root>"
python -m pytest tests/ -q            # 15 tests
python scripts\check_anchors.py       # lead-time alignment gates
python scripts\verify_organizer_baseline.py
python -m vn2_from_scratch.experiment # CV sweep -> frozen config -> replay
```

The experiment writes `artifacts/frozen_config.json` (every evaluated
candidate), `artifacts/retrospective_replay.json`, `reports/RESULTS.md`, and one
valid submission file per round under `submissions/`.

## Data provenance

Input CSVs are SHA-256 verified against `data/SHA256SUMS.txt` on every load.

Citation: Vandeput, Nicolas. "VN2 Inventory Planning Challenge."
DataSource.ai. https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge/description
