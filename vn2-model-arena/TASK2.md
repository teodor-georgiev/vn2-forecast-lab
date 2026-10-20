# Round 2 — the ordering game

Same 599 anonymised series. Now you **order stock**, and you are scored on money, not
on forecast error.

## Data (read-only)

- `..\data\sales.csv`, `..\data\availability.csv`, `..\data\product_attributes.csv`
  — the same 157 weeks of history as round 1.
- `..\data\initial_state.csv` — starting position per series:
  `series_id, on_hand, arrival_w1, arrival_w2` (units already on hand, arriving in
  week 1, arriving in week 2).

## The scoring tool (read-only, provided)

`..\tools\inventory_sim.py` — this is the **exact scoring engine**. Read it. Rules:

- 8 weekly periods. You may order in weeks **1–6**; weeks 7–8 have no orders.
- An order placed in week *w* arrives in week *w+2*.
- Week flow: `supply = on_hand + arrival_next`; `served = min(supply, demand)`;
  `end_inventory = max(0, supply - demand)`; `missed = max(0, demand - supply)`.
- Week cost = **1.00 × missed units + 0.20 × end_inventory units**.
- Your **score is the total cost over weeks 3–8**. Weeks 1–2 are a settling period:
  they still cost money, but they are not part of the score.

You may use `simulate()` and `order_plan_from_policy()` freely to **backtest any
policy of your own design on historical windows** (pick your own cutoffs inside the
history and build your own trailing starting states). Use it as much as you like.

## Deliverable

Exactly one file, in your run directory:

```
orders.csv
header: series_id,week,quantity
```

- 599 series × 6 weeks = **3594 rows**, one per (series_id, week).
- `week` in 1..6, `quantity` numeric and >= 0 (it is rounded to the nearest whole unit).
- Long format, comma separated, exactly those three columns.

You are deciding all six weeks **up front**: you do not get feedback during the game,
and your plan is replayed against demand you cannot see. So a good plan needs a demand
forecast and a sensible response to it — the simulator's reference rule is a starting
point, not a ceiling.

## Rules

- Work only inside your own run directory. Read only the arena `data` and `tools`
  folders.
- No internet. Python 3.12 with pandas, numpy, scikit-learn, lightgbm, xgboost,
  catboost and statsmodels is available.
- You have a **25-minute working budget**.
- Do not attempt to locate or read any scoring data, answer files or private
  directories. Grading uses `orders.csv` only.
