"""Dev harness: replay policy.decide on historical windows (not a deliverable).

Mirrors helper_backtest but correctly feeds the revealed in-game demand back to
decide each week (the shipped helper never appends to `revealed`). Uses the
censoring-corrected sales history as both revealed demand and sim demand, which
approximates the real game where revealed demand is actual (uncensored) demand.
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
arena = here.parent.parent

from policy import decide, _ensure_loaded

_ensure_loaded()

init = pd.read_csv(arena / "data" / "initial_state.csv")
sales = pd.read_csv(arena / "data" / "sales.csv")
weeks = sorted(sales["week"].astype(str).unique())
raw = (
    sales.pivot(index="series_id", columns="week", values="y")
    .reindex(index=init["series_id"], columns=weeks)
    .fillna(0.0)
    .to_numpy(float)
)
avail = pd.read_csv(arena / "data" / "availability.csv")
av = (
    avail.pivot(index="series_id", columns="week", values="is_available")
    .reindex(index=init["series_id"], columns=weeks)
    .fillna(True)
    .to_numpy(bool)
)
corrected = raw.copy()
for i in range(raw.shape[0]):
    vals = raw[i][av[i]]
    fill = float(vals.mean()) if vals.size else 0.0
    corrected[i] = np.where(av[i], raw[i], fill)

initial = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)


def replay(decide, history, start_week, initial=None, hold=0.20, short=1.00):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    state = initial.copy() if initial is not None else np.zeros((n, 3))
    demand = history[:, start_week : start_week + 8].T
    revealed = []
    weekly = []
    t0 = time.perf_counter()
    for w in range(8):
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(0.0, np.asarray(decide(w, state.copy(), rev), dtype=float))
            )
        else:
            order = np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {
                "week": w + 1,
                "holding": float(hold * end.sum()),
                "shortage": float(short * missed.sum()),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    setup = sum(r["holding"] + r["shortage"] for r in weekly[:2])
    elapsed = time.perf_counter() - t0
    return score, setup, weekly, elapsed


for sw in (100, 130):
    score, setup, weekly, elapsed = replay(decide, corrected, sw, initial=initial)
    wk = " | ".join(
        f"w{r['week']}:o{r['ordered']:5d} h{r['holding']:7.1f} s{r['shortage']:7.1f}"
        for r in weekly
    )
    print(f"cutoff {sw}: score={score:.1f} setup={setup:.1f} elapsed={elapsed:.2f}s")
    print("   ", wk)
