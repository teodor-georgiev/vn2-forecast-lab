"""w02: full-size replay timing + sanity check with the arena's initial state.

The hidden panel is never touched: the demand used here is only a stand-in
(the last 8 known weeks of sales.csv) used to time the replay.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARENA = HERE.parents[1]
t_import = time.perf_counter()
spec = importlib.util.spec_from_file_location("agent_policy", HERE / "policy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
t_import = time.perf_counter() - t_import

init = pd.read_csv(ARENA / "data" / "initial_state.csv")
series = sorted(init["series_id"].tolist())
state0 = (
    init.set_index("series_id")
    .loc[series][["on_hand", "arrival_w1", "arrival_w2"]]
    .to_numpy(float)
)

sales = pd.read_csv(ARENA / "data" / "sales.csv", parse_dates=["week"])
sp = sales.pivot(index="series_id", columns="week", values="y").sort_index(axis=1)
stand_in = sp.reindex(index=series).to_numpy(float)[:, -8:].T  # (8, n) stand-in demand

n = len(series)
state = state0.copy()
revealed = []
orders = []
t0 = time.perf_counter()
for w in range(8):
    if w < 6:
        rev = (
            np.array(revealed, dtype=float).reshape(w, n)
            if revealed
            else np.empty((0, n))
        )
        q = np.asarray(mod.decide(w, state.copy(), rev), dtype=float)
        assert q.shape == (n,) and np.isfinite(q).all() and (q >= 0).all()
        orders.append(np.rint(q))
    else:
        orders.append(np.zeros(n))
    on_hand, arr1, arr2 = state.T.copy()
    supply = on_hand + arr1
    end = np.maximum(0.0, supply - stand_in[w])
    missed = np.maximum(0.0, stand_in[w] - supply)
    state = np.column_stack((end, arr2, orders[-1]))
    revealed.append(stand_in[w])
elapsed = time.perf_counter() - t0

score = 0.0
state = state0.copy()
for w in range(8):
    on_hand, arr1, arr2 = state.T.copy()
    supply = on_hand + arr1
    end = np.maximum(0.0, supply - stand_in[w])
    missed = np.maximum(0.0, stand_in[w] - supply)
    if w >= 2:
        score += 0.20 * end.sum() + 1.00 * missed.sum()
    state = np.column_stack((end, arr2, orders[w]))

print(f"import policy.py      : {t_import:6.2f}s")
print(f"8-week replay (n={n}): {elapsed:6.2f}s   (limit 30s)")
print(f"stand-in score        : {score:,.0f}")
print(f"units ordered         : {sum(o.sum() for o in orders):,.0f}")
print(f"order totals per week : {[int(o.sum()) for o in orders]}")
