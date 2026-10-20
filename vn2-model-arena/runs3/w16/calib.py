"""Calibrate: zero-order, static reference, and my policy on cutoffs 100 & 130,
with the real initial state. Print weekly breakdown."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
sys.path.insert(0, str(ARENA / "tools"))
sys.path.insert(0, str(ARENA / "runs3" / "w16"))
import inventory_sim as sim
import tuning as T

init = pd.read_csv(ARENA / "data" / "initial_state.csv")
init = init.sort_values("series_id").reset_index(drop=True)
initial = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
print(
    "initial state: on_hand sum=%.1f arr1 sum=%.1f arr2 sum=%.1f"
    % (initial[:, 0].sum(), initial[:, 1].sum(), initial[:, 2].sum())
)


def weekly_text(res):
    parts = []
    for i, r in enumerate(res["weekly"]):
        parts.append(
            "w%d:h%.0f/s%.0f/o%d" % (i + 1, r["holding"], r["shortage"], r["ordered"])
        )
    return " ".join(parts)


def static_reference(start_week):
    mean13 = T.H[:, start_week - 13 : start_week].mean(axis=1)
    state = initial.copy()
    orders = np.zeros((6, T.N_HIST))
    for w in range(8):
        if w < 6:
            forecast = np.repeat(mean13[None, :], 3, axis=0)
            orders[w] = sim.reference_orders(forecast, state, 1.0, 1.0, 0.75)
        order = orders[w] if w < 6 else np.zeros(T.N_HIST)
        on_hand, arr1, arr2 = state.T
        supply = on_hand + arr1
        demand_w = T.H[:, start_week + w]
        end = np.maximum(0.0, supply - demand_w)
        missed = np.maximum(0.0, demand_w - supply)
        state = np.column_stack((end, arr2, order))
    demand = T.H[:, start_week : start_week + 8].T
    return sim.simulate(demand, initial, orders)


bp = json.loads(Path(ARENA / "runs3" / "w16" / "best_params.json").read_text())
p = bp["params"]
params = tuple(
    p[k]
    for k in ["scale", "buffer", "projection", "inflate", "shrink_k", "safety_mode"]
)
dec = lambda wk, st, rev: T.decide_params(wk, st, rev, *params)

for cw in [100, 130]:
    print("\n===== cutoff start_week=%d =====" % cw)
    demand = T.H[:, cw : cw + 8].T
    r0 = sim.simulate(demand, initial, np.zeros((6, T.N_HIST)))
    print("ZERO orders   : score=%.2f" % r0["score"])
    rref = static_reference(cw)
    print("STATIC ref13  : score=%.2f" % rref["score"])
    print("   ", weekly_text(rref))
    res = T.backtest_fixed(dec, T.H, start_week=cw, initial=initial)
    print("MY policy %s : score=%.2f" % (params, res["score"]))
    print("   ", weekly_text(res))
