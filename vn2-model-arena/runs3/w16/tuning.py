"""Tune the order-rule parameters for w16 policy via historical cutoffs.

Mirrors the arena simulator (inventory_sim.order_plan_from_policy) including the
weekly reveal of actual demand. Grid-searches (scale, buffer, projection,
censoring inflation, shrinkage, safety mode) over cutoffs. Hard-codes the best
parameters into policy.py.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
HERE = ARENA / "runs3" / "w16"
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA = ARENA / "data"
ORDER_WEEKS = 6
WEEKS = 8
SHORT = 1.00
HOLD = 0.20


def load_history():
    sales = pd.read_csv(DATA / "sales.csv")
    avail = pd.read_csv(DATA / "availability.csv")
    psales = sales.pivot(index="series_id", columns="week", values="y").sort_index()
    pavail = avail.pivot(
        index="series_id", columns="week", values="is_available"
    ).sort_index()
    H = psales.to_numpy(dtype=float)  # (599,157) sorted series_id order == game order
    A = pavail.to_numpy(dtype=bool)
    return H, A


H, A = load_history()
N_HIST = H.shape[0]

_init = (
    pd.read_csv(DATA / "initial_state.csv")
    .sort_values("series_id")
    .reset_index(drop=True)
)
INITIAL = _init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)

# Censoring-corrected baseline: mean over available (not sold-out) weeks.
avail_cnt = A.sum(axis=1)
avail_sum = np.where(A, H, 0.0).sum(axis=1)
mu = np.where(avail_cnt > 0, avail_sum / np.maximum(avail_cnt, 1), H.mean(axis=1))

# Per-series demand volatility over available weeks (for std-based safety stock).
sq = np.where(A, H * H, 0.0).sum(axis=1)
var = np.where(avail_cnt > 1, sq / np.maximum(avail_cnt, 1) - mu**2, 0.0)
sigma = np.sqrt(np.maximum(var, 0.0))
# Never let safety stock fall below a small Poisson-like floor.
sigma_floor = np.sqrt(np.maximum(mu, 1e-6))
sigma = np.maximum(sigma, 0.25 * sigma_floor)


def decide_params(
    week_index,
    state,
    revealed,
    scale,
    buffer_,
    projection,
    inflate,
    shrink_k,
    safety_mode,
):
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    on_hand, arr1, arr2 = state.T
    total_inv = on_hand + arr1 + arr2

    w = int(week_index)
    if n == N_HIST:
        base = mu * inflate
        if w > 0:
            revealed = np.asarray(revealed, dtype=float)
            r = revealed.mean(axis=0) if revealed.size else np.zeros(n)
            lam = (w * r + shrink_k * base) / (w + shrink_k)
        else:
            lam = base
    else:
        # Fallback for synthetic / mismatched n (selfcheck).
        if w > 0:
            revealed = np.asarray(revealed, dtype=float)
            lam = revealed.mean(axis=0) if revealed.size else np.full(n, 1.0)
        else:
            lam = np.full(n, 1.0)
    lam = np.maximum(lam, 1e-6)

    c = lam * scale  # flat forecast h1 = h2 = h3 = c
    projected = np.maximum(0.0, total_inv - projection * (c + c))
    if safety_mode == "std":
        safety = buffer_ * sigma[:n] if n == N_HIST else buffer_ * np.sqrt(c)
    else:
        safety = buffer_ * np.sqrt(c)
    deficit = c + safety - projected
    return np.rint(np.clip(deficit, 0.0, None))


def backtest_fixed(decide, history, start_week=0, initial=None):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    if initial is None:
        state = np.zeros((n, 3), dtype=float)
    else:
        state = np.asarray(initial, dtype=float).copy()
    demand = history[:, start_week : start_week + WEEKS].T
    revealed = []
    orders = np.zeros((ORDER_WEEKS, n))
    weekly = []
    for w in range(WEEKS):
        if w < ORDER_WEEKS:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            decision = decide(w, state.copy(), rev)
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, dtype=float)))
        order = orders[w] if w < ORDER_WEEKS else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {
                "holding": HOLD * end.sum(),
                "shortage": SHORT * missed.sum(),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}


def run_backtest(params, start_week):
    scale, buffer_, projection, inflate, shrink_k, safety_mode = params
    dec = lambda wk, st, rev: decide_params(
        wk, st, rev, scale, buffer_, projection, inflate, shrink_k, safety_mode
    )
    return backtest_fixed(dec, H, start_week=start_week, initial=INITIAL)["score"]


GRID = {
    "scale": [1.0, 1.3, 1.6],
    "buffer": [0.6, 1.0, 1.4],
    "projection": [0.75, 1.0],
    "inflate": [1.0, 1.3],
    "shrink_k": [3, 13],
    "safety_mode": ["sqrt", "std"],
}

cutoffs = [100, 130]
keys = list(GRID.keys())
results = []
for combo in itertools.product(*[GRID[k] for k in keys]):
    params = dict(zip(keys, combo))
    scores = [run_backtest(tuple(params[k] for k in keys), cw) for cw in cutoffs]
    results.append((float(np.mean(scores)), scores, params))

results.sort(key=lambda x: x[0])
print("=== Best parameterizations (lower mean score better) ===")
for mean_score, scores, params in results[:10]:
    print(
        f"mean={mean_score:9.2f}  c100={scores[0]:9.2f}  c130={scores[1]:9.2f}  {params}"
    )
best = results[0]
print(
    "\nBEST:",
    best[2],
    "mean=",
    round(best[0], 2),
    "scores=",
    [round(s, 2) for s in best[1]],
)

Path(HERE / "best_params.json").write_text(
    json.dumps({"params": best[2], "mean_score": best[0], "scores": best[1]}, indent=2)
)
print("written best_params.json")
