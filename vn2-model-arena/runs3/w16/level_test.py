"""Global level-adaptation policy test on c100 & c130 (real initial state)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
sys.path.insert(0, str(ARENA / "tools"))
sys.path.insert(0, str(ARENA / "runs3" / "w16"))
import inventory_sim as sim
import tuning as T

init = (
    pd.read_csv(ARENA / "data" / "initial_state.csv")
    .sort_values("series_id")
    .reset_index(drop=True)
)
INITIAL = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
BASE_TOTAL = float(T.mu.sum())
print("baseline total/wk:", BASE_TOTAL)


def decide(week_index, state, revealed, inflate, shrink_k, buffer_, scale, projection):
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    on_hand, arr1, arr2 = state.T
    total_inv = on_hand + arr1 + arr2
    w = int(week_index)
    base = T.mu * inflate
    base_total = BASE_TOTAL * inflate
    if w > 0:
        revealed = np.asarray(revealed, dtype=float)
        r_total = float(revealed.sum())
        alpha_obs = r_total / max(base_total * w, 1e-9)
        # shrink the level multiplier toward 1.0
        alpha = (w * alpha_obs + shrink_k * 1.0) / (w + shrink_k)
        alpha = float(np.clip(alpha, 0.3, 3.0))
        lam = base * alpha
    else:
        lam = base
    lam = np.maximum(lam, 1e-6)
    c = lam * scale
    projected = np.maximum(0.0, total_inv - projection * (c + c))
    deficit = c + buffer_ * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


def run(start_week, inflate, shrink_k, buffer_, scale, projection):
    dec = lambda wk, st, rev: decide(
        wk, st, rev, inflate, shrink_k, buffer_, scale, projection
    )
    return T.backtest_fixed(dec, T.H, start_week=start_week, initial=INITIAL)["score"]


print("inflate  k   buffer  c100     c130     mean")
best = None
for inflate in [1.0, 1.1]:
    for k in [13, 6, 3]:
        for buffer_ in [0.75, 1.0]:
            s100 = run(100, inflate, k, buffer_, 1.0, 1.0)
            s130 = run(130, inflate, k, buffer_, 1.0, 1.0)
            m = (s100 + s130) / 2
            print(
                "%.2f  %d  %.2f   %.2f  %.2f  %.2f"
                % (inflate, k, buffer_, s100, s130, m)
            )
            if best is None or m < best[0]:
                best = (m, inflate, k, buffer_, s100, s130)
print("\nBEST mean:", best)
