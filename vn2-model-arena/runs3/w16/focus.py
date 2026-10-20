"""Focused test: faster adaptation to revealed demand on c100 & c130 (real initial state)."""

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


def decide_adapt(
    week_index,
    state,
    revealed,
    scale,
    buffer_,
    projection,
    inflate,
    shrink_k,
    safety_mode,
    adapt_mode,
):
    """adapt_mode: 'blend' = (w*r + k*mu)/(w+k); 'revealed_after2' = revealed mean if w>=2 else mu."""
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    on_hand, arr1, arr2 = state.T
    total_inv = on_hand + arr1 + arr2
    w = int(week_index)
    base = T.mu * inflate
    if w > 0:
        revealed = np.asarray(revealed, dtype=float)
        r = revealed.mean(axis=0) if revealed.size else np.zeros(n)
        if adapt_mode == "revealed_after2" and w >= 2:
            lam = r
        else:
            lam = (w * r + shrink_k * base) / (w + shrink_k)
    else:
        lam = base
    lam = np.maximum(lam, 1e-6)
    c = lam * scale
    projected = np.maximum(0.0, total_inv - projection * (c + c))
    safety = buffer_ * np.sqrt(c) if safety_mode == "sqrt" else buffer_ * T.sigma
    deficit = c + safety - projected
    return np.rint(np.clip(deficit, 0.0, None))


def run(
    start_week, scale, buffer_, projection, inflate, shrink_k, safety_mode, adapt_mode
):
    dec = lambda wk, st, rev: decide_adapt(
        wk,
        st,
        rev,
        scale,
        buffer_,
        projection,
        inflate,
        shrink_k,
        safety_mode,
        adapt_mode,
    )
    return T.backtest_fixed(dec, T.H, start_week=start_week, initial=INITIAL)["score"]


print("cw  scale buffer proj inflate k  mode       adapt       c100    c130")
results = []
for scale in [1.0, 1.15]:
    for inflate in [1.0, 1.15]:
        for k in [1, 3, 6, 13]:
            for adapt in ["blend", "revealed_after2"]:
                s100 = run(100, scale, 0.75, 1.0, inflate, k, "sqrt", adapt)
                s130 = run(130, scale, 0.75, 1.0, inflate, k, "sqrt", adapt)
                results.append((s100 + s130, s100, s130, scale, inflate, k, adapt))
                print(
                    "%d  %.2f  0.75  1.0   %.2f   %d  sqrt    %-9s %.2f  %.2f"
                    % (100, scale, inflate, k, adapt, s100, s130)
                )
results.sort()
print("\nBEST (mean c100+c130):", results[0])
