"""Trace a single c130 run to understand inventory dynamics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
sys.path.insert(0, str(ARENA / "tools"))
sys.path.insert(0, str(ARENA / "runs3" / "w16"))
import tuning as T

init = (
    pd.read_csv(ARENA / "data" / "initial_state.csv")
    .sort_values("series_id")
    .reset_index(drop=True)
)
INITIAL = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
BASE_TOTAL = float(T.mu.sum())


def run_trace(
    start_week, inflate=1.0, shrink_k=13, buffer_=0.75, scale=1.0, projection=1.0
):
    state = INITIAL.copy()
    n = state.shape[0]
    demand = T.H[:, start_week : start_week + 8].T
    revealed = []
    total_ordered = 0.0
    print(
        "initial total_inv=%.1f on_hand=%.1f arr1=%.1f arr2=%.1f"
        % (state.sum(), state[:, 0].sum(), state[:, 1].sum(), state[:, 2].sum())
    )
    print("baseline total/wk=%.1f" % BASE_TOTAL)
    for w in range(8):
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            base = T.mu * inflate
            base_total = BASE_TOTAL * inflate
            if w > 0:
                r_total = float(np.asarray(revealed).sum())
                alpha_obs = r_total / max(base_total * w, 1e-9)
                alpha = (w * alpha_obs + shrink_k * 1.0) / (w + shrink_k)
                alpha = float(np.clip(alpha, 0.3, 3.0))
                lam = base * alpha
            else:
                lam = base
            lam = np.maximum(lam, 1e-6)
            c = lam * scale
            on_hand, arr1, arr2 = state.T
            total_inv = on_hand + arr1 + arr2
            projected = np.maximum(0.0, total_inv - projection * (c + c))
            order = np.rint(np.clip(c + buffer_ * np.sqrt(c) - projected, 0.0, None))
        else:
            order = np.zeros(n)
        total_ordered += order.sum()
        on_hand, arr1, arr2 = state.T
        supply = on_hand + arr1
        d = demand[w]
        end = np.maximum(0.0, supply - d)
        missed = np.maximum(0.0, d - supply)
        print(
            "w%d: demand=%.1f supply=%.1f order=%.1f end=%.1f missed=%.1f inv_position=%.1f"
            % (
                w + 1,
                d.sum(),
                supply.sum(),
                order.sum(),
                end.sum(),
                missed.sum(),
                total_inv.sum(),
            )
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(d)
    print("total ordered=%.1f total demand w1-8=%.1f" % (total_ordered, demand.sum()))
    scored_shortage = sum(
        np.maximum(0.0, demand[w] - (np.zeros(n) if False else 0)) for w in range(2, 8)
    )


run_trace(130)
