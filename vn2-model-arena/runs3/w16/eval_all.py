"""Evaluate the global level-adaptation policy across every cutoff 0..149."""

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
avail = pd.read_csv(ARENA / "data" / "availability.csv")
pavail = avail.pivot(
    index="series_id", columns="week", values="is_available"
).sort_index()
A = pavail.to_numpy(dtype=bool)
print("availability fraction:", A.mean())


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


def run(start_week, inflate=1.0, shrink_k=13, buffer_=0.75, scale=1.0, projection=1.0):
    dec = lambda wk, st, rev: decide(
        wk, st, rev, inflate, shrink_k, buffer_, scale, projection
    )
    return T.backtest_fixed(dec, T.H, start_week=start_week, initial=INITIAL)["score"]


scores = []
for cw in range(0, 150):
    if cw + 8 > T.H.shape[1]:
        break
    s = run(cw)
    scores.append(s)
scores = np.array(scores)
print("\nAll-cutoff summary (n=%d):" % len(scores))
print(
    "mean=%.2f  median=%.2f  min=%.2f  max=%.2f"
    % (scores.mean(), np.median(scores), scores.min(), scores.max())
)
print("pct below 3983: %.1f%%" % (100 * (scores < 3983).mean()))
print("pct below 4200: %.1f%%" % (100 * (scores < 4200).mean()))
print("pct below 4500: %.1f%%" % (100 * (scores < 4500).mean()))
print("\nworst 10 cutoffs:")
order = np.argsort(scores)[::-1][:10]
for cw in order:
    print("  cw=%d score=%.2f" % (cw, scores[cw]))
print("\nbest 10 cutoffs:")
order = np.argsort(scores)[:10]
for cw in order:
    print("  cw=%d score=%.2f" % (cw, scores[cw]))
