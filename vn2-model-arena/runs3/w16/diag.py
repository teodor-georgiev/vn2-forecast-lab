"""Diagnose demand structure over the 157-week history and calibrate baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
DATA = ARENA + r"\data"

sales = pd.read_csv(DATA + r"\sales.csv")
avail = pd.read_csv(DATA + r"\availability.csv")
psales = sales.pivot(index="series_id", columns="week", values="y").sort_index()
pavail = avail.pivot(
    index="series_id", columns="week", values="is_available"
).sort_index()
H = psales.to_numpy(float)  # (599,157) sorted series_id order
A = pavail.to_numpy(bool)
print("H shape:", H.shape, "A shape:", A.shape)
print("censored fraction (avail=0):", (~A).mean())
print("sales mean over avail weeks:", np.nanmean(np.where(A, H, np.nan)))
print("raw sales mean (censored as 0):", H.mean())

weekly_total = H.sum(axis=0)
print("\nWeekly total demand:")
print("first 8:", np.round(weekly_total[:8], 1))
print("last 8 :", np.round(weekly_total[-8:], 1))
print(
    "min:",
    weekly_total.min(),
    "max:",
    weekly_total.max(),
    "mean:",
    round(weekly_total.mean(), 1),
)
for cw in [0, 100, 130, 149]:
    print(
        f"window {cw}-{cw + 7} total demand:",
        round(weekly_total[cw : cw + 8].sum(), 1),
        "mean/wk:",
        round(weekly_total[cw : cw + 8].mean(), 1),
    )

avail_cnt = A.sum(axis=1)
am = np.where(
    avail_cnt > 0, np.where(A, H, 0.0).sum(axis=1) / np.maximum(avail_cnt, 1), H.mean()
)
print(
    "\navail-mean baseline: min",
    round(am.min(), 3),
    "max",
    round(am.max(), 3),
    "mean",
    round(am.mean(), 3),
    "total/wk:",
    round(am.sum(), 1),
)
print(
    "full-history per-series mean (censored=0): mean",
    round(H.mean(axis=1).mean(), 3),
    "total/wk:",
    round(H.mean(axis=1).sum(), 1),
)
