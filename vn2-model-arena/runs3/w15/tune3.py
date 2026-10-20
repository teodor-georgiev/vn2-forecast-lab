"""Final focused grid: refine around the global winner, validate on holdout."""

from __future__ import annotations

import itertools

import numpy as np

import tune as T
import tune2 as T2

Y, YC, INIT = T.Y, T.YC, T.INIT
CFG = [
    dict(scale=s, buffer=b, projection=p, base=base, adapt=a, trend=False)
    for s, b, p, base, a in itertools.product(
        [0.85, 0.95, 1.05, 1.15, 1.3],
        [0.0, 0.4],
        [1.05, 1.15, 1.25, 1.35],
        ["corr", "raw"],
        [4.0, 6.0, 9.0],
    )
]
print("n configs", len(CFG))
tune_windows = [95, 105, 115, 125, 130, 145]
hold_windows = [100, 120, 140, 149]

tot_t, _ = T2.evaluate(CFG, tune_windows, None)
rank = np.argsort(tot_t.mean(1))
top = list(rank[:10])
tot_h, _ = T2.evaluate([CFG[i] for i in top], hold_windows, None)
print("=== top-10 by tune mean (6 windows), with holdout mean (4 windows) ===")
for k, i in enumerate(top):
    print(f"tune {tot_t[i].mean():8.1f}  hold {tot_h[k].mean():8.1f}  {CFG[i]}")
