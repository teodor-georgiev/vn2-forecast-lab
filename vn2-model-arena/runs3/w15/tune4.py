"""Head-to-head of finalists across tune + holdout + robustness windows."""

import numpy as np
import tune as T
import tune2 as T2

Y, YC, INIT = T.Y, T.YC, T.INIT

F = [
    dict(scale=0.95, buffer=0.0, projection=1.15, base="corr", adapt=6.0, trend=False),
    dict(scale=0.95, buffer=0.0, projection=1.20, base="corr", adapt=6.0, trend=False),
    dict(scale=0.95, buffer=0.0, projection=1.20, base="raw", adapt=6.0, trend=False),
    dict(scale=0.95, buffer=0.0, projection=1.35, base="corr", adapt=6.0, trend=False),
    dict(scale=1.05, buffer=0.0, projection=1.25, base="raw", adapt=6.0, trend=False),
    dict(scale=0.95, buffer=0.4, projection=1.20, base="corr", adapt=6.0, trend=False),
    dict(scale=1.00, buffer=0.75, projection=1.00, base="raw", adapt=0.0, trend=False),
]
W = [90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 149]
tot, per = T2.evaluate(F, W, None)
for i, cfg in enumerate(F):
    print(
        f"{tot[i].mean():8.1f}  per-window {np.round(tot[i], 0).astype(int).tolist()}  {cfg}"
    )
