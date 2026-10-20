"""w02: sanity check of the real-game forecast (post-history panel)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy as pol  # noqa: E402

P = pol._prepare(None, None)
f, factor = pol._forecast(P, 0, None)
print("game weeks (doy):", P["gdoy"])
print("forecast total per game week:", np.round(f.sum(axis=1), 0))
print("forecast per series (mean):", np.round(f.mean(axis=1), 2))
print("implied 6 scored weeks total:", round(float(f[2:].sum()), 0))
print(
    "raw level total/week:",
    round(float(P["raw"].sum()), 0),
    " deseasonalised level total/week:",
    round(float(P["level"].sum()), 0),
)
kap = np.quantile(P["pools"][1], 0.70, axis=1)
print(
    "kappa1 (q=0.70) median/p90:",
    round(float(np.median(kap)), 2),
    round(float(np.quantile(kap, 0.9)), 2),
)
init = np.zeros((P["n"], 3))
init[:, 0] = 3.7
init[:, 2] = 2.8
for w in range(6):
    q = pol.decide(w, init.copy(), np.empty((0, P["n"])) if w == 0 else None)
    print(f"  week {w}: ordered {int(np.sum(q)):,}")
