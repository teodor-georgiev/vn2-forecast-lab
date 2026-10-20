"""w02: in-sample check of the seasonal/level model + comparison with the reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402

start = 100
pol.HISTORY_END = start
pol.GAME_DOY = tune.DOY[start : start + 8].tolist()
P = pol._prepare(start, tune.DOY[start : start + 8].tolist())
dem, seas, level = P["dem"], P["seas"], P["level"]
doy = P["doy"]
actual = tune.SALES[:, start : start + 8].T

# in-sample: actual vs model over the last 26 history weeks
print("in-sample check (last 26 history weeks): model = level * seas[doy]")
for t in range(P["T"] - 26, P["T"]):
    mod = (level * seas[:, doy[t] - 1]).sum()
    act = dem[:, t].sum()
    print(
        f"  t={t:3d} doy={doy[t]:3d} model {mod:9,.0f}  act {act:9,.0f}  ratio {act / max(mod, 1e-9):5.2f}"
    )

# aggregate profile check
print(
    "\nportfolio profile at a few doys:", {d: round(float(np.mean([1])), 3) for d in []}
)
raw = pol._raw()
sales = raw[0]
dates = pd.DatetimeIndex(raw[3]) if False else None
import pandas as pd

dates = pd.DatetimeIndex(raw[3])
tot = sales.sum(axis=0)
for d in (10, 20, 60, 180, 300, 342, 356, 363):
    sel = np.where(dates.dayofyear == d)[0]
    if sel.size:
        print(f"  doy {d:3d}: weekly totals {np.round(tot[sel], 0)}")

# what does the 13-week mean say for the window?
m13 = tune.SALES[:, start - 13 : start].mean(axis=1)
print(
    "\n13w-mean total per week:",
    m13.sum(),
    " vs actual mean/wk:",
    actual.mean(axis=1).sum(),
)
f, factor = pol._forecast(P, 0, None)
print("my forecast totals:", np.round(f.sum(axis=1), 0))
print("actual totals      :", np.round(actual.sum(axis=1), 0))
print(
    "level total/wk:",
    level.sum(),
    " seas at game doys:",
    np.round(P["seas"][:, P["gdoy"] - 1].sum(axis=0), 3),
)
