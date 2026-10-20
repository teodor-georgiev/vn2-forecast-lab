"""w02: inspect the recent history and the same-season weeks of earlier years."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402

s = pd.read_csv(HERE.parents[1] / "data" / "sales.csv", parse_dates=["week"])
sp = s.pivot(index="series_id", columns="week", values="y").sort_index(axis=1)
tot = sp.sum(axis=0)
dates = sp.columns
print("last 30 weeks of history (aggregate demand):")
for i in range(len(dates) - 30, len(dates)):
    print(f"  {dates[i].date()}  {tot.iloc[i]:8,.0f}")
print("\nsame calendar weeks (day-of-year 1..70) in every year:")
doy = dates.dayofyear
for d in range(1, 70, 3):
    row = []
    for yv in (2019, 2020, 2021, 2022):
        sel = np.where((doy == d) & (dates.year == yv))[0]
        row.append(f"{yv}:{tot.iloc[sel[0]]:7,.0f}" if sel.size else f"{yv}:      -")
    print(f"  doy {d:3d}  " + "  ".join(row))
