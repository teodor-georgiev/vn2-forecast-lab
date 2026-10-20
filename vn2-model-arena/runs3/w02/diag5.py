"""w02: check the imputed demand against the raw sales for the recent weeks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy as pol  # noqa: E402

P = pol._prepare(None, None)
raw = pol._raw()
sales, avail, sids, weeks = raw
dem, T = P["dem"], P["T"]
dates = pd.DatetimeIndex(weeks)
print(f"{'week':>12} {'sales':>8} {'imputed':>8} {'avail%':>7} {'censor cells':>13}")
for t in range(T - 16, T):
    print(
        f"{str(dates[t].date()):>12} {sales[:, t].sum():8,.0f} {dem[:, t].sum():8,.0f} "
        f"{100 * avail[:, t].mean():6.1f}% {int((avail[:, t] == 0).sum()):13d}"
    )
print("\nlevel check:")
print(
    "  dem last 13 weeks total:",
    round(float(dem[:, T - 13 :].sum()), 0),
    " sales:",
    round(float(sales[:, T - 13 :].sum()), 0),
)
print(
    "  raw EWMA total:",
    round(float(P["raw"].sum()), 0),
    " deseasonalised level total:",
    round(float(P["level"].sum()), 0),
)
print(
    "  deseasonalised factor (seas) at recent doys:",
    np.round(P["seas"][:, P["doy"][T - 13 :] - 1].mean(axis=0), 3),
)
