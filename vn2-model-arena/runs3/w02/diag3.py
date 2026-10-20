"""w02: side-by-side weekly breakdown, mine vs the manager's reference rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402
import calib  # noqa: E402

start = int(sys.argv[1]) if len(sys.argv) > 1 else 100
pol.PARAMS.update({"q": 0.65, "W": 3, "start_off": 0})
pol.HISTORY_END = start
pol.GAME_DOY = tune.DOY[start : start + 8].tolist()
st = tune.make_initial(start)

mine = tune.backtest(pol.decide, tune.SALES, start, initial=st)
ref = tune.backtest(calib.make_ref_policy(start, adaptive=True), tune.SALES, start)
zero = tune.backtest(calib.zero_policy, tune.SALES, start)
dem = tune.SALES[:, start : start + 8].T

print(f"start {start}")
print(
    f"{'wk':>3} {'demand':>8} | {'mine ord':>9} {'mine h':>7} {'mine s':>7} | {'ref ord':>8} {'ref h':>7} {'ref s':>7}"
)
for w in range(8):
    m, r = mine["weekly"][w], ref["weekly"][w]
    print(
        f"{w + 1:3d} {dem[w].sum():8,.0f} | {m['ordered']:9d} {m['holding']:7,.0f} {m['shortage']:7,.0f}"
        f" | {r['ordered']:8d} {r['holding']:7,.0f} {r['shortage']:7,.0f}"
    )
print(
    f"score mine {mine['score']:,.0f}  ref {ref['score']:,.0f}  zero {zero['score']:,.0f}"
)
print(
    "mine orders total", mine["orders"].sum(), " ref orders total", ref["orders"].sum()
)
