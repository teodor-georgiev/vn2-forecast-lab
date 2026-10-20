"""w02: final report — selfcheck, backtests at the requested cutoffs, timing."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402
import calib  # noqa: E402

print(
    "shipped PARAMS:",
    {
        k: v
        for k, v in pol.PARAMS.items()
        if k
        in (
            "mode",
            "q1",
            "q4",
            "q5",
            "blend",
            "alpha",
            "fb_c",
            "hl",
            "raw_hl",
            "shrink_k",
            "pool_years",
        )
    },
)
print()
print(
    subprocess.run(
        [sys.executable, str(HERE / "selfcheck.py")],
        capture_output=True,
        text=True,
        cwd=str(HERE),
    ).stdout.strip()
)
print()
print(f"{'cutoff':>7} {'demand3-8':>10} {'ref 13w':>9} {'mine':>9} {'mine/ref':>9}")
tot_m = tot_r = 0.0
for s in (100, 130, 60, 145, 30, 40, 80, 115, 120, 140):
    pol.HISTORY_END = s
    pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
    st = tune.make_initial(s)
    t0 = time.perf_counter()
    mine = tune.backtest(pol.decide, tune.SALES, s, initial=st)["score"]
    el = time.perf_counter() - t0
    ref = tune.backtest(
        calib.make_ref_policy(s, adaptive=True), tune.SALES, s, initial=st
    )["score"]
    dem = tune.SALES[:, s : s + 8].T[2:].sum()
    tot_m += mine
    tot_r += ref
    tag = " *" if s in (100, 130) else "  "
    print(
        f"{s:7d} {dem:10,.0f} {ref:9,.0f} {mine:9,.0f} {mine / ref:9.3f}{tag}  ({el:.2f}s)"
    )
print(f"{'TOTAL':>7} {'':>10} {tot_r:9,.0f} {tot_m:9,.0f} {tot_m / tot_r:9.3f}")
