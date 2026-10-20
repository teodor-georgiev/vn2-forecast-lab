"""w02: final validation — train cutoffs vs held-out cutoffs, mine vs reference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402
import calib  # noqa: E402

FINAL = dict(
    mode="deficit",
    q1=0.7,
    q4=0.8,
    q5=0.9,
    beta=0.0,
    blend=0.9,
    alpha=0.4,
    fb_c=2.0,
    hl=8.0,
    raw_hl=13.0,
    level_scale=1.0,
    damp=1.0,
    fb_cap=0.6,
    shrink_k=8.0,
    pool_years=3,
    W=3,
    start_off=0,
)
pol.PARAMS.update(FINAL)

TRAIN = (60, 100, 130, 145)
TEST = (30, 40, 80, 115, 120, 140)


def run_set(starts):
    rows = []
    for s in starts:
        pol.HISTORY_END = s
        pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
        st = tune.make_initial(s)
        mine = tune.backtest(pol.decide, tune.SALES, s, initial=st)["score"]
        ref = tune.backtest(
            calib.make_ref_policy(s, adaptive=True), tune.SALES, s, initial=st
        )["score"]
        zero = tune.backtest(calib.zero_policy, tune.SALES, s)["score"]
        dem = tune.SALES[:, s : s + 8].T[2:].sum()
        rows.append((s, zero, ref, mine, dem))
    return rows


for name, starts in (("TRAIN", TRAIN), ("TEST", TEST)):
    print(f"--- {name} ---")
    print(
        f"{'start':>6} {'demand3-8':>10} {'ref13w':>9} {'mine':>9} {'mine/ref':>9} {'mine/dem':>9}"
    )
    tot_m = tot_r = 0.0
    for s, zero, ref, mine, dem in run_set(starts):
        tot_m += mine
        tot_r += ref
        print(
            f"{s:6d} {dem:10,.0f} {ref:9,.0f} {mine:9,.0f} {mine / ref:9.3f} {100 * mine / dem:8.1f}%"
        )
    print(f"  TOTAL mine {tot_m:,.0f}  ref {tot_r:,.0f}  ratio {tot_m / tot_r:.3f}\n")
