"""w02: final head-to-head over 10 historical cutoffs (identical initial state)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402
import calib  # noqa: E402

STARTS = (30, 40, 60, 80, 100, 115, 120, 130, 140, 145)
BASE = dict(
    mode="deficit",
    beta=0.0,
    W=3,
    start_off=0,
    hl=26.0,
    raw_hl=13.0,
    level_scale=1.0,
    damp=1.0,
    shrink_k=8.0,
    pool_years=3,
    fb_cap=0.6,
    fb_c=1.0,
)

CANDIDATES = {
    "A q1.65 q4.70 q5.85 bl.85 a2": dict(
        q1=0.65, q4=0.7, q5=0.85, blend=0.85, alpha=2.0
    ),
    "B q1.70 q4.80 q5.90 bl.90 a2": dict(q1=0.7, q4=0.8, q5=0.9, blend=0.9, alpha=2.0),
    "C q1.70 q4.80 q5.90 bl.85 a2": dict(q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0),
    "D q1.65 q4.80 q5.90 bl.90 a2": dict(q1=0.65, q4=0.8, q5=0.9, blend=0.9, alpha=2.0),
    "E q1.75 q4.85 q5.95 bl.90 a2": dict(
        q1=0.75, q4=0.85, q5=0.95, blend=0.9, alpha=2.0
    ),
    "F q1.70 q4.80 q5.90 bl1.0 a2": dict(q1=0.7, q4=0.8, q5=0.9, blend=1.0, alpha=2.0),
    "G q1.70 q4.80 q5.90 bl.90 a1": dict(q1=0.7, q4=0.8, q5=0.9, blend=0.9, alpha=1.0),
    "H hl6 raw6": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=6.0, raw_hl=6.0
    ),
    "I hl8 raw8": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=8.0, raw_hl=8.0
    ),
    "J hl13 raw13": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=13.0, raw_hl=13.0
    ),
    "K hl8 raw13": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=8.0, raw_hl=13.0
    ),
    "L hl13 raw8": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=13.0, raw_hl=8.0
    ),
    "M hl8 raw8 q1.65": dict(
        q1=0.65, q4=0.8, q5=0.9, blend=0.85, alpha=2.0, hl=8.0, raw_hl=8.0
    ),
    "N hl8 raw8 a1": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=1.0, hl=8.0, raw_hl=8.0
    ),
    "P fb c1 a1.0": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=1.0, hl=8.0, raw_hl=13.0, fb_c=1.0
    ),
    "Q fb c1 a0.75": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=0.75, hl=8.0, raw_hl=13.0, fb_c=1.0
    ),
    "R fb c2 a1.0": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=1.0, hl=8.0, raw_hl=13.0, fb_c=2.0
    ),
    "S fb c0.5 a1.0": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=1.0, hl=8.0, raw_hl=13.0, fb_c=0.5
    ),
    "T fb c4 a1.5": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.85, alpha=1.5, hl=8.0, raw_hl=13.0, fb_c=4.0
    ),
    "U ship a.40 c2 ls1.0": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.9, alpha=0.4, fb_c=2.0, hl=8.0, raw_hl=13.0
    ),
    "V a.55 c1 ls1.0": dict(
        q1=0.7, q4=0.8, q5=0.9, blend=0.9, alpha=0.55, fb_c=1.0, hl=8.0, raw_hl=13.0
    ),
    "W a.40 c2 ls0.9": dict(
        q1=0.7,
        q4=0.8,
        q5=0.9,
        blend=0.9,
        alpha=0.4,
        fb_c=2.0,
        hl=8.0,
        raw_hl=13.0,
        level_scale=0.9,
    ),
    "X a.55 c1 ls0.92": dict(
        q1=0.7,
        q4=0.8,
        q5=0.9,
        blend=0.9,
        alpha=0.55,
        fb_c=1.0,
        hl=8.0,
        raw_hl=13.0,
        level_scale=0.92,
    ),
    "Y a.70 c1 ls0.9": dict(
        q1=0.7,
        q4=0.8,
        q5=0.9,
        blend=0.9,
        alpha=0.7,
        fb_c=1.0,
        hl=8.0,
        raw_hl=13.0,
        level_scale=0.9,
    ),
    "Z a.55 c1.5 ls0.95": dict(
        q1=0.7,
        q4=0.8,
        q5=0.9,
        blend=0.9,
        alpha=0.55,
        fb_c=1.5,
        hl=8.0,
        raw_hl=13.0,
        level_scale=0.95,
    ),
}


def evaluate(p):
    pol.PARAMS.clear()
    pol.PARAMS.update(BASE)
    pol.PARAMS.update(p)
    pol._CACHE.clear()  # the prepare cache must follow the params
    tot = 0.0
    per = {}
    for s in STARTS:
        pol.HISTORY_END = s
        pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
        st = tune.make_initial(s)
        sc = tune.backtest(pol.decide, tune.SALES, s, initial=st)["score"]
        per[s] = sc
        tot += sc
    return tot, per


def reference():
    tot = 0.0
    per = {}
    for s in STARTS:
        pol.HISTORY_END = s
        pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
        st = tune.make_initial(s)
        sc = tune.backtest(
            calib.make_ref_policy(s, adaptive=True), tune.SALES, s, initial=st
        )["score"]
        per[s] = sc
        tot += sc
    return tot, per


rtot, rper = reference()
print(
    f"{'reference 13w':28s} TOT {rtot:9,.0f}   "
    + " ".join(f"{s}:{v:,.0f}" for s, v in rper.items())
)
for name, p in CANDIDATES.items():
    tot, per = evaluate(p)
    print(
        f"{name:28s} TOT {tot:9,.0f}   "
        + " ".join(f"{s}:{v:,.0f}" for s, v in per.items())
        + f"   ratio {tot / rtot:.3f}"
    )
