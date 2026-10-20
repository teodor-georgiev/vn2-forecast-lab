"""Realistic dry run: real initial state + plausible hidden demand (weeks 149-156)."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hb_fixed as hb
from final_check import load_initial
from work import Y, D

initial = load_initial()
cut = 149
h = D  # corrected demand as the hidden truth
os.environ["VN2_CUTOFF"] = str(cut)
sys.modules.pop("policy", None)
import policy

t0 = time.perf_counter()
res = hb.backtest(policy.decide, h, cut, initial=initial)
el = time.perf_counter() - t0
print(
    f"dry run (cutoff {cut}, real initial state): score={res['score']:.0f}  time={el:.2f}s"
)
for i, wk in enumerate(res["weekly"]):
    print(
        f"  wk{i + 1}: ordered={wk['ordered']:7d} hold={wk['holding']:7.0f} "
        f"short={wk['shortage']:7.0f} demand={h[:, cut + i].sum():8.0f}"
    )
print("total ordered:", int(res["orders"].sum()))
