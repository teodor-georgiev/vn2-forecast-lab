"""Final tuning round for worker w12 (scratch)."""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402

CUTOFFS = [85, 93, 101, 109, 117, 125, 133, 141, 149]


def score_config(Y, I, **overrides):
    policy.PARAMS.update(overrides)
    z, r = [], []
    for c in CUTOFFS:
        policy._LIMIT = c
        policy._PREP = {}
        z.append(bt.backtest(policy.decide, Y, start_week=c, initial=None)["score"])
        policy._LIMIT = c
        policy._PREP = {}
        r.append(bt.backtest(policy.decide, Y, start_week=c, initial=I)["score"])
    policy._LIMIT = None
    policy._PREP = {}
    return z, r


def main():
    Y, I = bt.load()
    base = dict(policy.PARAMS)
    t0 = time.perf_counter()
    configs = []
    for qs, qi in [
        (0.75, 0.75),
        (0.80, 0.80),
        (0.85, 0.85),
        (0.70, 0.70),
        (0.80, 0.90),
    ]:
        configs.append((f"q {qs}/{qi}", dict(q_smooth=qs, q_interm=qi)))
    configs.append(("no-season", dict(q_smooth=0.8, q_interm=0.8, seasonal_blend=0.0)))
    configs.append(
        ("season-half", dict(q_smooth=0.8, q_interm=0.8, seasonal_blend=0.5))
    )
    configs.append(("lw4", dict(q_smooth=0.8, q_interm=0.8, level_window=4)))
    configs.append(("lw13", dict(q_smooth=0.8, q_interm=0.8, level_window=13)))
    configs.append(("nodamp", dict(q_smooth=0.8, q_interm=0.8, trend_damp=0.0)))
    configs.append(("damp.8", dict(q_smooth=0.8, q_interm=0.8, trend_damp=0.8)))
    configs.append(
        ("norm-z1", dict(q_smooth=0.8, q_interm=0.8, target_mode="norm", z_norm=1.0))
    )
    for label, over in configs:
        policy.PARAMS.clear()
        policy.PARAMS.update(base)
        policy.PARAMS.update(over)
        z, r = score_config(Y, I)
        print(
            f"{label:14s} zero-init mean {np.mean(z):8.1f}   real-init mean {np.mean(r):8.1f}"
            f"   | real: " + " ".join(f"{x:6.0f}" for x in r),
            flush=True,
        )
    print(f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
