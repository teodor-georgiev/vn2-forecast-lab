"""Final focused tuning (scratch)."""

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
    rows = []
    for z, damp, lw in itertools.product([0.8, 1.0, 1.3], [0.0, 0.3], [8, 13]):
        over = dict(
            target_mode="norm",
            target_mode_interm="norm",
            z_norm=z,
            trend_damp=damp,
            level_window=lw,
        )
        policy.PARAMS.clear()
        policy.PARAMS.update(base)
        policy.PARAMS.update(over)
        zz, rr = score_config(Y, I)
        rows.append((np.mean(rr), np.mean(zz), z, damp, lw))
        print(
            f"z {z:4.2f} damp {damp:3.1f} lw {lw:2d} -> real {np.mean(rr):8.1f}  zero {np.mean(zz):8.1f}",
            flush=True,
        )
    rows.sort()
    print("\nTOP:")
    for r in rows[:6]:
        print(f"  real {r[0]:8.1f} zero {r[1]:8.1f}  z={r[2]} damp={r[3]} lw={r[4]}")
    print(f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
