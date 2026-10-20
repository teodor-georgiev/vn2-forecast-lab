"""Parameter tuning for worker w12 (scratch)."""

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

CUTOFFS = [85, 100, 115, 130, 145]


def score_config(Y, **overrides):
    policy.PARAMS.update(overrides)
    out = []
    for c in CUTOFFS:
        policy._LIMIT = c
        policy._PREP = {}
        r = bt.backtest(policy.decide, Y, start_week=c)
        out.append(r["score"])
    policy._LIMIT = None
    policy._PREP = {}
    return out


def main():
    Y, I = bt.load()
    grid_qs = [0.80, 0.88, 0.93, 0.97]
    grid_qi = [0.80, 0.90, 0.97]
    t0 = time.perf_counter()
    rows = []
    for qs, qi in itertools.product(grid_qs, grid_qi):
        s = score_config(Y, q_smooth=qs, q_interm=qi)
        rows.append((np.mean(s), qs, qi, s))
        print(
            f"q_smooth {qs:5.2f} q_interm {qi:5.2f} -> "
            + " ".join(f"{x:7.0f}" for x in s)
            + f"  mean {np.mean(s):8.1f}",
            flush=True,
        )
    rows.sort()
    print("\nBEST:", rows[0][:3], rows[0][3], f"({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
