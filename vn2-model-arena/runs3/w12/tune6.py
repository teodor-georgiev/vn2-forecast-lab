"""z sweep (scratch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402
from tune3 import CUTOFFS  # noqa: E402


def score(Y, I, **over):
    policy.PARAMS.update(over)
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
    for z in [0.3, 0.5, 0.7, 0.9, 1.1]:
        zz, rr = score(Y, I, z_norm=z)
        print(
            f"z {z:4.2f} real mean {np.mean(rr):8.1f} worst {max(rr):8.1f} | "
            + " ".join(f"{x:6.0f}" for x in rr)
            + f" | zero mean {np.mean(zz):8.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
