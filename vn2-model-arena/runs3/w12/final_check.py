"""Final validation of the deliverable policy.py (scratch)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402
from cmp import ref_decide_factory  # noqa: E402

CUTOFFS = [85, 93, 101, 109, 117, 125, 133, 141, 149]


def main():
    Y, I = bt.load()
    ref = ref_decide_factory(Y)
    print("cutoff | reference(13w mean) | w12 policy | delta")
    rz, pz, rr, pr = [], [], [], []
    for c in CUTOFFS:
        a = bt.backtest(ref, Y, start_week=c, initial=I)["score"]
        policy._LIMIT = c
        policy._PREP = {}
        b = bt.backtest(policy.decide, Y, start_week=c, initial=I)["score"]
        policy._LIMIT = c
        policy._PREP = {}
        a0 = bt.backtest(ref, Y, start_week=c, initial=None)["score"]
        policy._LIMIT = c
        policy._PREP = {}
        b0 = bt.backtest(policy.decide, Y, start_week=c, initial=None)["score"]
        rr.append(a)
        pr.append(b)
        rz.append(a0)
        pz.append(b0)
        print(
            f"  {c:4d} | {a:9.1f} | {b:9.1f} | {b - a:+8.1f}   (zero-init {a0:8.1f} -> {b0:8.1f})"
        )
    print(
        f"\nMEAN real-init: reference {np.mean(rr):8.1f}  policy {np.mean(pr):8.1f}"
        f"  delta {np.mean(pr) - np.mean(rr):+8.1f} ({100 * (np.mean(pr) - np.mean(rr)) / np.mean(rr):+.1f}%)"
    )
    print(
        f"MEAN zero-init : reference {np.mean(rz):8.1f}  policy {np.mean(pz):8.1f}"
        f"  delta {np.mean(pz) - np.mean(rz):+8.1f} ({100 * (np.mean(pz) - np.mean(rz)) / np.mean(rz):+.1f}%)"
    )
    policy._LIMIT = None
    policy._PREP = {}


if __name__ == "__main__":
    main()
