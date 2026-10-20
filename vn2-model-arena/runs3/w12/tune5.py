"""Per-class safety multiplier test (scratch)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402
from tune3 import CUTOFFS, score_config  # noqa: E402


def main():
    Y, I = bt.load()
    base = dict(policy.PARAMS)
    base.update(
        dict(
            target_mode="norm",
            target_mode_interm="norm",
            z_norm=0.8,
            trend_damp=0.0,
            level_window=8,
            err_window=104,
        )
    )
    variants = [
        ("base", {}),
        ("C=1.0", {"z_class": {"C": 1.0}}),
        ("C=1.2", {"z_class": {"C": 1.2}}),
        ("C=0.6", {"z_class": {"C": 0.6}}),
        ("B=1.0,C=1.2", {"z_class": {"B": 1.0, "C": 1.2}}),
        ("A=1.0,C=0.6", {"z_class": {"A": 1.0, "C": 0.6}}),
    ]
    rows = []
    for label, over in variants:
        policy.PARAMS.clear()
        policy.PARAMS.update(base)
        policy.PARAMS.update(over)
        z, r = score_config(Y, I)
        rows.append((np.mean(r), np.mean(z), label))
        print(f"{label:14s} real {np.mean(r):8.1f}  zero {np.mean(z):8.1f}", flush=True)
    rows.sort()
    print("\nTOP:")
    for r in rows[:4]:
        print(f"  real {r[0]:8.1f} zero {r[1]:8.1f}  {r[2]}")


if __name__ == "__main__":
    main()
