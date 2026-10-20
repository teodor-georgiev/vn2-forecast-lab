"""Last tuning round + timing (scratch)."""

from __future__ import annotations

import sys
import time
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
        )
    )
    rows = []
    variants = []
    for ti in ["norm", "emp"]:
        for ew in [52, 78, 104]:
            variants.append(
                (f"ti={ti} ew={ew}", dict(target_mode_interm=ti, err_window=ew))
            )
    variants.append(
        (
            "ti=emp ew=78 q_i=.7",
            dict(target_mode_interm="emp", err_window=78, q_interm=0.7),
        )
    )
    variants.append(
        (
            "ti=emp ew=78 q_i=.9",
            dict(target_mode_interm="emp", err_window=78, q_interm=0.9),
        )
    )
    variants.append(("season-shrink8", dict(season_shrink=8.0)))
    variants.append(("season-shrink2", dict(season_shrink=2.0)))
    variants.append(("trendwin13", dict(trend_window=13)))
    variants.append(("croston.3", dict(croston_alpha=0.3)))
    for label, over in variants:
        policy.PARAMS.clear()
        policy.PARAMS.update(base)
        policy.PARAMS.update(over)
        z, r = score_config(Y, I)
        rows.append((np.mean(r), np.mean(z), label))
        print(f"{label:22s} real {np.mean(r):8.1f}  zero {np.mean(z):8.1f}", flush=True)
    rows.sort()
    print("\nTOP:")
    for r in rows[:5]:
        print(f"  real {r[0]:8.1f} zero {r[1]:8.1f}  {r[2]}")

    # timing of a full 6-decision replay on the full history (as the manager runs it)
    policy.PARAMS.clear()
    policy.PARAMS.update(base)
    policy.PARAMS.update(dict(target_mode_interm="norm", err_window=78))
    policy._LIMIT = None
    policy._PREP = {}
    t0 = time.perf_counter()
    state = np.zeros((599, 3))
    state[:, 0] = 1.0
    for w in range(6):
        rev = np.zeros((w, 599)) if w else np.empty((0, 599))
        o = policy.decide(w, state.copy(), rev)
    print(
        f"\nfull replay timing (6 decisions, n=599, full history): {time.perf_counter() - t0:.2f}s"
    )


if __name__ == "__main__":
    main()
