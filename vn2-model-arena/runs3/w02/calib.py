"""w02: calibrate against the manager's reference bars on the same windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tune  # noqa: E402
import policy as pol  # noqa: E402

STARTS = (60, 100, 130, 145)


def reference_orders(forecast, state, scale=1.0, projection=1.0, buffer=0.75):
    """Copy of tools/inventory_sim.reference_orders."""
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) * scale for f in forecast]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


def make_ref_policy(start, adaptive=False):
    """13-week-mean forecast rule (the manager's 'adaptive' reference bar)."""
    hist = tune.SALES[:, max(0, start - 13) : start]

    def decide(week_index, state, revealed):
        f = hist.mean(axis=1)
        if adaptive and revealed is not None and len(revealed):
            rev = np.asarray(revealed, dtype=float)
            f = (f * (start - 13) + rev.sum(axis=0)) / ((start - 13) + rev.shape[0])
        fc = np.vstack([f, f, f])
        return reference_orders(fc, state)

    return decide


def zero_policy(week_index, state, revealed):
    return np.zeros(state.shape[0])


def demand_total(start):
    dem = tune.SALES[:, start : start + 8].T
    return dem[2:].sum()


if __name__ == "__main__":
    print(
        f"{'start':>6} {'demand3-8':>11} {'zero':>9} {'ref13w':>9} {'ref13w_ad':>10} {'mine':>9} {'ratio%':>7}"
    )
    for s in STARTS:
        dem = demand_total(s)
        z = tune.backtest(zero_policy, tune.SALES, s)["score"]
        r = tune.backtest(make_ref_policy(s), tune.SALES, s)["score"]
        ra = tune.backtest(make_ref_policy(s, adaptive=True), tune.SALES, s)["score"]
        pol.HISTORY_END = s
        pol.GAME_DOY = tune.DOY[s : s + 8].tolist()
        st = tune.make_initial(s)
        mine = tune.backtest(pol.decide, tune.SALES, s, initial=st)["score"]
        print(
            f"{s:6d} {dem:11,.0f} {z:9,.0f} {r:9,.0f} {ra:10,.0f} {mine:9,.0f} "
            f"{100 * mine / dem:6.1f}%"
        )

