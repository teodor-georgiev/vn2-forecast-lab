"""Diagnostics + reference baselines at the same cutoffs."""

from __future__ import annotations

import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hb_fixed as hb
from work import Y, D, corrected_truth


def no_order(h, start):
    def decide(w, state, revealed):
        return np.zeros(h.shape[0])

    return hb.backtest(decide, h, start)


def reference_rule(h, start, lookback=13, buffer=0.75, scale=1.0):
    """13-week-mean forecast + the arena reference deficit rule."""
    n = h.shape[0]
    hist = h[:, :start]

    def decide(w, state, revealed):
        L = hist.shape[1]
        means = [
            hist[:, max(0, L - lookback + k) : L + k].mean(axis=1)
            if L + k > 0
            else np.zeros(n)
            for k in (0, 1, 2)
        ]
        a, b, c = [np.maximum(0.0, m) * scale for m in means]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - 1.0 * (a + b))
        deficit = c + buffer * np.sqrt(c) - projected
        return np.rint(np.clip(deficit, 0.0, None))

    return hb.backtest(decide, h, start)


def my_policy(h, start, cutoff, truth_label="raw"):
    os.environ["VN2_CUTOFF"] = str(cutoff)
    if "policy" in sys.modules:
        del sys.modules["policy"]
    import policy

    times = []
    orig = policy.decide

    def decide(w, state, revealed):
        t0 = time.perf_counter()
        q = orig(w, state, revealed)
        times.append(time.perf_counter() - t0)
        return q

    res = hb.backtest(decide, h, start)
    return res, times


if __name__ == "__main__":
    for cutoff in (100, 130):
        for name, h in (("raw", Y), ("corr", D)):
            r0 = no_order(h, cutoff)
            r1 = reference_rule(h, cutoff)
            r2, times = my_policy(h, cutoff, cutoff)
            dem = h[:, cutoff + 2 : cutoff + 8].sum()
            print(
                f"cutoff={cutoff} truth={name}: no-order={r0['score']:.0f} "
                f"ref13={r1['score']:.0f} mine={r2['score']:.0f} "
                f"| demand(w3-8)={dem:.0f} ordered={r2['orders'].sum():.0f} "
                f"| holding={sum(x['holding'] for x in r2['weekly'][2:]):.0f} "
                f"shortage={sum(x['shortage'] for x in r2['weekly'][2:]):.0f} "
                f"| decide times {[round(t, 2) for t in times]}"
            )
