"""Baseline comparison + diagnostics for worker w12."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402

CUTOFFS = [85, 100, 115, 130, 145]


def ref_decide_factory(Y):
    """Frozen contender: 13-week mean forecast + tools/inventory_sim.reference_orders."""
    T = Y.shape[1]

    def ref(week_index, state, revealed):
        n = state.shape[0]
        t = T + week_index
        hist = Y[:, max(0, t - 13) : t] if t > 0 else np.zeros((n, 1))
        m = hist.mean(axis=1) if hist.shape[1] else np.zeros(n)
        f = np.stack([m, m, m], axis=0)
        a, b, c = [np.maximum(0.0, f[i]) for i in range(3)]
        on_hand, arr1, arr2 = state[:, 0], state[:, 1], state[:, 2]
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - (a + b))
        deficit = c + 0.75 * np.sqrt(c) - projected
        return np.rint(np.clip(deficit, 0.0, None))

    return ref


def main():
    Y, I = bt.load()
    print("=== reference: 13-week mean + reference_orders ===")
    ref = ref_decide_factory(Y)
    for c in CUTOFFS:
        r = bt.backtest(ref, Y, start_week=c)
        print(f"  cutoff {c}: {r['score']:8.1f}")
    print("=== my policy ===")
    for c in CUTOFFS:
        policy._LIMIT = c
        policy._PREP = {}
        r = bt.backtest(policy.decide, Y, start_week=c)
        dem = Y[:, c : c + 8].sum()
        print(
            f"  cutoff {c}: {r['score']:8.1f}  ordered {r['orders'].sum():9.0f}  demand8 {dem:9.0f}"
        )
        if c == 100:
            for row in r["weekly"]:
                print(
                    f"     w{row['week']} hold {row['holding']:8.1f} short {row['shortage']:8.1f} ord {row['ordered']:8d}"
                )
    policy._LIMIT = None
    policy._PREP = {}


if __name__ == "__main__":
    main()
