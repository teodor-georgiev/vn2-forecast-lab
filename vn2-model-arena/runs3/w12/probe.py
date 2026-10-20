"""Calibration probe: replicate the manager's reference bars on candidate windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402
import policy  # noqa: E402


def ref_decide_factory(Y, window=13, buffer=0.75, projection=1.0):
    T = Y.shape[1]

    def ref(week_index, state, revealed):
        n = state.shape[0]
        t = T + week_index
        hist = Y[:, max(0, t - window) : t]
        m = hist.mean(axis=1) if hist.shape[1] else np.zeros(n)
        f = np.stack([m, m, m], axis=0)
        a, b, c = [np.maximum(0.0, f[i]) for i in range(3)]
        on_hand, arr1, arr2 = state[:, 0], state[:, 1], state[:, 2]
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
        deficit = c + buffer * np.sqrt(c) - projected
        return np.rint(np.clip(deficit, 0.0, None))

    return ref


def main():
    Y, I = bt.load()
    ref = ref_decide_factory(Y)
    for c in [149, 141, 133, 125, 117, 109, 101, 93, 85]:
        r0 = bt.backtest(ref, Y, start_week=c, initial=None)
        r1 = bt.backtest(ref, Y, start_week=c, initial=I)
        print(
            f"cutoff {c:4d}  zero-init {r0['score']:8.1f}   real-init {r1['score']:8.1f}"
            f"   demand8 {Y[:, c : c + 8].sum():8.0f}"
        )


if __name__ == "__main__":
    main()
