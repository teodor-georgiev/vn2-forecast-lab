"""Final validation: timing, backtests at several cutoffs, reference comparison."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hb_fixed as hb
from work import Y, D

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"


def load_initial():
    ini = pd.read_csv(os.path.join(DATA, "initial_state.csv"))
    ids = np.array(sorted(ini["series_id"].astype(str).unique()))
    ini = ini.set_index(ini["series_id"].astype(str)).loc[ids]
    return np.column_stack(
        [
            ini["on_hand"].to_numpy(float),
            ini["arrival_w1"].to_numpy(float),
            ini["arrival_w2"].to_numpy(float),
        ]
    )


def no_order(h, start, initial=None):
    def decide(w, state, revealed):
        return np.zeros(h.shape[0])

    return hb.backtest(decide, h, start, initial=initial)


def reference_rule(h, start, initial=None, lookback=13, buffer=0.75):
    n = h.shape[0]
    hist = h[:, :start]

    def decide(w, state, revealed):
        L = hist.shape[1]
        means = [
            hist[:, max(0, L - lookback + k) : L + k].mean(axis=1) for k in (0, 1, 2)
        ]
        a, b, c = [np.maximum(0.0, m) for m in means]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - 1.0 * (a + b))
        deficit = c + buffer * np.sqrt(c) - projected
        return np.rint(np.clip(deficit, 0.0, None))

    return hb.backtest(decide, h, start, initial=initial)


def mine(h, start, cutoff, initial=None):
    os.environ["VN2_CUTOFF"] = str(cutoff)
    sys.modules.pop("policy", None)
    import policy

    t0 = time.perf_counter()
    res = hb.backtest(policy.decide, h, start, initial=initial)
    return res, time.perf_counter() - t0


if __name__ == "__main__":
    initial = load_initial()
    print(
        "initial state: on_hand sum=%.0f arr1=%.0f arr2=%.0f"
        % (initial[:, 0].sum(), initial[:, 1].sum(), initial[:, 2].sum())
    )
    print()
    for truth_name, h in (("raw", Y), ("corr", D)):
        for cut in (100, 130, 140, 149):
            for tag, init in (("zero-init", None), ("real-init", initial)):
                r0 = no_order(h, cut, init)
                r1 = reference_rule(h, cut, init)
                r2, el = mine(h, cut, cut, init)
                print(
                    f"truth={truth_name} cutoff={cut} {tag}: no-order={r0['score']:7.0f} "
                    f"ref13={r1['score']:7.0f}  MINE={r2['score']:7.0f}  "
                    f"(hold={sum(x['holding'] for x in r2['weekly'][2:]):6.0f} "
                    f"short={sum(x['shortage'] for x in r2['weekly'][2:]):6.0f}) "
                    f"game time={el:5.2f}s"
                )
        print()

    # full-game timing in the real configuration (whole history visible)
    os.environ["VN2_CUTOFF"] = "157"
    sys.modules.pop("policy", None)
    import policy

    t0 = time.perf_counter()
    st = initial.copy()
    rev = []
    for w in range(6):
        q = policy.decide(
            w, st.copy(), np.array(rev).reshape(w, 599) if rev else np.empty((0, 599))
        )
        print(f"  week {w}: ordered={q.sum():9.0f} units")
        on_hand, arr1, arr2 = st.T
        st = np.column_stack((np.maximum(0, on_hand + arr1), arr2, q))
        rev.append(np.zeros(599))
    print(
        f"REAL-GAME wall time (model build + 6 decisions): {time.perf_counter() - t0:.2f}s"
    )
