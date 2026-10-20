"""Experiment driver: backtest policy variants on historical cutoffs (no leakage).

NOTE: the manager's helper_backtest.py never appends to its `revealed` list, so it
raises ValueError for w>=1 for any policy; selfcheck.py also has a broadcast bug
((N,HIST)*(N,)).  This driver therefore replicates the exact flow of the
authoritative engine tools/inventory_sim.py::order_plan_from_policy (which does
append revealed), including the same scoring.
"""

import importlib
import sys
import time

import numpy as np
import pandas as pd

BASE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w01")

sales = pd.read_csv(BASE + r"\sales.csv")
avail = pd.read_csv(BASE + r"\availability.csv")
_p = sales.pivot(index="series_id", columns="week", values="y").sort_index()
S = _p.to_numpy(float)
A = (
    avail.pivot(index="series_id", columns="week", values="is_available")
    .sort_index()
    .to_numpy(bool)
)
INIT = (
    pd.read_csv(BASE + r"\initial_state.csv")
    .set_index("series_id")
    .reindex(_p.index)[["on_hand", "arrival_w1", "arrival_w2"]]
    .to_numpy(float)
)
N = S.shape[0]
rng0 = np.random.default_rng(7)
PERM = rng0.permutation(N)
PERM2 = rng0.permutation(N)


def replay(decide, history, start_week, initial, hold=0.20, short=1.00):
    """Same flow/scoring as tools/inventory_sim.py order_plan_from_policy."""
    n = history.shape[0]
    demand = history[:, start_week : start_week + 8].T
    state = np.asarray(initial, dtype=float).copy()
    revealed = []
    weekly = []
    for w in range(8):
        order = np.zeros(n)
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(
                    0.0,
                    np.nan_to_num(
                        np.asarray(decide(w, state.copy(), rev), dtype=float)
                    ),
                )
            )
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(hold * end.sum() + short * missed.sum())
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    return float(sum(weekly[2:]))


def run(cfg, cutoffs=(100, 130), shuffle=False):
    pol = importlib.import_module("policy")
    importlib.reload(pol)
    t0 = time.perf_counter()
    scores = {}
    fit_secs = {}
    for c in cutoffs:
        t1 = time.perf_counter()
        pol.fit(cutoff=c, cfg=cfg)
        fit_secs[c] = round(time.perf_counter() - t1, 2)
        init = INIT[PERM] if shuffle else INIT
        scores[c] = replay(pol.decide, S, c, init)
    el = time.perf_counter() - t0
    return scores, el, fit_secs


def ref_bar(cutoffs=(100, 130)):
    sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
    import inventory_sim as inv

    sc = {}
    for c in cutoffs:
        mu = S[:, :c].mean(1)

        def forecast_fn(w, state, hist, mu=mu):
            return np.array([mu, mu, mu])

        def pol(forecast, state):
            return inv.reference_orders(forecast, state)

        r = inv.order_plan_from_policy(S[:, c : c + 8].T, INIT, forecast_fn, pol)
        sc[c] = r["score"]
    return sc
