"""Final verification of policy.py: contract, timing, and historical replays."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data"
import sys

sys.path.insert(0, str(HERE.parent.parent / "tools"))
import inventory_sim as sim  # noqa: E402


def load_policy():
    spec = importlib.util.spec_from_file_location("agent_policy", HERE / "policy.py")
    mod = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    spec.loader.exec_module(mod)
    print(f"import+history-load: {time.perf_counter() - t0:.2f}s")
    return mod.decide


def replay(decide, demand8, initial):
    """Full 8-week replay exactly like the arena engine (orders weeks 0..5)."""
    state = np.asarray(initial, dtype=float).copy()
    n = state.shape[0]
    orders = np.zeros((6, n))
    revealed = []
    t0 = time.perf_counter()
    for w in range(8):
        if w < 6:
            d = decide(w, state.copy(), np.array(revealed).reshape(w, n))
            orders[w] = np.rint(np.maximum(0.0, np.asarray(d, dtype=float)))
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand8[w])
        missed = np.maximum(0.0, demand8[w] - supply)
        state = np.column_stack((end, arr2, orders[w] if w < 6 else np.zeros(n)))
        revealed.append(demand8[w].copy())
    el = time.perf_counter() - t0
    res = sim.simulate(demand8, initial, orders)
    return res, el, orders


def main():
    decide = load_policy()
    sales = pd.read_csv(DATA / "sales.csv")
    weeks = sorted(sales["week"].unique())
    series = sorted(sales["series_id"].unique())
    sidx = {s: i for i, s in enumerate(series)}
    widx = {w: i for i, w in enumerate(weeks)}
    n, t = len(series), len(weeks)
    y = np.zeros((n, t))
    y[sales["series_id"].map(sidx).to_numpy(), sales["week"].map(widx).to_numpy()] = (
        sales["y"].to_numpy()
    )
    init = pd.read_csv(DATA / "initial_state.csv")[
        ["on_hand", "arrival_w1", "arrival_w2"]
    ].to_numpy(float)

    print("--- contract on real-shaped game (599 series) ---")
    state = init.copy()
    for w in range(6):
        d = np.asarray(decide(w, state.copy(), y[:, :w].T.copy()), dtype=float).ravel()
        assert d.shape == (n,) and np.isfinite(d).all() and (d >= 0).all(), w
        on_hand, arr1, arr2 = state.T.copy()
        end = np.maximum(0.0, on_hand + arr1 - y[:, w])
        state = np.column_stack((end, arr2, d))
    print("shape/finiteness/sign OK for weeks 0..5")

    print("--- historical replays (score = weeks 3-8, EUR) ---")
    for s in (100, 115, 130, 145, 149):
        dem = y[:, s : s + 8].T
        res, el, orders = replay(decide, dem, init)
        print(
            f"  start {s:3d}: score {res['score']:8.1f}  setup {res['setup']:7.1f}  "
            f"ordered {int(orders.sum()):6d}  replay {el:.2f}s"
        )

    print("--- reference bar rule for comparison ---")
    for s in (100, 115, 130, 145, 149):
        dem = y[:, s : s + 8].T
        hist = y[:, :s]

        def ref_decide(w, st, rev, hist=hist):
            a = np.maximum(hist[:, -13:].mean(axis=1), 0.0)
            return sim.reference_orders(
                [a, a, a], st, scale=1.0, projection=1.0, buffer=0.75
            )

        res, el, _ = replay(ref_decide, dem, init)
        print(f"  start {s:3d}: score {res['score']:8.1f}")


if __name__ == "__main__":
    main()
