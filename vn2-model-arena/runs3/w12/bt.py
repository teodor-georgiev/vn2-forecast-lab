"""Backtest harness for worker w12 (scratch, not the deliverable)."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import helper_backtest as hb  # noqa: E402

DATA = HERE.parents[1] / "data"


def backtest(
    decide, history, start_week=0, weeks=8, initial=None, hold=0.20, short=1.00
):
    """Mirror of helper_backtest.backtest, but with the revealed demand appended.

    helper_backtest.py never appends `demand[w]` to `revealed`, so it always hands
    the policy an empty history and even crashes at week 1 (reshape of a size-0
    array into (1, n)).  tools/inventory_sim.py shows the intended flow
    (`revealed.append(demand[w])`), so this is the corrected version.
    """
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    demand = history[:, start_week : start_week + weeks].T  # (8, n)
    if initial is None:
        state = np.zeros((n, 3))
    else:
        state = np.asarray(initial, dtype=float).copy()
    revealed = []
    orders = np.zeros((6, n))
    weekly = []
    for w in range(weeks):
        if w < 6:
            dec = decide(
                w,
                state.copy(),
                np.array(revealed, dtype=float).reshape(w, n)
                if w
                else np.empty((0, n)),
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(dec, dtype=float)))
        order = orders[w] if w < 6 else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {
                "week": w + 1,
                "holding": float(hold * end.sum()),
                "shortage": float(short * missed.sum()),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}


def load():
    sales = pd.read_csv(DATA / "sales.csv")
    ids = sorted(sales["series_id"].unique().tolist())
    weeks = sorted(sales["week"].unique().tolist())
    Y = (
        sales.pivot(index="series_id", columns="week", values="y")
        .reindex(index=ids, columns=weeks)
        .values.astype(float)
    )
    init = pd.read_csv(DATA / "initial_state.csv")
    init = init.sort_values("series_id").reset_index(drop=True)
    I = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
    return Y, I


def run(mod, cutoffs, initial=None, label=""):
    Y, I = load()
    out = {}
    for c in cutoffs:
        mod._LIMIT = c
        mod._PREP = {}
        r = backtest(mod.decide, Y, start_week=c, initial=initial)
        out[c] = r["score"]
    mod._LIMIT = None
    mod._PREP = {}
    vals = list(out.values())
    print(
        f"{label:38s} "
        + " ".join(f"{c}:{s:8.1f}" for c, s in out.items())
        + f"  mean {np.mean(vals):8.1f}"
    )
    return out


if __name__ == "__main__":
    import policy

    t0 = time.perf_counter()
    Y, I = load()
    print("load", time.perf_counter() - t0)
    cutoffs = [85, 100, 115, 130, 145]
    print("--- zero initial state ---")
    run(policy, cutoffs, None, "default")
    print("--- real initial state (only meaningful as a shape check) ---")
    run(policy, [100, 130], I, "default+realinit")
    print("elapsed", time.perf_counter() - t0)
