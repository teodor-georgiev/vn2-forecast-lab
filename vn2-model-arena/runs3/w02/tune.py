"""w02 tuner: backtest the policy on historical cutoffs and grid-search knobs.

NOTE: the manager's tools/helper_backtest.py never appends the realised week to
`revealed`, so it raises "cannot reshape array of size 0 into shape (1,599)" at
week 1.  `backtest` below is a byte-for-byte copy of that flow with the single
missing `revealed.append(demand[w])` line added, so the flow (lead time 2,
scored weeks 3-8, 1.00 missed / 0.20 held) is identical.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import policy as pol  # noqa: E402

DATA = HERE.parents[1] / "data"


def load():
    s = pd.read_csv(DATA / "sales.csv", parse_dates=["week"])
    sp = s.pivot(index="series_id", columns="week", values="y").sort_index(axis=1)
    DOY = np.asarray(sp.columns.dayofyear, dtype=int)
    return sp.to_numpy(float), DOY


SALES, DOY = load()


def backtest(
    decide, history, start_week=0, weeks=8, initial=None, hold=0.20, short=1.00
):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    if initial is None:
        state = np.zeros((n, 3), dtype=float)
    else:
        state = np.asarray(initial, dtype=float).copy()
    demand = history[:, start_week : start_week + weeks].T
    revealed = []
    orders = np.zeros((6, n))
    weekly = []
    for w in range(weeks):
        if w < 6:
            decision = decide(
                w, state.copy(), np.array(revealed, dtype=float).reshape(w, n)
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, dtype=float)))
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
    return {
        "score": float(score),
        "weekly": weekly,
        "orders": orders,
        "setup": sum(r["holding"] + r["shortage"] for r in weekly[:2]),
    }


def make_initial(start):
    """Proxy starting state, similar in shape to the arena's initial_state.csv."""
    pol.HISTORY_END = start
    pol.GAME_DOY = None
    P = pol._prepare(start, None)
    if P is None:
        return np.zeros((SALES.shape[0], 3))
    mu = P["level"]
    st = np.zeros((P["n"], 3))
    st[:, 0] = np.maximum(1.0, np.rint(1.3 * mu))
    st[:, 2] = np.rint(1.0 * mu)
    return st


def run(start, weeks=8, params=None, initial="proxy"):
    if params:
        pol.PARAMS.update(params)
    pol.HISTORY_END = start
    pol.GAME_DOY = DOY[start : start + 8].tolist()
    st = make_initial(start) if initial == "proxy" else None
    t0 = time.perf_counter()
    res = backtest(pol.decide, SALES, start, weeks=weeks, initial=st)
    res["elapsed"] = time.perf_counter() - t0
    res["start"] = start
    return res


def score_start(start, params=None, initial="proxy"):
    r = run(start, params=params, initial=initial)
    return r["score"], r["elapsed"]


if __name__ == "__main__":
    t0 = time.perf_counter()
    for start in (100, 130, 60, 145):
        sc, el = score_start(start)
        print(f"start {start:3d}  score {sc:10,.1f}   ({el:.2f}s)")
    print(f"total {time.perf_counter() - t0:.1f}s")
