"""Corrected backtester: mirrors tools/inventory_sim.order_plan_from_policy exactly.

(helper_backtest.py in this directory never appends to `revealed`, so it crashes at
week_index 1 for every policy; this mirrors the authoritative engine instead.)
"""

from __future__ import annotations
import numpy as np

WEEKS = 8
ORDER_WEEKS = 6


def backtest(
    decide, history, start_week=0, initial=None, hold=0.20, short=1.00, weeks=8
):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    demand = history[:, start_week : start_week + weeks].T
    state = np.zeros((n, 3)) if initial is None else np.asarray(initial, float).copy()
    revealed = []
    orders = np.zeros((ORDER_WEEKS, n))
    weekly = []
    for w in range(weeks):
        order = np.zeros(n)
        if w < ORDER_WEEKS:
            r = np.array(revealed, dtype=float) if revealed else np.empty((0, n))
            dec = np.asarray(decide(w, state.copy(), r), dtype=float).ravel()
            assert dec.shape == (n,) and np.isfinite(dec).all()
            order = np.rint(np.clip(dec, 0.0, None))
            orders[w] = order
        supply = state[:, 0] + state[:, 1]
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
        state = np.column_stack((end, state[:, 2], order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {
        "score": float(score),
        "weekly": weekly,
        "orders": orders,
        "setup": sum(r["holding"] + r["shortage"] for r in weekly[:2]),
    }
