"""Faithful local backtester (my own, in my dir).

The manager's helper_backtest.py never appends to `revealed`, so it crashes at
week_index 1 (reshape of an empty array). This mirrors tools/inventory_sim.py's
order_plan_from_policy flow instead, which does append the revealed demand.
"""

from __future__ import annotations

import numpy as np

ORDER_WEEKS = 6


def backtest(
    decide, history, start_week=0, weeks=8, initial=None, hold=0.20, short=1.00
):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    if start_week + weeks > history.shape[1]:
        raise ValueError("start_week + 8 runs past the known history")
    state = (
        np.zeros((n, 3)) if initial is None else np.asarray(initial, dtype=float).copy()
    )
    demand = history[:, start_week : start_week + weeks].T  # (8, n)
    revealed = []
    orders = np.zeros((ORDER_WEEKS, n))
    weekly = []
    for w in range(weeks):
        if w < ORDER_WEEKS:
            dec = decide(
                w,
                state.copy(),
                np.array(revealed, dtype=float).reshape(w, n)
                if w
                else np.empty((0, n)),
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(dec, dtype=float)))
        order = orders[w] if w < ORDER_WEEKS else np.zeros(n)
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
