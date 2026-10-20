"""Local fix of the manager's helper_backtest.py (w05 scratch, not a deliverable).

Identical flow, but the revealed history is actually accumulated each week
(the manager's helper forgot `revealed.append(...)`, so any adaptive policy
crashed at week 1).  Kept byte-for-byte equivalent otherwise.
"""

from __future__ import annotations

import numpy as np

ORDER_WEEKS = 6


def backtest(
    decide, history, start_week=0, weeks=8, initial=None, hold=0.20, short=1.00
):
    history = np.asarray(history, dtype=float)
    if history.ndim != 2:
        raise ValueError("history must be (n_series, n_weeks)")
    if weeks != 8:
        raise ValueError("the arena game has exactly 8 weeks")
    if start_week + weeks > history.shape[1]:
        raise ValueError("start_week + 8 runs past the known history")
    n = history.shape[0]
    if initial is None:
        state = np.zeros((n, 3), dtype=float)
    else:
        state = np.asarray(initial, dtype=float).copy()
        if state.shape != (n, 3):
            raise ValueError("initial must be (n_series, 3)")

    demand = history[:, start_week : start_week + weeks].T  # (8, n)
    revealed = []
    orders = np.zeros((ORDER_WEEKS, n))
    weekly = []
    for w in range(weeks):
        if w < ORDER_WEEKS:
            decision = decide(
                w,
                state.copy(),
                np.array(revealed, dtype=float).reshape(w, n),
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, dtype=float)))
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
        revealed.append(demand[w])  # <-- the missing line

    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}
