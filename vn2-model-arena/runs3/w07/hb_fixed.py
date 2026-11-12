"""Backtest an adaptive policy on a historical window (manager-provided helper).

Mirrors the frozen scoring engine `tools/inventory_sim.py` step by step:

    supply_w = on_hand + delivery arr1
    served   = min(supply, demand); missed = demand - supply (>= 0)
    end_inv  = max(0, supply - demand)
    state    = (end_inv, arr2, order_placed_now)
    an order returned at week_index w is included in the supply of week_index w+2

Usage from your own scripts:

    import helper_backtest as hb
    result = hb.backtest(decide, history, start_week, initial=state)
    # decide: your policy.py decide(week_index, state, revealed)
    # history: (n_series, n_weeks) numpy array of KNOWN weekly sales
    # start_week: 0-based column of the game's first week inside history
    # initial: optional (n_series, 3) starting state (on_hand, delivery_w1, delivery_w2);
    #          default is all zeros
    # result: {"score": EUR weeks 3-8, "weekly": [...], "orders": {...}}

The score uses 1.00 per missed unit and 0.20 per held unit-week, weeks 3-8, exactly
like the arena. `decide` is called for week_index 0..5 only (six order rounds), as
in the real game. Do NOT score backtests against hidden future demand — use weeks
you already have. This helper never touches the arena's private answer panel.
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
        revealed.append(demand[w])
        state = np.column_stack((end, arr2, order))

    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}
