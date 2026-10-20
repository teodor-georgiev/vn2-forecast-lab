"""Corrected copy of the manager's helper_backtest (that one never appends the
revealed demand, so it crashes at week_index 1). Flow is otherwise identical;
parity is checked against the manager helper for reveal-blind policies.

    hb2.backtest(decide, history, start_week, initial=None)
    history: (n_series, n_weeks); decide(week_index, state, revealed)
    revealed: (week_index, n_series) actual demand so far.
"""

from __future__ import annotations

import numpy as np

ORDER_WEEKS = 6


def backtest(
    decide, history, start_week=0, weeks=8, initial=None, hold=0.20, short=1.00
):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    if history.ndim != 2 or weeks != 8:
        raise ValueError("bad input")
    if start_week + weeks > history.shape[1]:
        raise ValueError("start_week + 8 runs past the known history")
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
                w, state.copy(), np.array(revealed, dtype=float).reshape(w, n)
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
        revealed.append(demand[w].copy())  # <-- the manager's helper misses this

    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}


def parity_check():
    """Compare against the frozen engine tools/inventory_sim.order_plan_from_policy."""
    import importlib.util
    from pathlib import Path

    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "invsim", here.parent.parent / "tools" / "inventory_sim.py"
    )
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)

    rng = np.random.default_rng(7)
    n, T = 20, 60
    hist_full = np.maximum(0, rng.poisson(3.0, size=(n, T))).astype(float)

    for s in (0, 10, 30):
        H = hist_full[:, :s]

        def decide(w, state, revealed, H=H):
            rev = np.asarray(revealed, dtype=float).reshape(w, n).T
            avail = np.concatenate([H, rev], axis=1) if w else H
            a = np.maximum(avail[:, -13:].mean(axis=1), 0.0)
            return sim.reference_orders(
                [a, a, a], state, scale=1.1, projection=1.0, buffer=0.9
            )

        mine = backtest(decide, hist_full, start_week=s, initial=None)

        def forecast_fn(w, state, history, H=H):
            rev = np.zeros((n, 0)) if history is None else np.asarray(history, float).T
            avail = np.concatenate([H, rev], axis=1) if w else H
            a = np.maximum(avail[:, -13:].mean(axis=1), 0.0)
            return (a, a, a)

        theirs = sim.order_plan_from_policy(
            hist_full[:, s : s + 8],
            np.zeros((n, 3)),
            forecast_fn,
            policy=lambda f, st: sim.reference_orders(
                f, st, scale=1.1, projection=1.0, buffer=0.9
            ),
        )
        assert abs(mine["score"] - theirs["score"]) < 1e-9, (
            mine["score"],
            theirs["score"],
        )
    print("parity OK vs frozen engine inventory_sim.order_plan_from_policy")


if __name__ == "__main__":
    parity_check()
