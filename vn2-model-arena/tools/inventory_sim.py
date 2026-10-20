"""VN2-style inventory simulator — the proven research implementation, extracted.

Cost model and stock flow are byte-for-byte equivalent to the working research
script `run_vn2_lgbm.py` (functions `order_for` and `simulate`, 2026-10-08),
verified by `scripts/parity_check_simulator.py`.

Rules
-----
- 8 weekly periods. Orders may be placed in weeks 1..6 only; weeks 7-8 have no orders.
- An order placed in week w arrives in week w+2. State = (on_hand, arrival_w+1, arrival_w+2).
- Week flow: supply = on_hand + arrival_w+1; served = min(supply, demand);
  end_inventory = max(0, supply - demand); missed = max(0, demand - supply).
- Week cost = 1.00 * missed_units + 0.20 * end_inventory_units.
- The scored result covers weeks 3..8 (weeks 1-2 are the settling period, reported
  separately as `setup`).

Everything is expressed per series; arrays are (weeks, n_series) / (n_series, ...).
Demand units are the anonymised arena units.
"""

from __future__ import annotations

import numpy as np

WEEKS = 8
ORDER_WEEKS = 6
SHORTAGE_COST = 1.00
HOLDING_COST = 0.20
SCORED_FROM_WEEK = 3  # 1-based; weeks 1-2 are setup


def reference_orders(forecast, state, scale=1.0, projection=1.0, buffer=0.75):
    """The reference deficit rule used by the frozen research contender.

    forecast: (3, n) arrays for h1, h2, h3 demand forecasts.
    state:    (n, 3) arrays (on_hand, arrival_w+1, arrival_w+2).
    Returns integer order quantities (n,) for this week.
    """
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) * scale for f in forecast]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


def simulate(demand, initial, orders):
    """Score a fixed order plan against known demand.

    demand:  (8, n) true demand per week, weeks 1..8.
    initial: (n, 3) starting (on_hand, arrival_w+1, arrival_w+2).
    orders:  (6, n) order quantities for weeks 1..6.

    Returns dict(score=EUR weeks 3-8, setup=EUR weeks 1-2, weekly=[...]).
    """
    demand = np.asarray(demand, dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    orders = np.asarray(orders, dtype=float)
    weekly = []
    for w in range(WEEKS):
        order = orders[w] if w < ORDER_WEEKS else np.zeros(demand.shape[1])
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        hold_cost = HOLDING_COST * end.sum()
        short_cost = SHORTAGE_COST * missed.sum()
        weekly.append(
            {
                "week": w + 1,
                "holding": float(hold_cost),
                "shortage": float(short_cost),
                "total": float(hold_cost + short_cost),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
    score = sum(row["total"] for row in weekly[SCORED_FROM_WEEK - 1 :])
    setup = sum(row["total"] for row in weekly[: SCORED_FROM_WEEK - 1])
    return {"score": score, "setup": setup, "weekly": weekly}


def order_plan_from_policy(demand, initial, forecast_fn, policy=None):
    """Simulate an adaptive policy with the same flow as `simulate`.

    forecast_fn(week_index, state, history) -> (3, n) demand forecasts for h1..h3,
    where history is the (week_index, n) demand actually revealed so far.
    policy(forecast, state) -> order quantities (n,); defaults to reference_orders.
    """
    if policy is None:
        policy = reference_orders
    demand = np.asarray(demand, dtype=float)
    state = np.asarray(initial, dtype=float).copy()
    revealed: list[np.ndarray] = []
    weekly = []
    orders = np.zeros((ORDER_WEEKS, demand.shape[1]))
    for w in range(WEEKS):
        order = np.zeros(demand.shape[1])
        if w < ORDER_WEEKS:
            forecast = forecast_fn(w, state, np.array(revealed) if revealed else None)
            order = np.asarray(policy(forecast, state), dtype=float)
            orders[w] = order
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        hold_cost = HOLDING_COST * end.sum()
        short_cost = SHORTAGE_COST * missed.sum()
        weekly.append(
            {
                "week": w + 1,
                "holding": float(hold_cost),
                "shortage": float(short_cost),
                "total": float(hold_cost + short_cost),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(row["total"] for row in weekly[SCORED_FROM_WEEK - 1 :])
    setup = sum(row["total"] for row in weekly[: SCORED_FROM_WEEK - 1])
    return {
        "score": score,
        "setup": setup,
        "weekly": weekly,
        "orders": orders,
        "states": state,
    }
