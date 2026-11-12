"""Independent second stock-flow implementation, for reconciliation.

`simulator.simulate` walks the eight weeks with a per-week loop over an
arrivals vector. This module reaches the same numbers through a different
route, so that a shared misunderstanding in one is caught rather than
propagated: it builds the arrival contributions with an explicit placement
map, and it carries a closed-form check for the no-stockout case where the
inventory recursion collapses to a cumulative sum.
"""

from __future__ import annotations

import numpy as np

from .config import GAME_WEEKS, HOLDING_COST, N_DECISIONS, SHORTAGE_COST

SCORE_START = 2


def independent_simulate(orders, demand, on_hand0, pipe_w1, pipe_w2):
    """Return (total, holding, shortage) for game weeks 1..8."""
    orders = np.clip(np.asarray(orders, dtype=np.float64), 0.0, None)
    demand = np.atleast_2d(np.asarray(demand, dtype=np.float64))
    if demand.shape[0] != GAME_WEEKS and demand.shape[1] == GAME_WEEKS:
        demand = demand.T
    n = demand.shape[1]

    # Placement map: a quantity ordered in round r (0-based) lands in week r+2.
    landed = np.zeros((GAME_WEEKS, n), dtype=np.float64)
    landed[0] = np.clip(np.asarray(pipe_w1, dtype=np.float64), 0.0, None)
    landed[1] = np.clip(np.asarray(pipe_w2, dtype=np.float64), 0.0, None)
    for r in range(N_DECISIONS):
        landed[r + 2] += orders[r]

    stock = np.clip(np.asarray(on_hand0, dtype=np.float64), 0.0, None).copy()
    hold = np.zeros(GAME_WEEKS)
    short = np.zeros(GAME_WEEKS)
    for w in range(GAME_WEEKS):
        opened = stock + landed[w]
        shortfall = np.clip(demand[w] - opened, 0.0, None)
        stock = np.clip(opened - demand[w], 0.0, None)
        hold[w] = HOLDING_COST * float(stock.sum())
        short[w] = SHORTAGE_COST * float(shortfall.sum())

    return (
        float(hold.sum() + short.sum()),
        float(hold.sum()),
        float(short.sum()),
        float(hold[SCORE_START:].sum() + short[SCORE_START:].sum()),
    )


def closed_form_no_shortage(orders, demand, on_hand0, pipe_w1, pipe_w2):
    """Ending inventory per week when supply always covers demand.

    With no stockouts the recursion is linear, so week-w ending stock is the
    opening stock plus the cumulative surplus through w. This is the oracle the
    loop implementation is checked against in the no-shortage regime.
    """
    orders = np.clip(np.asarray(orders, dtype=np.float64), 0.0, None)
    demand = np.atleast_2d(np.asarray(demand, dtype=np.float64))
    if demand.shape[0] != GAME_WEEKS and demand.shape[1] == GAME_WEEKS:
        demand = demand.T
    landed = np.zeros((GAME_WEEKS, demand.shape[1]))
    landed[0] = np.clip(np.asarray(pipe_w1, dtype=np.float64), 0.0, None)
    landed[1] = np.clip(np.asarray(pipe_w2, dtype=np.float64), 0.0, None)
    for r in range(N_DECISIONS):
        landed[r + 2] += orders[r]
    surplus = landed - demand
    return np.clip(np.asarray(on_hand0, dtype=np.float64), 0.0, None) + np.cumsum(
        surplus, axis=0
    )


def randomized_crosscheck(simulate_fn, n=25, seed=0):
    """Max absolute discrepancy between the two implementations on random inputs."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        k = int(rng.integers(1, 60))
        orders = rng.integers(0, 12, size=(N_DECISIONS, k)).astype(float)
        demand = rng.integers(0, 9, size=(GAME_WEEKS, k)).astype(float)
        if _ % 5 == 0:
            demand[:] = 0.0
        if _ % 7 == 0:
            orders[:] = 0.0
        on_hand = rng.integers(0, 6, size=k).astype(float)
        p1 = rng.integers(0, 4, size=k).astype(float)
        p2 = rng.integers(0, 4, size=k).astype(float)
        a = simulate_fn(orders, demand, on_hand, p1, p2)
        at, ah, asc = (
            (a.total_cost, a.holding_cost, a.shortage_cost)
            if not isinstance(a, tuple)
            else a[:3]
        )
        b_total, b_hold, b_short, _ = independent_simulate(
            orders, demand, on_hand, p1, p2
        )
        worst = max(worst, abs(at - b_total), abs(ah - b_hold), abs(asc - b_short))
    return worst
