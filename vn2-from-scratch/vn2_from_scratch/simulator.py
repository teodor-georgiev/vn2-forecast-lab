"""Exact VN2 stock-flow simulator.

Semantics (see CONTRACT.md):
    Order round d (1-based) is received at the start of game week d+2 and is
    sellable there. Game weeks are 1..8 (0-indexed 0..7). Holding is charged
    EUR 0.20 per unit of ENDING inventory per week, with no charge while goods
    are in transit. Shortage is EUR 1.00 per lost unit, with no backorders.
    Only game weeks 3..8 are scored.
"""

from __future__ import annotations

import numpy as np

from .config import GAME_WEEKS, HOLDING_COST, N_DECISIONS, SHORTAGE_COST

SCORE_START = 2  # 0-indexed: game week 3


def arrival_schedule(
    orders: np.ndarray, pipe_w1: np.ndarray, pipe_w2: np.ndarray
) -> np.ndarray:
    """Build the (8, n) arrival schedule from the two in-flight orders and rounds."""
    n = pipe_w1.shape[0]
    arr = np.empty((GAME_WEEKS, n), dtype=np.float64)
    arr[0] = pipe_w1
    arr[1] = pipe_w2
    if N_DECISIONS:
        arr[2:] = orders  # orders (6, n): row i is round i+1, arriving week i+3
    return arr


class SimResult:
    """Container for one 8-week simulation."""

    __slots__ = (
        "total_cost",
        "holding_cost",
        "shortage_cost",
        "weekly_total",
        "weekly_holding",
        "weekly_shortage",
        "scored_cost",
        "scored_holding",
        "scored_shortage",
        "fill_rate_scored",
        "demand_scored",
        "lost_units_scored",
        "held_units_scored",
        "scored_demand",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw[name])

    def as_dict(self) -> dict:
        return {
            "total_cost": float(self.total_cost),
            "holding_cost": float(self.holding_cost),
            "shortage_cost": float(self.shortage_cost),
            "weekly_total": self.weekly_total.tolist(),
            "weekly_holding": self.weekly_holding.tolist(),
            "weekly_shortage": self.weekly_shortage.tolist(),
            "scored_cost": float(self.scored_cost),
            "scored_holding": float(self.scored_holding),
            "scored_shortage": float(self.scored_shortage),
            "fill_rate_scored": float(self.fill_rate_scored),
        }


def simulate(
    orders: np.ndarray,
    demand: np.ndarray,
    on_hand0: np.ndarray,
    pipe_w1: np.ndarray,
    pipe_w2: np.ndarray,
) -> SimResult:
    """Roll the 8-week stock flow forward and price it.

    orders : (6, N) non-negative quantities, index 0 is round 1.
    demand : (8, N) demand per game week, index 0 is game week 1.
    """
    orders = np.clip(np.asarray(orders, dtype=np.float64), 0.0, None)
    demand = np.asarray(demand, dtype=np.float64)
    oh = np.clip(np.asarray(on_hand0, dtype=np.float64), 0.0, None)
    p1 = np.clip(np.asarray(pipe_w1, dtype=np.float64), 0.0, None)
    p2 = np.clip(np.asarray(pipe_w2, dtype=np.float64), 0.0, None)

    arr = arrival_schedule(orders, p1, p2)

    if demand.shape != arr.shape:
        raise ValueError(
            f"demand must be (game_weeks, series) = {arr.shape}, got {demand.shape}. "
            "Transpose a (series, game_weeks) array before calling."
        )

    weekly_holding = np.zeros(GAME_WEEKS, dtype=np.float64)
    weekly_shortage = np.zeros(GAME_WEEKS, dtype=np.float64)
    weekly_held_units = np.zeros(GAME_WEEKS, dtype=np.float64)
    weekly_lost_units = np.zeros(GAME_WEEKS, dtype=np.float64)
    weekly_demand_units = np.zeros(GAME_WEEKS, dtype=np.float64)

    for t in range(GAME_WEEKS):
        avail = oh + arr[t]
        d = demand[t]
        lost = np.maximum(0.0, d - avail)
        oh_end = np.maximum(0.0, avail - d)
        weekly_holding[t] = HOLDING_COST * oh_end.sum()
        weekly_shortage[t] = SHORTAGE_COST * lost.sum()
        weekly_held_units[t] = oh_end.sum()
        weekly_lost_units[t] = lost.sum()
        weekly_demand_units[t] = d.sum()
        oh = oh_end

    weekly_total = weekly_holding + weekly_shortage

    sl = slice(SCORE_START, GAME_WEEKS)
    scored_holding = float(weekly_holding[sl].sum())
    scored_shortage = float(weekly_shortage[sl].sum())
    scored_demand = float(weekly_demand_units[sl].sum())
    lost_scored = float(weekly_lost_units[sl].sum())
    held_scored = float(weekly_held_units[sl].sum())

    return SimResult(
        total_cost=float(weekly_total.sum()),
        holding_cost=float(weekly_holding.sum()),
        shortage_cost=float(weekly_shortage.sum()),
        weekly_total=weekly_total,
        weekly_holding=weekly_holding,
        weekly_shortage=weekly_shortage,
        scored_cost=scored_holding + scored_shortage,
        scored_holding=scored_holding,
        scored_shortage=scored_shortage,
        fill_rate_scored=1.0 - (lost_scored / scored_demand)
        if scored_demand > 0
        else 1.0,
        demand_scored=scored_demand,
        lost_units_scored=lost_scored,
        held_units_scored=held_scored,
        scored_demand=scored_demand,
    )


def pipeline_at(orders: np.ndarray, d: int, pipe_w1: np.ndarray, pipe_w2: np.ndarray):
    """Pipeline visible at decision round ``d`` (1-based).

    Returns ``(arrivals_now, arrivals_next)``: what is due in the week about to
    start, and in the week after it. Both are already-committed quantities.
    """
    if d == 1:
        return pipe_w1, pipe_w2
    if d == 2:
        return pipe_w2, orders[0]
    return orders[d - 3], orders[d - 2]


def simulate_many(
    demand: np.ndarray,
    on_hand0: np.ndarray,
    pipe_w1: np.ndarray,
    pipe_w2: np.ndarray,
    order_fn,
    forecasts=None,
    **kwargs,
) -> SimResult:
    """Decide each round with the real information set, then roll one simulation.

    ``order_fn(d, on_hand, arrivals_now, arrivals_next, fc, **kwargs)`` is called
    once per round. ``on_hand`` is the ending inventory of game week d-1, i.e.
    exactly what a planner knows when placing order d. ``fc`` is
    ``forecasts[d - 1]``, a ``(3, n)`` block of horizons 1..3 seen at that moment.
    """
    n = demand.shape[1]
    orders = np.zeros((N_DECISIONS, n), dtype=np.float64)
    oh = np.clip(np.asarray(on_hand0, dtype=np.float64), 0.0, None)
    p1 = np.clip(np.asarray(pipe_w1, dtype=np.float64), 0.0, None)
    p2 = np.clip(np.asarray(pipe_w2, dtype=np.float64), 0.0, None)

    for d in range(1, N_DECISIONS + 1):
        t = d - 1
        a_now, a_next = pipeline_at(orders, d, p1, p2)
        fc = None if forecasts is None else forecasts[t]
        orders[t] = order_fn(d, oh, a_now, a_next, fc, **kwargs)

        # Sell game week t, leaving the on-hand state the next round observes.
        arr_t = p1 if t == 0 else (p2 if t == 1 else orders[t - 2])
        oh = np.maximum(0.0, oh + arr_t - demand[t])

    return simulate(orders, demand, on_hand0, pipe_w1, pipe_w2)
