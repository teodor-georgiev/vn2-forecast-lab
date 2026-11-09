"""Monte Carlo expected-cost scoring of fixed order matrices.

``mc_score`` prices one (6, n_series) order matrix against a panel of
full 8-week demand scenarios of shape (n_scenarios, n_series, 8), as the
mean of the existing ``simulator.simulate()`` ``scored_cost`` -- weeks
3-8 only, holding EUR 0.20 per unit of ending inventory, shortage EUR
1.00 per lost unit, no backorders, no new stock-flow code. Scenario
panels are the horizon expansion of the SIP-style packs emitted by
``scenarios.pack_scenarios`` (3-horizon draws of shape
(n_scenarios, n_series, 3)): rolling the 3-horizon pack forward and
stacking horizon-1 draws yields the (S, N, 8) panels scored here. In
production the panel carries the scenario index, so cross-week
dependence of a scenario is preserved end to end.

``best_of`` scores every candidate with ``mc_score`` and returns the
minimum (first wins on ties). The functions are pure in their inputs --
no RNG, no mutation -- so the same (orders, scenarios, initial state)
always reproduces byte-identical scores.
"""

from __future__ import annotations

import numpy as np

from .config import GAME_WEEKS, N_DECISIONS
from .simulator import simulate


def _coerce_panel(
    orders: np.ndarray, scenarios: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and shape-check one order matrix against a scenario panel."""
    orders_arr = np.asarray(orders, dtype=np.float64)
    sc = np.asarray(scenarios, dtype=np.float64)
    if orders_arr.ndim != 2 or orders_arr.shape[0] != N_DECISIONS:
        raise ValueError(
            f"orders must be (decisions, series) = ({N_DECISIONS}, n), got {orders_arr.shape}"
        )
    n_series = orders_arr.shape[1]
    if sc.ndim != 3:
        raise ValueError(
            f"scenarios must be (n_scenarios, series, weeks) = (S, {n_series}, {GAME_WEEKS}), "
            f"got {sc.shape}"
        )
    if sc.shape[1] != n_series or sc.shape[2] != GAME_WEEKS:
        raise ValueError(
            f"scenarios must be (S, {n_series}, {GAME_WEEKS}), got {sc.shape}"
        )
    if sc.shape[0] == 0:
        raise ValueError("scenarios must hold at least one scenario (mean undefined)")
    return orders_arr, sc


def mc_score(
    orders: np.ndarray,
    scenarios: np.ndarray,
    on_hand0: np.ndarray,
    pipe_w1: np.ndarray,
    pipe_w2: np.ndarray,
) -> float:
    """Mean scored cost (weeks 3-8) of ``orders`` over ``scenarios``.

    Each scenario ``scenarios[s]`` of shape (n_series, 8) is transposed
    to the (8, n_series) demand panel ``simulate()`` expects; the score
    is the arithmetic mean of its ``scored_cost``. ``orders`` is
    (6, n_series); the initial state is shared across scenarios. The
    result is always finite for finite inputs, however large the
    orders: stress (huge) orders merely inflate the holding branch of
    the existing simulator.
    """
    orders_arr, sc = _coerce_panel(orders, scenarios)
    costs = [
        simulate(orders_arr, sc[s].T, on_hand0, pipe_w1, pipe_w2).scored_cost
        for s in range(sc.shape[0])
    ]
    return float(np.mean(costs))


def best_of(
    candidates,
    scenarios: np.ndarray,
    on_hand0: np.ndarray,
    pipe_w1: np.ndarray,
    pipe_w2: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Best (lowest expected cost) candidate and its score.

    Iterates ``candidates`` in order, scores each with ``mc_score`` and
    returns ``(best_orders, best_score)``. Ties keep the earliest
    candidate. An empty candidate list raises ``ValueError``.
    """
    cands = list(candidates)
    if not cands:
        raise ValueError("candidates must hold at least one order matrix")
    best_orders: np.ndarray | None = None
    best_score = np.inf
    for cand in cands:
        score = mc_score(cand, scenarios, on_hand0, pipe_w1, pipe_w2)
        if score < best_score:
            best_score = score
            best_orders = np.asarray(cand, dtype=np.float64)
    assert best_orders is not None  # cands non-empty, so the loop ran
    return best_orders, float(best_score)
