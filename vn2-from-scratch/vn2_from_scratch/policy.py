"""Ordering policies.

All policies share one shape: given the three-horizon demand forecast, the
on-hand quantity, and the two committed pipeline slots, return a non-negative
integer order.

`deficit` is the parameterisation the CV tunes. With ``coverage=1.0`` and
``buffer_coef=0.0`` it is the textbook base-stock rule: order exactly the
lead-time demand the pipeline does not already cover. Raising ``coverage``
buys service at the cost of holding, and ``buffer_coef`` adds a square-root
safety term that scales with demand magnitude rather than level, which suits
these intermittent series.
"""

from __future__ import annotations

import numpy as np


def deficit_order(
    fc: np.ndarray,
    on_hand: np.ndarray,
    arrivals_now: np.ndarray,
    arrivals_next: np.ndarray,
    coverage: float = 1.0,
    buffer_coef: float = 0.0,
) -> np.ndarray:
    """Order up to ``coverage`` weeks of lead-time demand plus a safety buffer.

    ``fc`` is ``(n, 3)`` cumulative demand: column 0 is demand over the next
    week, column 1 over the next two, column 2 over the next three. The order
    placed now does not arrive until the third of those weeks, so the relevant
    target is column 2, net of supply already committed to the first two.
    """
    fc = np.asarray(fc, dtype=np.float64)
    lead_demand = np.maximum(fc[:, 2], 0.0)
    position = on_hand + arrivals_now + arrivals_next
    target = coverage * lead_demand + buffer_coef * np.sqrt(lead_demand)
    return np.round(np.clip(target - position, 0.0, None))


def organizer_order(
    fc: np.ndarray, on_hand: np.ndarray, a_now: np.ndarray, a_next: np.ndarray
):
    """Reference-only alias so the benchmark can be expressed in this interface."""
    raise NotImplementedError(
        "The organizer benchmark has its own implementation in organizer.py; "
        "it needs the seasonality index, which this interface does not carry."
    )


def naive_zero(fc, on_hand, a_now, a_next, **_):
    """Order nothing. Used as the CV floor and as a leakage canary."""
    return np.zeros_like(on_hand, dtype=np.float64)
