"""Causal ABC/XYZ segmentation of the VN2 series panel.

Each series is classified along two axes estimated only from columns
``[0:upto)`` of ``history`` / ``in_stock`` (stockout weeks are censored
demand, never observed zeros, so they never enter either statistic):

* **ABC (importance)** — cumulative observed demand over the window,
  ranked descending. ``A`` is the smallest set of series covering the
  top 80% of total cumulative demand, ``B`` the next set up to 95%,
  ``C`` the remainder. The ``master`` hierarchy is accepted read-only:
  only its row count is used, to enforce index alignment with the
  panel; it is never mutated.
* **XYZ (variability)** — coefficient of variation (std / mean) of
  observed weekly sales, split into terciles across series: ``X``
  stable, ``Y`` moderate, ``Z`` volatile. A series with fewer than two
  observed weeks, or a non-positive mean, has no meaningful CV and
  lands in ``Y``.

Each of the nine combinations maps to a fixed ``(coverage, buffer_coef)``
pair (coverage in weeks of forecast cover; buffer coefficient scales
the safety-stock term). The table is monotone: coverage and buffer
both decrease from AX to CZ, so high-value, stable lines get the most
cover and the largest buffers, and every coverage value lies in the
published band ``[0.5, 4.0]``.
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd

#: Fixed (coverage, buffer_coef) per ABC/XYZ combination.
PARAM_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    ("A", "X"): (4.0, 1.6),
    ("A", "Y"): (3.5, 1.3),
    ("A", "Z"): (3.0, 1.1),
    ("B", "X"): (2.5, 1.2),
    ("B", "Y"): (2.0, 1.0),
    ("B", "Z"): (1.5, 0.8),
    ("C", "X"): (1.0, 0.7),
    ("C", "Y"): (0.75, 0.6),
    ("C", "Z"): (0.5, 0.5),
}

A_SHARE = 0.80  # A = smallest set covering this share of cumulative demand
B_SHARE = 0.95  # B = next set up to this share; C = remainder

COVERAGE_MIN = 0.5
COVERAGE_MAX = 4.0


def _validate(history, in_stock, upto, master) -> tuple[np.ndarray, np.ndarray, int]:
    hist = np.asarray(history, dtype=np.float64)
    stock = np.asarray(in_stock, dtype=bool)
    if hist.ndim != 2 or stock.shape != hist.shape:
        raise ValueError(
            f"shape mismatch: history {hist.shape} vs in_stock {stock.shape}"
        )
    try:
        upto = operator.index(upto)
    except TypeError:
        if isinstance(upto, float) and upto.is_integer():
            upto = int(upto)
        else:
            raise TypeError(f"upto must be an integer, got {upto!r}") from None
    n_series, n_weeks = hist.shape
    if not 0 <= upto <= n_weeks:
        raise ValueError(f"upto={upto} outside [0, {n_weeks}]")
    if master is not None and len(master) != n_series:
        raise ValueError(
            f"master has {len(master)} rows but the panel has {n_series} series"
        )
    return hist, stock, upto


def _observed(history: np.ndarray, in_stock: np.ndarray, upto: int) -> np.ndarray:
    """Observed-demand panel over ``[0:upto)`` with NaN where censored."""
    valid = (
        in_stock[:, :upto] & np.isfinite(history[:, :upto]) & (history[:, :upto] >= 0)
    )
    return np.where(valid, history[:, :upto], np.nan)


def _cum_demand(history: np.ndarray, in_stock: np.ndarray, upto: int) -> np.ndarray:
    """Per-series cumulative observed demand over ``[0:upto)``."""
    obs = _observed(history, in_stock, upto)
    return np.nansum(obs, axis=1)


def _cv(history: np.ndarray, in_stock: np.ndarray, upto: int) -> np.ndarray:
    """Per-series coefficient of variation of observed weekly sales.

    NaN where undefined (fewer than two observed weeks, or non-positive
    mean). Uses the population std (ddof=0) so a single observation
    would give 0 rather than NaN; the n>=2 rule still gates it.
    """
    obs = _observed(history, in_stock, upto)
    counts = np.isfinite(obs).sum(axis=1).astype(np.float64)
    sums = np.nansum(obs, axis=1)
    sumsq = np.nansum(obs * np.where(np.isfinite(obs), obs, 0.0), axis=1)
    safe = np.maximum(counts, 1.0)
    mean = sums / safe
    var = np.maximum(sumsq / safe - mean * mean, 0.0)
    safe_mean = np.where(mean > 0, mean, 1.0)
    cv = np.where((counts >= 2) & (mean > 0), np.sqrt(var) / safe_mean, np.nan)
    return cv


def abc_classes(
    history: np.ndarray,
    in_stock: np.ndarray,
    upto: int,
    master: pd.DataFrame | None = None,
) -> np.ndarray:
    """ABC class per series from cumulative demand over ``[0:upto)``.

    Returns a ``(n_series,)`` unicode array of ``"A"`` / ``"B"`` / ``"C"``.
    A covers the top 80% of total cumulative demand, B the next 15%
    (up to 95%), C the remainder. An all-zero window assigns every
    series to ``C`` (no demand evidence for a higher class).
    """
    hist, stock, upto = _validate(history, in_stock, upto, master)
    n_series = hist.shape[0]
    classes = np.full(n_series, "C", dtype="<U1")
    if n_series == 0 or upto == 0:
        return classes
    cum = _cum_demand(hist, stock, upto)
    total = float(cum.sum())
    if total <= 0:
        return classes
    ranked = np.argsort(-cum, kind="stable")
    share_before = 0.0
    for i in ranked:
        if share_before >= B_SHARE:
            break
        classes[i] = "A" if share_before < A_SHARE else "B"
        share_before += float(cum[i]) / total
    return classes


def xyz_classes(
    history: np.ndarray,
    in_stock: np.ndarray,
    upto: int,
    master: pd.DataFrame | None = None,
) -> np.ndarray:
    """XYZ class per series from CV terciles over ``[0:upto)``.

    Returns a ``(n_series,)`` unicode array of ``"X"`` / ``"Y"`` / ``"Z"``.
    Tercile cut points are the 1/3 and 2/3 quantiles of the finite CVs
    in the window; series with undefined CV land in ``Y``. When fewer
    than two finite CVs exist, every defined series is ``X`` (no
    variability evidence to justify a wider buffer).
    """
    hist, stock, upto = _validate(history, in_stock, upto, master)
    n_series = hist.shape[0]
    classes = np.full(n_series, "Y", dtype="<U1")
    if n_series == 0 or upto == 0:
        return classes
    cv = _cv(hist, stock, upto)
    finite = np.isfinite(cv)
    if finite.sum() == 0:
        return classes
    if finite.sum() < 2:
        classes[finite] = "X"
        return classes
    q_lo, q_hi = np.quantile(cv[finite], [1.0 / 3.0, 2.0 / 3.0])
    classes[finite & (cv <= q_lo)] = "X"
    classes[finite & (cv >= q_hi)] = "Z"
    return classes


def segment_params(
    master: pd.DataFrame,
    history: np.ndarray,
    in_stock: np.ndarray,
    upto: int,
) -> dict[int, dict[str, float]]:
    """Map each series index to ``{coverage, buffer_coef}`` at ``upto``.

    ``master`` is the index-aligned hierarchy DataFrame (read-only,
    never mutated); ``history`` / ``in_stock`` are the ``(n_series, T)``
    observed-sales panel and availability mask. Only columns
    ``[0:upto)`` are read. Every returned coverage lies in
    ``[0.5, 4.0]`` by construction of ``PARAM_TABLE``.
    """
    hist, stock, upto = _validate(history, in_stock, upto, master)
    abc = abc_classes(hist, stock, upto, master)
    xyz = xyz_classes(hist, stock, upto, master)
    out: dict[int, dict[str, float]] = {}
    for i in range(hist.shape[0]):
        coverage, buffer_coef = PARAM_TABLE[(str(abc[i]), str(xyz[i]))]
        out[i] = {"coverage": float(coverage), "buffer_coef": float(buffer_coef)}
    return out
