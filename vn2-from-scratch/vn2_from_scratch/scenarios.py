"""SIP/SLURP-style scenario packs from quantile forecasts.

Consumes the output of ``quantiles.quantile_forecast`` — array ``q`` of
shape (n_series, 3, n_taus), monotone in tau and non-negative — and
produces demand scenarios of shape (n_scenarios, n_series, 3).

Method (per series, per horizon): the tau-grid points define an empirical
CDF; a uniform draw ``U`` is inverted through the piecewise-constant
step function of those points (inverse CDF): ``U <= tau[0]`` maps to
``q[0]``, ``tau[k-1] < U <= tau[k]`` maps to ``q[k]`` and ``U > tau[-1]``
maps to ``q[-1]`` (flat extrapolation). Draws therefore always equal one
of the supplied grid quantiles -- no interpolation between grid points --
and lie in ``[q_min, q_max]`` for that series/horizon.

Cross-week dependence: exactly **one uniform per series-scenario** is
drawn — shape (n_scenarios, n_series) — and the same ``U`` is inverted
through each horizon's CDF. A high scenario is therefore high at
horizons 1..3 jointly for that series (comonotonic / SLURP coupling),
rather than drawing the horizons independently. Uniforms differ across
series and scenarios, so cross-series variation is preserved.

The function reads no history panel: it is purely a function of the
causal ``quantiles`` input, so it cannot leak post-decision data.
"""

from __future__ import annotations

import operator

import numpy as np

from .quantiles import DEFAULT_TAUS

N_HORIZONS = 3


def _resolve_taus(n_taus: int, taus) -> np.ndarray:
    """Tau grid for ``n_taus`` grid points.

    Explicit ``taus`` (length ``n_taus``) always wins and is validated.
    Without it, the default grid is only defined for the
    ``quantile_forecast`` default: ``n_taus == len(DEFAULT_TAUS)`` uses
    ``DEFAULT_TAUS``, and ``n_taus == 1`` uses ``(0.5,)``. Any other
    ``n_taus`` falls back to the equal-probability grid
    ``1/(k+1), 2/(k+1), ..., k/(k+1)`` -- this is a documented assumption,
    not the caller's true taus, so callers with non-default tau counts
    (e.g. ``quantile_forecast(taus=...)`` with 2 levels) must pass
    ``taus=`` explicitly for an exact grid. ``n_taus == 0`` is rejected.
    """
    if n_taus == 0:
        raise ValueError("quantiles must have n_taus >= 1, got 0")
    if taus is not None:
        arr = np.asarray(tuple(taus), dtype=np.float64).ravel()
        if arr.size != n_taus:
            raise ValueError(
                f"taus length {arr.size} does not match quantiles n_taus {n_taus}"
            )
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"taus must lie in [0, 1], got {taus!r}")
        if np.any(np.diff(arr) <= 0.0):
            raise ValueError(f"taus must be strictly increasing, got {taus!r}")
        return arr
    if n_taus == len(DEFAULT_TAUS):
        return np.asarray(DEFAULT_TAUS, dtype=np.float64)
    if n_taus == 1:
        return np.asarray([0.5], dtype=np.float64)
    return (np.arange(n_taus, dtype=np.float64) + 1.0) / (n_taus + 1.0)


def _invert_cdf(uniforms: np.ndarray, taus: np.ndarray, q_h: np.ndarray) -> np.ndarray:
    """Piecewise-constant CDF inversion (step) for one horizon.

    ``uniforms`` is (n_scenarios, n_series); ``q_h`` is (n_series, n_taus)
    monotone non-decreasing in tau. Returns (n_scenarios, n_series) with
    every draw equal to a grid quantile: ``idx = searchsorted(taus, U,
    side="left")`` picks the first tau ``>= U`` (clamped to the last grid
    point when ``U`` exceeds ``taus[-1]``).
    """
    if q_h.shape[1] == 1:
        return np.broadcast_to(q_h[:, 0][None, :], uniforms.shape).copy()
    idx = np.clip(
        np.searchsorted(taus, uniforms, side="left"), 0, taus.size - 1
    )  # (S, N)
    series_idx = np.arange(q_h.shape[0])[None, :]
    return q_h[series_idx, idx].copy()


def pack_scenarios(quantiles, n_scenarios, seed, taus=None) -> np.ndarray:
    """Draw SIP-style scenarios from quantile forecasts.

    ``quantiles`` is (n_series, 3, n_taus) as returned by
    ``quantile_forecast``. Returns (n_scenarios, n_series, 3), finite and
    non-negative. ``seed`` seeds ``numpy.random.default_rng`` so the same
    seed reproduces byte-identical draws. ``taus`` optionally supplies the
    tau grid (length ``n_taus``, in [0, 1], strictly increasing); it is
    required for an exact grid whenever ``n_taus`` is not 1 or
    ``len(DEFAULT_TAUS)`` -- otherwise an equal-probability fallback grid
    is assumed, see :func:`_resolve_taus`. Non-finite quantiles are
    rejected with ``ValueError`` rather than silently zeroed.
    """
    try:
        n_scenarios = operator.index(n_scenarios)
    except TypeError:
        if isinstance(n_scenarios, float) and n_scenarios.is_integer():
            n_scenarios = int(n_scenarios)
        else:
            raise TypeError(
                f"n_scenarios must be an integer, got {n_scenarios!r}"
            ) from None
    if n_scenarios < 0:
        raise ValueError(f"n_scenarios must be >= 0, got {n_scenarios}")
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 3 or q.shape[1] != N_HORIZONS:
        raise ValueError(
            f"quantiles must have shape (n_series, 3, n_taus), got {q.shape}"
        )
    n_series, _, n_taus = q.shape
    if n_taus == 0:
        raise ValueError(
            f"quantiles must have shape (n_series, 3, n_taus) with n_taus >= 1, got {q.shape}"
        )
    taus_arr = _resolve_taus(n_taus, taus)
    if not np.all(np.isfinite(q)):
        raise ValueError("quantiles must be finite, got NaN or +/-inf")
    # Sanitise: non-negative, monotone in tau (quantile_forecast output
    # already is; enforce so negative inputs stay non-negative).
    q = np.clip(q, 0.0, None)
    q = np.maximum.accumulate(q, axis=2)
    if n_scenarios == 0 or n_series == 0:
        return np.zeros((n_scenarios, n_series, N_HORIZONS), dtype=np.float64)

    rng = np.random.default_rng(seed)
    # One uniform per series-scenario: shared across horizons (h=1..3),
    # so the horizons co-move for a given series (cross-week dependence).
    uniforms = rng.random((n_scenarios, n_series))
    out = np.empty((n_scenarios, n_series, N_HORIZONS), dtype=np.float64)
    for h in range(N_HORIZONS):
        out[:, :, h] = _invert_cdf(uniforms, taus_arr, q[:, h, :])
    out = np.clip(out, 0.0, None)
    return out
