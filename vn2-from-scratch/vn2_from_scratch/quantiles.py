"""Causal quantile forecasts from seasonal-forecast residuals.

``quantile_forecast`` first evaluates the existing seasonal reference
(``forecast.seasonal_forecast``, shape (n, 3)) at the decision time ``t``,
then adds per-series, per-horizon empirical residual quantiles estimated
from origins strictly before ``t``. For each origin ``k < t`` and each
horizon ``h`` the residual is ``actual[k + h - 1] - forecast_h(k)``,
taken only where that week was observed (in stock, finite, non-negative);
stockout weeks are censored, never zero demand, so they never enter the
residual sample. The output is clipped at 0 and made monotone in the
quantile levels via ``np.maximum.accumulate`` along the tau axis.

Only columns ``[0:t)`` of ``y`` and ``in_stock`` are read. The week-of-year
calendar needed by the seasonal reference is supplied by the caller via
``woy`` when available (e.g. from ``organizer.full_panel``); otherwise a
synthetic calendar (``arange % 52 + 1``) is used. ``impute_panel`` is
invoked once, causally at ``upto == t``, to sanitise the ``[0:t)`` slice
into a finite panel; imputed cells are never used as residual actuals —
residuals come only from originally observed weeks.
"""

from __future__ import annotations

import operator

import numpy as np

from .forecast import seasonal_forecast
from .impute import impute_panel

DEFAULT_TAUS = (0.5, 0.7, 0.85, 0.95)
N_HORIZONS = 3


def _extended_woy(woy: np.ndarray | None, need: int) -> np.ndarray:
    """Return a length-``need`` week-of-year array.

    ``need == t + 3`` covers every seasonal target week referenced by a
    decision at ``t``. A supplied calendar is extended cyclically when
    short; with no calendar a synthetic arange calendar is used.
    """
    if woy is None:
        return np.arange(need) % 52 + 1
    arr = np.asarray(woy).ravel().astype(np.int64)
    if arr.size == 0:
        raise ValueError("woy must be non-empty when supplied")
    if arr.size >= need:
        return arr[:need]
    last = int(arr[-1])
    extra = [(last + i - 1) % 52 + 1 for i in range(1, need - arr.size + 1)]
    return np.concatenate([arr, np.asarray(extra, dtype=np.int64)])


def _censored_panel(
    y_raw: np.ndarray, in_stock: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-sanitised inputs plus the censored panel and observed mask.

    ``impute_panel`` is called causally on the already-sliced ``[0:t)``
    window only. Its imputed values are never read below: the returned
    panel keeps NaN wherever the original week was censored or invalid,
    so forward-filled cells cannot leak into residuals as ground truth.
    """
    _ = impute_panel(y_raw, in_stock, y_raw.shape[1])
    valid = in_stock & np.isfinite(y_raw) & (y_raw >= 0)
    censored = np.where(valid, y_raw, np.nan)
    return censored, valid


def quantile_forecast(
    y: np.ndarray,
    in_stock: np.ndarray,
    t: int,
    taus: tuple[float, ...] = DEFAULT_TAUS,
    woy: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile forecasts for horizons 1..3, shape (n, 3, len(taus)).

    ``y`` is (n, T) observed weekly sales and ``in_stock`` the aligned
    boolean availability mask (False = demand censored, never zero).
    Only columns ``[0:t)`` of both are read. ``taus`` is any non-empty
    sequence of levels in [0, 1]; output columns follow the taus sorted
    in ascending order so the tau axis is monotone by construction.
    ``woy`` optionally supplies the ISO week-of-year calendar.
    """
    try:
        t = operator.index(t)
    except TypeError:
        if isinstance(t, float) and t.is_integer():
            t = int(t)
        else:
            raise TypeError(f"t must be an integer, got {t!r}") from None
    y_arr = np.asarray(y, dtype=np.float64)
    s_arr = np.asarray(in_stock, dtype=bool)
    if y_arr.ndim != 2 or s_arr.shape != y_arr.shape:
        raise ValueError(f"shape mismatch: y {y_arr.shape} vs in_stock {s_arr.shape}")
    n_series, n_weeks = y_arr.shape
    if not 0 <= t <= n_weeks:
        raise ValueError(f"t={t} outside [0, {n_weeks}]")
    taus_arr = np.asarray(tuple(taus), dtype=np.float64).ravel()
    if taus_arr.size == 0 or np.any((taus_arr < 0) | (taus_arr > 1)):
        raise ValueError(f"taus must be non-empty levels in [0, 1], got {taus!r}")
    taus_sorted = np.sort(taus_arr)
    n_taus = taus_sorted.size
    if n_series == 0:
        return np.zeros((0, N_HORIZONS, n_taus), dtype=np.float64)
    if t == 0:
        return np.zeros((n_series, N_HORIZONS, n_taus), dtype=np.float64)

    # Causal slice: everything below reads [0:t) only. Columns at or after
    # t of y / in_stock are never touched again in this function.
    y_t = y_arr[:, :t]
    s_t = s_arr[:, :t]
    censored, _valid = _censored_panel(y_t, s_t)
    woy_ext = _extended_woy(woy, t + N_HORIZONS)

    base = seasonal_forecast(censored, woy_ext, t)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    base = np.clip(base, 0.0, None)

    burn = min(13, max(1, t // 3))
    origins = list(range(burn, t))
    n_orig = len(origins)
    residuals = np.full((n_series, N_HORIZONS, n_orig), np.nan)
    for j, k in enumerate(origins):
        fc = seasonal_forecast(censored, woy_ext, k)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        for h in range(N_HORIZONS):
            idx = k + h
            if idx < t:
                residuals[:, h, j] = censored[:, idx] - fc[:, h]

    offsets = np.zeros((n_series, N_HORIZONS, n_taus), dtype=np.float64)
    for i in range(n_series):
        for h in range(N_HORIZONS):
            vals = residuals[i, h, :]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                offsets[i, h, :] = np.quantile(vals, taus_sorted)

    q = base[:, :, None] + offsets
    q = np.clip(q, 0.0, None)
    q = np.maximum.accumulate(q, axis=2)
    return q
