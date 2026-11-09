"""Causal stock-out imputation.

Out-of-stock weeks are censored demand: the week's demand is unobserved, not
zero. ``impute_panel`` replaces a censored week with the series' most recent
observed (in-stock) sale, using only columns ``[0:upto)`` so a decision-time
call never reads the future. The returned panel is an imputation, not observed
demand: callers must keep the ``in_stock`` mask alongside it and never treat
imputed cells as ground truth.

Guarantees (for inputs honouring the data contract: finite, non-negative
``history``; boolean ``in_stock``):

* output shape equals ``history`` shape; columns ``[upto:]`` are passed
  through unchanged (they are beyond the decision time and are never read
  for filling);
* columns ``[0:upto)`` are finite and non-negative; observed weeks keep
  their observed value;
* a series with no observed week in the window (all-censored, say) falls
  back to the cross-series median of the per-series censored means (mean of
  observed weeks), so leading censored cells of partially-observed series
  also resolve to that same finite value.

Edge cases: ``upto == 0`` (empty window) leaves the panel unchanged, and if
no series has any observed week the median fallback is ``0.0`` - the output
is still finite. Non-finite or negative history values inside the window
are treated as unobserved, so they can never propagate a negative or
non-finite fill downstream.
"""

from __future__ import annotations

import operator

import numpy as np


def impute_panel(history: np.ndarray, in_stock: np.ndarray, upto: int) -> np.ndarray:
    """Forward-fill censored weeks in ``history[:, :upto]`` causally.

    ``history`` is (n_series, T) observed weekly sales and ``in_stock`` the
    aligned boolean availability mask (False = demand censored, never zero).
    Only columns ``[0:upto)`` are read for filling; the same-shape result
    passes columns ``[upto:]`` through unchanged.
    """
    hist = np.array(history, dtype=np.float64, copy=True)
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
    if upto == 0 or n_series == 0:
        # No window to impute from (t=0 edge): return the panel unchanged.
        return hist

    hist_w = hist[:, :upto]
    stock_w = stock[:, :upto]

    # Observed = in stock, finite, and non-negative. Negative or
    # non-finite history values are unobserved demand: they must not
    # become an observed zero (via clipping) and propagate via LOCF.
    observed = stock_w & np.isfinite(hist_w) & (hist_w >= 0)
    values = np.where(observed, hist_w, np.nan)

    # Cross-series median of censored means (per-series observed means).
    counts = observed.sum(axis=1)
    totals = np.where(observed, np.nan_to_num(values), 0.0).sum(axis=1)
    means = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    finite_means = means[np.isfinite(means)]
    fallback = float(np.median(finite_means)) if finite_means.size else 0.0
    fallback = max(fallback, 0.0)

    # Per-series forward-fill: each cell takes the latest observed value at
    # or before it within the window; cells with no prior observed value
    # (leading censored, all-censored series) take the fallback.
    cols = np.arange(upto)[None, :]
    last = np.maximum.accumulate(np.where(observed, cols, -1), axis=1)
    last_vals = np.take_along_axis(values, np.clip(last, 0, None), axis=1)
    hist[:, :upto] = np.where(last >= 0, last_vals, fallback)
    return hist
