"""Causal demand forecasting for VN2.

Two independent forecast sources are provided so the policy layer can blend
them:

* ``seasonal_forecast`` — the organizer-style deseasonalised 13/26-week moving
  average re-seasonalised onto each target week. Causal, parameter-free, and
  already known to reproduce the published benchmark.
* ``lgbm_forecast`` — a direct three-horizon LightGBM model on causal features.

Every function takes an explicit cutoff ``t`` and may only read columns
``[0:t]``. `tests/test_forecast_causality.py` enforces this by mutating all
columns at and after the cutoff and requiring byte-identical output.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor

from .organizer import global_season, nanmean_rows, series_scale

LAGS = (1, 2, 3, 4, 8, 13, 26, 52)
WINDOWS = (4, 8, 13, 26, 52)
EWMA_SPANS = (4, 8, 13, 26)
HORIZONS = (1, 2, 3)


def censored(y_raw: np.ndarray, in_stock: np.ndarray) -> np.ndarray:
    """Sales with out-of-stock weeks removed rather than counted as zero."""
    return np.where(in_stock, y_raw, np.nan)


def _causal_level(block: np.ndarray) -> np.ndarray:
    """Per-series mean of a trailing block, falling back to wider history."""
    m = nanmean_rows(block)
    return np.nan_to_num(m, nan=0.0)


def seasonal_forecast(
    y: np.ndarray, woy: np.ndarray, t: int, window: int = 13
) -> np.ndarray:
    """Deseasonalised trailing mean re-seasonalised onto horizons 1..3.

    ``y`` is the already-censored panel. Reads only columns ``[0:t]``.
    """
    season = global_season(y, woy, t)
    start = max(0, t - window)
    ratio = season[woy[start:t]][None, :]
    base = _causal_level(y[:, start:t] / ratio)
    base = np.nan_to_num(np.where(np.isnan(base), series_scale(y, t), base))
    target_weeks = woy[t : t + 3]
    if len(target_weeks) < 3:  # extrapolate past the end of the calendar
        pad = np.repeat(target_weeks[-1:], 3 - len(target_weeks))
        target_weeks = np.concatenate([target_weeks, pad])
    return base[:, None] * season[target_weeks][None, :]


def make_features(y: np.ndarray, in_stock: np.ndarray, t: int) -> np.ndarray:
    """Causal per-series feature matrix using only columns ``[0:t]``."""
    n = y.shape[0]
    obs = censored(y[:, :t], in_stock[:, :t]) if in_stock is not None else y[:, :t]
    cols: list[np.ndarray] = [np.full(n, float(t))]
    last = _causal_level(obs[:, max(0, t - 1) : t])
    cols.append(last)

    for lag in LAGS:
        cols.append(
            _causal_level(obs[:, max(0, t - lag - 1) : t - lag])
            if t > lag
            else np.zeros(n)
        )

    for w in WINDOWS:
        block = obs[:, max(0, t - w) : t]
        mu = _causal_level(block)
        sd = np.nan_to_num(
            np.nanstd(np.where(np.isfinite(block), block, np.nan), axis=1), nan=0.0
        )
        cols.extend([mu, sd])

    for span in EWMA_SPANS:
        block = obs[:, max(0, t - 4 * span) : t]
        if block.shape[1] == 0:
            cols.append(np.zeros(n))
            continue
        alpha = 2.0 / (span + 1.0)
        weights = (1.0 - alpha) ** np.arange(block.shape[1] - 1, -1, -1)
        weights = weights / weights.sum()
        vals = np.nan_to_num(np.where(np.isfinite(block), block, 0.0), nan=0.0)
        cols.append(vals @ weights)

    recent = _causal_level(obs[:, max(0, t - 4) : t])
    older = _causal_level(obs[:, max(0, t - 13) : t])
    cols.append(recent - older)
    denom = np.where(older > 0, older, 1.0)
    cols.append((recent - older) / denom)

    # Stockout rate over the observable history: high rates mean censored sales
    # sit far below latent demand, which the model must learn to correct for.
    if in_stock is not None:
        cols.append(1.0 - in_stock[:, :t].mean(axis=1))

    out = np.column_stack(cols).astype(np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _targets(y_censored: np.ndarray, t: int) -> np.ndarray:
    """Mean demand over each of the next three single weeks, from columns < t."""
    n = y_censored.shape[0]
    out = np.zeros((n, 3), dtype=np.float64)
    for h in range(1, 4):
        col = y_censored[:, t - h]
        fallback = series_scale(y_censored, t)
        out[:, h - 1] = np.nan_to_num(
            np.where(np.isfinite(col), col, fallback), nan=0.0
        )
    return out


def fit_lgbm(y: np.ndarray, in_stock: np.ndarray, t: int, train_from: int = 60):
    """Fit three direct single-week-ahead heads on columns ``[train_from:t]``."""
    obs = censored(y[:, :t], in_stock[:, :t])
    rows, labels = [], []
    for cutoff in range(max(train_from, 60), t):
        rows.append(make_features(y[:, :cutoff], in_stock[:, :cutoff], cutoff))
        labels.append(_targets(obs, cutoff))
    x = np.vstack(rows)
    lab = np.stack(labels)  # (n_rows, n_series, 3)
    models = []
    for h in range(3):
        m = LGBMRegressor(
            objective="poisson",
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=23,
            min_child_samples=120,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=6.0,
            random_state=0,
            n_jobs=4,
            verbosity=-1,
        )
        m.fit(x, lab[:, :, h].ravel())
        models.append(m)
    return models


def lgbm_forecast(y: np.ndarray, in_stock: np.ndarray, t: int, models) -> np.ndarray:
    """Direct h=1..3 single-week forecasts, summed to per-horizon demand."""
    x = make_features(y[:, :t], in_stock[:, :t], t)
    single = np.column_stack([np.maximum(m.predict(x), 0.0) for m in models])
    # Cumulative demand over the horizon: h1, h1+h2, h1+h2+h3.
    return np.cumsum(single, axis=1)


def blend(
    seasonal_fc: np.ndarray, lgbm_fc: np.ndarray, seasonal_weight: float
) -> np.ndarray:
    """Convex combination of the two forecast sources."""
    return seasonal_weight * seasonal_fc + (1.0 - seasonal_weight) * lgbm_fc
