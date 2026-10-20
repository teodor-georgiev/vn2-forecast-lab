import numpy as np
import pandas as pd
from pathlib import Path

# Load historical data once at import (read-only).
_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[2] / "data"

_BASE_Y = None
_BASE_A = None
_BASE_N = 0

try:
    _sales = pd.read_csv(_DATA / "sales.csv", parse_dates=["week"])
    _avail = pd.read_csv(_DATA / "availability.csv", parse_dates=["week"])
    _Y = _sales.pivot(index="series_id", columns="week", values="y")
    _A = _avail.pivot(index="series_id", columns="week", values="is_available")
    _BASE_Y = _Y.values.astype(float)
    _BASE_A = _A.values.astype(float)
    _BASE_N = _BASE_Y.shape[0]
except Exception:
    _BASE_Y = np.zeros((0, 0))
    _BASE_A = np.zeros((0, 0))
    _BASE_N = 0


def decide(week_index, state, revealed):
    state = np.asarray(state, dtype=float)
    revealed = np.asarray(revealed, dtype=float)
    n = state.shape[0]

    # Build extended history (base + revealed game weeks).
    if n == _BASE_N and _BASE_N > 0:
        hist = _BASE_Y.copy()
        av = _BASE_A.copy()
    else:
        hist = np.zeros((n, 0), dtype=float)
        av = np.ones((n, 0), dtype=float)

    if revealed.size:
        r = revealed.T  # (n, w)
        hist = np.column_stack([hist, r]) if hist.size else r.copy()
        av = (
            np.column_stack([av, np.ones((n, revealed.shape[0]))])
            if av.size
            else np.ones((n, revealed.shape[0]))
        )

    # 13-week mean, skipping censored (unavailable) weeks.
    k = 13
    if hist.shape[1] > 0:
        window = hist[:, -k:]
        window_av = av[:, -k:] if av.shape[1] >= hist.shape[1] else np.ones_like(window)
        # align availability to window
        if window_av.shape[1] != window.shape[1]:
            window_av = np.ones_like(window)
        w = np.where(window_av > 0.5, window, np.nan)
        m = np.nanmean(w, axis=1)
    else:
        m = np.zeros(n)

    # series mean fallback for all-nan
    if _BASE_N > 0 and n == _BASE_N:
        sm = np.nanmean(np.where(_BASE_A > 0.5, _BASE_Y, np.nan), axis=1)
        m = np.where(np.isnan(m), sm, m)
    m = np.where(np.isnan(m), 0.0, m)
    m = np.maximum(m, 0.0)

    # Forecast for h1, h2, h3 (same mean, robust).
    fc = np.tile(m, (3, 1))

    # Ordering: reference deficit rule defaults.
    scale = 1.0
    projection = 1.0
    buffer = 0.75
    a, b, c = [np.maximum(0.0, f) * scale for f in fc]
    on_hand, arr1, arr2 = state.T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0.0)) - projected
    out = np.clip(deficit, 0.0, None)
    return np.asarray(out, dtype=float).ravel()
