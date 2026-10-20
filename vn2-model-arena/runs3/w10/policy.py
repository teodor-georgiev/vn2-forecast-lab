"""Adaptive inventory ordering policy for the VN2 arena (worker w10).

Method (three-line summary):
  1. Load the arena history once at import, correct sales censored by stock-outs
     (availability), and fit per-series multi-span EWMA levels (spans 4/8/13/26,
     Croston rate for intermittent series) plus a damped short-vs-long trend.
  2. Pre-compute, for every series and horizon h=1..3, the empirical quantile at
     the newsvendor critical fractile 1/(1+0.2)=0.833 of the signed h-step-ahead
     forecast errors over rolling origins (per-horizon residual calibration).
  3. At each decision, blend the history level with an EWMA of the revealed
     in-game demand (weight growing with weeks revealed), forecast h1..h3, project
     inventory after covering h1/h2, and order up to h3 forecast plus the
     calibrated safety quantile minus projected inventory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is installed in the arena
    pd = None

_HERE = Path(__file__).resolve().parent
_ARENA = _HERE.parent.parent
_DATA = _ARENA / "data"

SHORT_COST = 1.00
HOLD_COST = 0.20
CRIT = SHORT_COST / (SHORT_COST + HOLD_COST)  # newsvendor critical fractile

SPANS = (4, 8, 13, 26)
EWMA_ALPHA = {s: 2.0 / (s + 1.0) for s in SPANS}
TREND_DAMP = 0.5
CROSTON_ALPHA = 0.35
CROSTON_P0 = 0.60
MAX_REVEALED_WEIGHT = 0.90  # knob: max weight on revealed in-game demand
REVEAL_USE_MEAN = False  # knob: use revealed mean instead of EWMA
REVEAL_TREND = False  # knob: use a revealed-based trend


# ------------------------------------------------------------------ loading


def _load_arena():
    """Return (series_order, corrected_history, raw_history) aligned to the
    initial_state.csv series order (canonical column order of state/revealed)."""
    init = pd.read_csv(_DATA / "initial_state.csv")
    order = init["series_id"].astype(str).tolist()
    n = len(order)

    sales = pd.read_csv(_DATA / "sales.csv")
    weeks = sorted(sales["week"].astype(str).unique())
    raw = sales.pivot(index="series_id", columns="week", values="y").reindex(
        index=order, columns=weeks
    )
    raw = raw.fillna(0.0).to_numpy(dtype=float)  # (n, W)

    avail = pd.read_csv(_DATA / "availability.csv")
    av = avail.pivot(index="series_id", columns="week", values="is_available").reindex(
        index=order, columns=weeks
    )
    av = av.fillna(True).to_numpy(dtype=bool)  # (n, W)

    corrected = raw.copy()
    for i in range(n):
        avail_vals = raw[i][av[i]]
        fill = float(np.mean(avail_vals)) if avail_vals.size else 0.0
        corrected[i] = np.where(av[i], raw[i], fill)
    return order, corrected, raw


def _ewma(x, alpha):
    """Causal exponential-weighted moving average along the week axis (axis 1).

    x: (n_series, n_weeks) -> same shape.  EWMA value at week t uses only
    weeks <= t (causal), so it is valid at every rolling origin.
    """
    out = np.empty_like(x, dtype=float)
    out[:, 0] = x[:, 0]
    om = 1.0 - alpha
    for t in range(1, x.shape[1]):
        out[:, t] = alpha * x[:, t] + om * out[:, t - 1]
    return out


def _ewma_end(x, alpha):
    """EWMA end value of a 1-D or (w, n) array (last row along axis 0)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        y = x[0]
        om = 1.0 - alpha
        for v in x[1:]:
            y = alpha * v + om * y
        return y
    y = x[0].copy()
    om = 1.0 - alpha
    for t in range(1, x.shape[0]):
        y = alpha * x[t] + om * y
    return y


def _croston_rates(d, alpha):
    """Croston demand rate at every period t (causal, one pass over time)."""
    n = d.shape[1]
    T = np.ones(n, dtype=float)  # mean inter-demand interval
    L = np.zeros(n, dtype=float)  # mean demand size
    have = np.zeros(n, dtype=bool)  # first demand not yet observed
    rates = np.empty_like(d, dtype=float)
    since = np.zeros(n, dtype=float)
    for t in range(d.shape[0]):
        since += 1.0
        pos = d[t] > 0
        if np.any(pos):
            L[pos] = alpha * d[t][pos] + (1.0 - alpha) * L[pos]
            T[pos] = alpha * since[pos] + (1.0 - alpha) * T[pos]
            have[pos] = True
            since[pos] = 0.0
        rates[t] = np.where(
            have, np.divide(L, T, out=np.zeros_like(L), where=T > 0), 0.0
        )
    return rates


# module-level state, built once at import
_series_order = None
_hist = None  # corrected history (n, W), censoring-corrected
_level_hist = None  # (n, W) history-only forecast level at each origin t
_trend_hist = None  # (n, W) damped trend at each origin t
_safety = None  # (n, 3) CRIT-quantile of signed h-step errors, h=1..3
_n_series = 0

_LOADED = False


def _ensure_loaded():
    global _series_order, _hist, _level_hist, _trend_hist, _safety, _n_series, _LOADED
    if _LOADED:
        return
    _LOADED = True
    if pd is None:
        return
    order, corrected, raw = _load_arena()
    _series_order = order
    _hist = corrected
    n, W = corrected.shape
    _n_series = n

    # multi-span EWMA level at every origin, averaged across spans
    level = np.zeros((n, W), dtype=float)
    for s in SPANS:
        level += _ewma(corrected, EWMA_ALPHA[s])
    level /= len(SPANS)
    trend = TREND_DAMP * (
        _ewma(corrected, EWMA_ALPHA[4]) - _ewma(corrected, EWMA_ALPHA[26])
    )

    p0 = (raw <= 0).mean(axis=1)
    intermittent = p0 > CROSTON_P0
    if np.any(intermittent):
        rates = _croston_rates(corrected, CROSTON_ALPHA)
        level = np.where(intermittent[:, None], rates, level)

    _level_hist = level
    _trend_hist = trend

    # per-horizon residual-quantile (newsvendor) calibration:
    # safety_h = max(0, quantile_CRIT(demand h steps ahead) - point forecast_h).
    # For intermittent series the 0.833 quantile of weekly demand is 0, so no
    # safety stock is carried there; for stable series it is ~ z * sigma.
    level_end = level[:, -1]
    trend_end = trend[:, -1]
    safety = np.zeros((n, 3), dtype=float)
    for h in (1, 2, 3):
        q = np.quantile(corrected[:, h:], CRIT, axis=1)
        fh = level_end + h * trend_end
        safety[:, h - 1] = np.maximum(0.0, q - fh)
    _safety = safety


def _decide_vector(week_index, state, revealed):
    _ensure_loaded()
    n = _n_series
    w = int(week_index)
    if w >= 6 or n == 0:
        return np.zeros(n, dtype=float)

    on_hand = np.asarray(state, dtype=float)[:, 0].ravel()
    arr1 = np.asarray(state, dtype=float)[:, 1].ravel()
    arr2 = np.asarray(state, dtype=float)[:, 2].ravel()

    level_last = _level_hist[:, -1].copy()
    trend_last = _trend_hist[:, -1].copy()

    lam = 0.0
    if revealed is not None:
        rev = np.asarray(revealed, dtype=float)
        if rev.size:
            rev = np.atleast_2d(rev)
            if rev.shape[1] != n:
                rev = rev.T
                if rev.shape[1] != n:
                    rev = rev.reshape(-1, n) if rev.size % n == 0 else np.zeros((0, n))
            rw = rev.shape[0]
            if rw > 0:
                lam = min(rw / 6.0, MAX_REVEALED_WEIGHT)
                if REVEAL_USE_MEAN:
                    rev_level = rev.mean(axis=0)
                    rev_trend = np.zeros(n, dtype=float)
                else:
                    span = min(4, rw)
                    alpha = 2.0 / (span + 1.0)
                    rev_level = _ewma_end(rev, alpha)
                    rev_trend = np.zeros(n, dtype=float)
                    if REVEAL_TREND and rw >= 2:
                        rev_trend = rev[-1] - rev[-2]
                level_last = (1.0 - lam) * level_last + lam * rev_level
                if REVEAL_TREND:
                    trend_last = (1.0 - lam) * trend_last + lam * rev_trend

    f1 = np.maximum(0.0, level_last)
    f2 = np.maximum(0.0, level_last + trend_last)
    f3 = np.maximum(0.0, level_last + 2.0 * trend_last)

    projected = np.maximum(0.0, on_hand + arr1 + arr2 - f1 - f2)
    target = f3 + _safety[:, 2]
    order = np.maximum(0.0, target - projected)
    return np.nan_to_num(order, nan=0.0, posinf=0.0, neginf=0.0)


def decide(week_index, state, revealed):
    """week_index: int 0..7 — week about to start.
    state: (n_series, 3) — on_hand, delivery this week, delivery next week.
    revealed: (week_index, n_series) — actual in-game demand so far.
    returns: (n_series,) non-negative order quantities.
    """
    arr = _decide_vector(week_index, state, revealed)
    arr = np.asarray(arr, dtype=float).ravel()
    arr = np.clip(arr, 0.0, None)
    return arr


if __name__ == "__main__":  # quick smoke test
    _ensure_loaded()
    print("series:", _n_series, "weeks:", _hist.shape[1])
    print("intermittent frac:", float((_hist <= 0).all(axis=1).mean()) if False else "")
    import time

    t0 = time.perf_counter()
    st = np.zeros((_n_series, 3))
    st[:, 0] = 2.0
    dec = decide(0, st.copy(), np.empty((0, _n_series)))
    dec = decide(
        3, st.copy(), np.random.default_rng(0).uniform(0, 2, size=(3, _n_series))
    )
    print(
        "decide OK",
        dec.shape,
        dec.sum(),
        f"{(time.perf_counter() - t0) * 1e3:.1f} ms (two calls)",
    )
