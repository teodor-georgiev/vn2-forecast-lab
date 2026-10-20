"""VN2 adaptive ordering policy — w15 (rolling-backtest-tuned deficit rule).

Method
------
1. Load the arena's 157-week history from ``data\\sales.csv`` +
   ``availability.csv`` (read-only) and impute the stockout weeks
   (availability == 0, where sales are censored to zero) with the local
   uncensored level, so intermittent stockouts do not depress the demand
   level of high-volume series.
2. Forecast h1 = h2 = h3 = 13-week trailing level of that corrected demand,
   then rescale it by an in-game multiplier: the demand revealed so far this
   game vs. the trailing forecasts of those same weeks
   (mult = (obs + 6*level) / (pred + 6*level), clipped to [0.4, 2.5]).
3. Order with the reference deficit rule
       order = rint(max(0, f - max(0, on_hand + arr1 + arr2 - 2*projection*f)))
   with projection = 1.35 (buffer = 0, scale = 0.95 folded into f).

(scale 0.95, buffer 0.0, projection 1.35, corrected base, adapt 6) was chosen
by rolling weekly backtests with the exact arena flow (tune.py / tune2.py /
tune3.py / tune4.py): it scores mean 4820 over 13 historical cutoffs
(90..149) vs. 4954 for the manager's reference bar rule, and 4220 vs. 4418
on the four cutoffs held out from selection.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))
_LEVEL_WIN = 13
_SCALE = 0.95
_PROJECTION = 1.35
_ADAPT_K = 6.0
_MULT_LO, _MULT_HI = 0.4, 2.5


def _load_history():
    """(n, T) demand matrix with stockout weeks imputed, sorted series order."""
    sales = pd.read_csv(os.path.join(_DATA, "sales.csv"))
    avail = pd.read_csv(os.path.join(_DATA, "availability.csv"))
    weeks = sorted(sales["week"].unique())
    series = sorted(sales["series_id"].unique())
    sidx = {s: i for i, s in enumerate(series)}
    widx = {w: i for i, w in enumerate(weeks)}
    n, t = len(series), len(weeks)
    y = np.zeros((n, t), dtype=float)
    a = np.zeros((n, t), dtype=float)
    y[sales["series_id"].map(sidx).to_numpy(), sales["week"].map(widx).to_numpy()] = (
        sales["y"].to_numpy()
    )
    a[avail["series_id"].map(sidx).to_numpy(), avail["week"].map(widx).to_numpy()] = (
        avail["is_available"].to_numpy()
    )
    valid = a > 0
    v = y * valid
    cs = np.zeros((n, t + 1))
    cc = np.zeros((n, t + 1))
    cs[:, 1:] = np.cumsum(v, axis=1)
    cc[:, 1:] = np.cumsum(valid, axis=1)
    half = 3
    idx = np.arange(t)
    lo = np.clip(idx - half, 0, t)
    hi = np.clip(idx + half, 0, t)
    cnt = cc[:, hi] - cc[:, lo]
    lvl = (cs[:, hi] - cs[:, lo]) / np.maximum(cnt, 1.0)
    return np.where(valid, y, lvl)


_LEVEL = _load_history()
_N, _L0 = _LEVEL.shape


def _mean(avail, upto):
    """Trailing _LEVEL_WIN mean over `avail` up to (exclusive) column `upto`."""
    if upto <= 0:
        return np.zeros(avail.shape[0])
    return np.maximum(avail[:, max(0, upto - _LEVEL_WIN) : upto].mean(axis=1), 0.0)


def decide(week_index, state, revealed):
    """week_index: int 0..7; state (n,3) on_hand/arr_w+1/arr_w+2;
    revealed (week_index, n) actual demand so far this game.
    Returns non-negative order quantities (n,) delivered during week w+2."""
    state = np.asarray(state, dtype=float)
    n = int(state.shape[0])
    w = int(week_index)
    if revealed is None:
        rev = np.zeros((n, 0), dtype=float)
    else:
        rev = np.asarray(revealed, dtype=float).reshape(w, n).T

    if n == _N:  # the real replay: 599 series in the arena's row order
        avail = np.concatenate([_LEVEL, rev], axis=1) if w else _LEVEL
        level_now = _mean(avail, _L0 + w)
        mult = 1.0
        if w > 0:
            pred = np.zeros(n)
            for j in range(w):
                pred += _mean(avail, _L0 + j)
            obs = rev.sum(axis=1)
            m = (obs + _ADAPT_K * level_now) / np.maximum(
                pred + _ADAPT_K * level_now, 1e-9
            )
            mult = np.clip(np.where(np.isfinite(m), m, 1.0), _MULT_LO, _MULT_HI)
        f = np.maximum(0.0, level_now) * _SCALE * mult
    else:  # defensive fallback when the series count is not the arena's 599
        level_now = (
            np.maximum(rev[:, -min(_LEVEL_WIN, w) :].mean(axis=1), 0.0)
            if w
            else np.zeros(n)
        )
        f = np.maximum(0.0, level_now) * _SCALE

    on_hand, arr1, arr2 = state.T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - 2.0 * _PROJECTION * f)
    deficit = f - projected
    return np.rint(np.clip(deficit, 0.0, None))
