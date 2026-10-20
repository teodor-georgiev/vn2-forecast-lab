"""Adaptive inventory policy for VN2 arena (round 3).

Multi-span EWMA point forecasts (spans 4, 13, 52 weeks) with a yearly
seasonal blend, plus a Poisson-style per-horizon residual-quantile safety
term (buffer * sqrt(forecast)) chosen by backtest. Static (censored) history
is corrected by imputing stockout weeks with the average of available weeks in
the same week-of-year. Revealed in-game demand is appended to the history so
forecasts update as feedback arrives.
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "..", "data")

_SPANS = (4, 13, 52)
_SW = 0.2
_PROJ = 1.2
_BUF = 0.5

_S = None
_A = None
_SH = None
_N0 = None


def _correct(S, A):
    n, T = S.shape
    Y = S.copy()
    for i in range(n):
        for t in range(T):
            if A[i, t] == 0:
                mask = (np.arange(T) % 52 == t % 52) & (A[i] == 1)
                if mask.any():
                    Y[i, t] = S[i, mask].mean()
                else:
                    av = S[i, A[i] == 1]
                    Y[i, t] = av.mean() if av.size else 0.0
    return Y


def _load():
    global _S, _A, _SH, _N0
    try:
        initial = pd.read_csv(os.path.join(_DATA_DIR, "initial_state.csv"))
        sales = pd.read_csv(os.path.join(_DATA_DIR, "sales.csv"))
        avail = pd.read_csv(os.path.join(_DATA_DIR, "availability.csv"))
    except Exception:
        return
    ids = initial["series_id"].tolist()
    wks = sorted(sales["week"].unique())
    n = len(ids)
    T = len(wks)
    S = np.zeros((n, T), dtype=float)
    A = np.zeros((n, T), dtype=float)
    id2i = {sid: i for i, sid in enumerate(ids)}
    sp = sales.pivot(index="series_id", columns="week", values="y")
    sp = sp.reindex(index=ids, columns=wks)
    S = sp.to_numpy(dtype=float)
    ap = avail.pivot(index="series_id", columns="week", values="is_available")
    ap = ap.reindex(index=ids, columns=wks)
    A = ap.to_numpy(dtype=float)
    A = (A > 0).astype(float)  # True/False -> 1/0
    _S = S
    _A = A
    _SH = _correct(S, A)
    _N0 = n


_load()


def decide(week_index, state, revealed):
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    revealed = np.asarray(revealed, dtype=float)
    if revealed.size == 0:
        rev = np.zeros((0, n), dtype=float)
    else:
        rev = revealed.reshape(-1, n)

    # Build extended (corrected static + revealed) history when series match.
    if _SH is not None and n == _N0:
        hist = _SH.copy()
        if rev.shape[0] > 0:
            hist = np.column_stack([hist, rev.T])
    else:
        if rev.shape[0] > 0:
            hist = rev.T.copy()
        else:
            hist = np.zeros((n, 1), dtype=float)

    # Multi-span EWMA level.
    levels = []
    for sp in _SPANS:
        alpha = 2.0 / (sp + 1)
        level = hist[:, 0].copy()
        for t in range(1, hist.shape[1]):
            level = alpha * hist[:, t] + (1.0 - alpha) * level
        levels.append(level)
    base = np.median(np.stack(levels, axis=0), axis=0)

    T = hist.shape[1]
    fc = np.zeros((3, n), dtype=float)
    for hi in range(3):
        idx = T - 52 + hi
        if 0 <= idx < T:
            seas = hist[:, idx]
        else:
            seas = base
        fc[hi] = (1.0 - _SW) * base + _SW * seas
    fc = np.maximum(fc, 0.0)
    a = fc[0]
    b = fc[1]
    c = fc[2]

    on_hand, arr1, arr2 = state.T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - _PROJ * (a + b))
    safety = _BUF * np.sqrt(np.maximum(c, 0.0))
    order = c + safety - projected
    order = np.clip(order, 0.0, None)
    return order
