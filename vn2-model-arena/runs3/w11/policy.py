"""Adaptive VN2 ordering policy (worker w11) — ABC/XYZ segmentation.

Theme: ABC/XYZ segmentation.
  * A (high-volume, stable): level + yearly seasonality forecast, per-series
    seasonal shape shrunk toward the global shape.
  * B (mid-volume): same but heavier shrinkage.
  * C (intermittent): longer-window rate estimate, flat seasonal shape.
All classes order up to a per-class newsvendor quantile of the 3-week
lead-time demand (weeks w, w+1, w+2) using a negative-binomial demand
model whose dispersion is estimated per series and shrunk to its class.

decide(week_index, state, revealed) exactly as TASK3.md specifies.
No file writes; reads only the arena data folder (read-only).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

PARAMS = {
    # level estimation (on de-seasonalised, de-censored demand)
    "win_short": 13,
    "win_long": 52,
    "w_recent": 0.7,
    "min_level": 0.01,
    # yearly seasonality: per-series shape shrunk toward the global shape
    "seas_k": 4.0,  # pseudo-counts of the global shape in the blend
    "flat_C": False,  # C-class series use a flat seasonal shape
    "c_win": 13,  # level window for C-class series
    "seas_cap_lo": 0.4,
    "seas_cap_hi": 3.0,
    # service fractiles per class (cost ratio 1.00 / 0.20)
    "q_A": 0.70,
    "q_B": 0.60,
    "q_C": 0.80,
    # dispersion shrinkage toward the class median
    "phi_shrink": 0.5,
    "phi_cap": 6.0,
    # class thresholds on the series' long-run weekly mean
    "a_thr": 3.0,
    "c_thr": 1.0,
    "decensor": True,
}

_HISTORY_BASE = None  # None -> the game starts right after the known history
_C: dict = {}


# ---------------------------------------------------------------------------
# data preparation (cached once)
# ---------------------------------------------------------------------------
def _arena_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "data" / "sales.csv").exists():
            return parent
    return here.parents[2]


def _decensor(y, a, wnd=11):
    """Replace stockout weeks by a local estimate of the censored demand."""
    n, T = y.shape
    wk = np.arange(T) % 52
    tot = y.sum(axis=0)
    prof = np.array([tot[wk == k].mean() for k in range(52)])
    prof = np.clip(prof / max(prof.mean(), 1e-9), 0.5, 2.5)
    ds = y / prof[wk]
    av, ker = a * ds, np.ones(wnd)
    num = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, av)
    den = np.maximum(
        np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, a), 1.0
    )
    smean = av.sum(1) / np.maximum(a.sum(1), 1.0)
    est = np.where(den > 1.5, num / den, smean[:, None]) * prof[wk]
    return np.where(a > 0.5, y, np.maximum(y, est))


def _profiles(d):
    """Detrended global week-of-year shape and per-series shapes (mean 1)."""
    n, T = d.shape
    idx, blk = np.arange(T) % 52, np.arange(T) // 52
    tot = d.sum(0)
    num, cnt = np.zeros(52), np.zeros(52)
    for b in np.unique(blk):
        m = tot[blk == b].mean()
        if m <= 0:
            continue
        sel = blk == b
        for k in np.unique(idx[sel]):
            num[k] += (tot[sel][idx[sel] == k] / m).mean()
            cnt[k] += 1
    glob = np.where(cnt > 0, num / np.maximum(cnt, 1), 1.0)
    glob = np.clip(glob / glob.mean(), 0.4, 3.0)

    dd = d / glob[idx]
    pnum, pcnt = np.zeros((n, 52)), np.zeros((n, 52))
    for b in np.unique(blk):
        sel = blk == b
        m = np.maximum(dd[:, sel].mean(1, keepdims=True), 1e-9)
        for k in np.unique(idx[sel]):
            pnum[:, k] += (dd[:, sel][:, idx[sel] == k] / m).mean(1)
            pcnt[:, k] += 1
    per = np.where(pcnt > 0, pnum / np.maximum(pcnt, 1), 1.0)
    return glob, np.clip(per, 0.2, 5.0)


def _load():
    if _C:
        return _C
    import pandas as pd

    root = _arena_root()
    sales = pd.read_csv(root / "data" / "sales.csv", parse_dates=["week"])
    avail = pd.read_csv(root / "data" / "availability.csv", parse_dates=["week"])
    Y = (
        sales.pivot(index="series_id", columns="week", values="y")
        .sort_index()
        .values.astype(float)
    )
    A = (
        avail.pivot(index="series_id", columns="week", values="is_available")
        .sort_index()
        .values
    )
    A = (np.asarray(A, dtype=float) > 0.5).astype(float)
    D = _decensor(Y, A) if PARAMS["decensor"] else Y.copy()
    glob, per = _profiles(D)

    mu = D.mean(1)
    phi = np.clip((D.var(1) - mu) / np.maximum(mu * mu, 1e-9), 0.0, PARAMS["phi_cap"])
    med = np.median(mu[mu > 0]) if (mu > 0).any() else 1.0
    hi = mu >= max(2.0, med)
    phi_hi = np.median(phi[hi]) if hi.any() else 1.0
    phi_lo = np.median(phi[~hi]) if (~hi).any() else 1.5
    w = PARAMS["phi_shrink"]
    phi = np.where(hi, w * phi + (1 - w) * phi_hi, w * phi + (1 - w) * phi_lo)

    _C.update(dict(Y=Y, A=A, D=D, glob=glob, per=per, mu=mu, phi=phi, T=D.shape[1]))
    return _C


# ---------------------------------------------------------------------------
# forecast: mean demand for the next three weeks
# ---------------------------------------------------------------------------
def _forecast(w, revealed):
    C = _load()
    p = PARAMS
    base = C["T"] if _HISTORY_BASE is None else int(_HISTORY_BASE)
    base = max(1, min(base, C["T"]))

    d = C["D"][:, :base]
    if w and revealed is not None and len(revealed):
        d = np.column_stack((d, np.asarray(revealed, dtype=float).T))
    T, n = d.shape[1], d.shape[0]

    glob, per = C["glob"], C["per"]
    mu = C["mu"]
    is_C = mu < p["c_thr"]
    k0 = p["seas_k"]
    S = (3.0 * per + k0 * glob[None, :]) / (3.0 + k0)
    if p["flat_C"]:
        S = np.where(is_C[:, None], glob[None, :], S)
    S = np.clip(np.maximum(S, 1e-6), p["seas_cap_lo"], p["seas_cap_hi"])

    dt = d / S[:, np.arange(T) % 52]

    def lvl(k):
        k = min(max(int(k), 1), T)
        return dt[:, T - k :].mean(axis=1)

    lv = p["w_recent"] * lvl(p["win_short"]) + (1 - p["w_recent"]) * lvl(p["win_long"])
    if p["c_win"] != p["win_short"]:
        lv_c = p["w_recent"] * lvl(p["c_win"]) + (1 - p["w_recent"]) * lvl(
            p["win_long"]
        )
        lv = np.where(is_C, lv_c, lv)
    lv = np.maximum(lv, p["min_level"])

    out = np.zeros((3, n))
    for hh in range(3):
        out[hh] = lv * S[:, (T + hh) % 52]
    return out


def _nb_quantile(mu, var, q):
    from scipy import stats

    mu = np.maximum(np.asarray(mu, float), 1e-9)
    var = np.maximum(np.asarray(var, float), mu * (1 + 1e-9))
    pois = (var - mu) <= 1e-9
    r = np.where(pois, 1e12, mu * mu / np.maximum(var - mu, 1e-9))
    p = r / (r + mu)
    out = stats.nbinom.ppf(q, r, p)
    return np.maximum(np.where(np.isfinite(out), out, mu), 0.0)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------
def decide(week_index, state, revealed):
    C = _load()
    state = np.asarray(state, dtype=float)
    if state.ndim == 1:
        state = state.reshape(1, -1)
    n = state.shape[0]
    revealed = (
        np.empty((0, n))
        if revealed is None
        else np.asarray(revealed, dtype=float).reshape(-1, n)
    )

    fc = _forecast(int(week_index), revealed)
    mu, phi = C["mu"], C["phi"]
    p = PARAMS

    mu3 = fc.sum(axis=0)
    var3 = (fc * (1.0 + phi[None, :] * fc)).sum(axis=0)

    is_A = mu >= p["a_thr"]
    is_C = mu < p["c_thr"]
    q = np.where(is_A, p["q_A"], np.where(is_C, p["q_C"], p["q_B"]))
    target = _nb_quantile(mu3, var3, q)
    ip = state.sum(axis=1)
    order = np.maximum(0.0, target - ip)
    order = np.where(np.isfinite(order), order, 0.0)
    return np.rint(order)
