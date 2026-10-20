"""VN2 adaptive ordering policy.

Design
------
Croston/TSB intermittent-demand forecaster + base-stock (order-up-to) policy.

* Forecast: demand size, inter-demand interval and probability of a
  non-zero week are exponentially smoothed (TSB; plain Croston when the
  probability weight is switched off).  The per-series smoothing state is
  warmed once at import from the public sales history (`data/sales.csv`,
  censoring-corrected using `data/availability.csv`) and then updated every
  round with the revealed in-game demand, exactly as the contract requires.
* Order: base-stock.  Inventory position = on_hand + arrival this week +
  arrival next week.  Order up to the forecast demand over the coverage
  horizon (2-week lead time + 1 review week) plus a safety-stock buffer.
  The buffer is the normal newsvendor quantile for the cost ratio
  (1.00 EUR per missed unit vs 0.20 EUR per held unit-week), i.e. a
  critical fractile of roughly 0.8.

Reads only the arena `data/` folder (next to this file), at import time.
Never writes anything.  Stateless between calls (recomputes from the warmed
state plus the revealed history each round), so any `n_series` works.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is installed in the arena
    pd = None

_HERE = Path(__file__).resolve().parent


def _candidate_data_dirs():
    here = _HERE
    yield here.parent.parent / "data"  # .../vn2-model-arena/data
    yield here.parent / "data"
    yield Path.cwd() / "data"
    yield Path.cwd().parent / "data"


# ------------------------------------------------------------------ tuning
ALPHA = 0.25  # smoothing constant for history warm-up (sizes / intervals / 2nd moment)
ALPHA_REVEAL = 0.50  # smoothing constant for revealed in-game demand (faster adaption)
ALPHA_P = 0.20  # smoothing constant for P(non-zero demand) (TSB)
HORIZON = 3  # coverage horizon: 2-week lead time + 1 review week
K_SAFETY = 0.90  # normal newsvendor quantile (critical fractile ~0.8)
CENSOR_FILL = True  # impute censored history weeks (availability == 0)
USE_TSB = True  # weight forecast by P(non-zero); False = plain Croston
USE_BIAS = True  # multiplicative level bias from revealed-vs-forecast fit
BIAS_MIN = 0.6  # clamp for the online level bias
BIAS_MAX = 1.8
SAFE_FRACTION = 0.30  # safety-stock floor as share of the forecast mean
VAR_BLEND = 0.50  # blend of model variance and empirical recent variance

_NL = 0  # number of series in the warmed history (set at import)
_SERIES_IDS = ()
_HIST = None  # (n, 157) censoring-corrected history
_Z = _P = _PROB = _Z2 = _LAST = None  # warmed TSB state (per series)
_VAR_EMP = None  # empirical weekly variance, recent 52 weeks
_MEAN_HIST = None  # per-series history mean (fallback scale)


def _load_arrays():
    """Load and align sales/availability history; return (ids, y, av) or None."""
    if pd is None:
        return None
    sales = avail = None
    for d in _candidate_data_dirs():
        try:
            if (d / "sales.csv").exists() and (d / "availability.csv").exists():
                sales = pd.read_csv(d / "sales.csv")
                avail = pd.read_csv(d / "availability.csv")
                break
        except Exception:
            continue
    if sales is None:
        return None
    sales = sales.sort_values(["series_id", "week"], kind="mergesort")
    avail = avail.sort_values(["series_id", "week"], kind="mergesort")
    ids = sorted(sales["series_id"].unique())
    n = len(ids)
    y = sales["y"].to_numpy(dtype=float).reshape(n, -1)
    av = avail["is_available"].to_numpy(dtype=float).reshape(n, -1)
    return ids, y, av


def _tsb_warm(y, av):
    """Exponential-smoothing state after the full history, per series."""
    n, T = y.shape
    alpha = ALPHA
    ap = ALPHA_P
    z = np.zeros(n)  # smoothed demand size
    p = np.ones(n)  # smoothed inter-demand interval
    prob = np.zeros(n)  # smoothed P(demand in a week)
    z2 = np.zeros(n)  # second moment of demand sizes
    last = np.full(n, -1.0)  # week index of last demand
    y_use = y
    if CENSOR_FILL:
        with np.errstate(invalid="ignore", divide="ignore"):
            avail_mean = np.where(av > 0, y, np.nan).mean(axis=1)
        avail_mean = np.where(np.isfinite(avail_mean), avail_mean, 0.0)
        censored = av <= 0
        if censored.any():
            y_use = y.copy()
            rows, cols = np.nonzero(censored)
            y_use[rows, cols] = np.take(avail_mean, rows)
    for t in range(T):
        d = y_use[:, t]
        occ = d > 0
        if occ.any():
            idx = np.nonzero(occ)[0]
            elapsed = t - last[idx]
            z[idx] = alpha * d[idx] + (1.0 - alpha) * z[idx]
            p[idx] = alpha * elapsed + (1.0 - alpha) * p[idx]
            z2[idx] = alpha * (d[idx] * d[idx]) + (1.0 - alpha) * z2[idx]
            last[idx] = t
        if USE_TSB:
            prob = ap * occ + (1.0 - ap) * prob
    # empirical recent variance (last 52 weeks) for the safety-stock blend
    recent = y_use[:, -52:]
    var_emp = recent.var(axis=1)
    mean_hist = y_use.mean(axis=1)
    return z, p, prob, z2, last, var_emp, mean_hist


def _warm_history():
    global _NL, _SERIES_IDS, _HIST, _Z, _P, _PROB, _Z2, _LAST, _VAR_EMP, _MEAN_HIST
    loaded = _load_arrays()
    if loaded is None:
        _NL = 0
        return
    ids, y, av = loaded
    _SERIES_IDS = tuple(ids)
    _HIST = y
    (_Z, _P, _PROB, _Z2, _LAST, _VAR_EMP, _MEAN_HIST) = _tsb_warm(y, av)
    _NL = len(ids)


_warm_history()


def _decide_history_path(n, state, revealed):
    """Croston/TSB base-stock decision when series align with history."""
    z = _Z.copy()
    p = _P.copy()
    prob = _PROB.copy()
    z2 = _Z2.copy()
    last = _LAST.copy()
    var_emp = _VAR_EMP
    T = _HIST.shape[1]

    w = revealed.shape[0]
    lam_hist = np.zeros((max(w, 1), n))  # per-period forecast at each revealed week
    for t in range(w):
        if USE_TSB:
            lam_hist[t] = prob * z / np.maximum(p, 1.0)
        else:
            lam_hist[t] = z / np.maximum(p, 1.0)
        d = revealed[t]
        occ = d > 0
        if occ.any():
            idx = np.nonzero(occ)[0]
            gweek = T + t
            elapsed = gweek - last[idx]
            z[idx] = ALPHA_REVEAL * d[idx] + (1.0 - ALPHA_REVEAL) * z[idx]
            p[idx] = ALPHA_REVEAL * elapsed + (1.0 - ALPHA_REVEAL) * p[idx]
            z2[idx] = ALPHA_REVEAL * (d[idx] * d[idx]) + (1.0 - ALPHA_REVEAL) * z2[idx]
            last[idx] = gweek
        if USE_TSB:
            prob = ALPHA_P * occ + (1.0 - ALPHA_P) * prob

    if USE_TSB:
        lam = prob * z / np.maximum(p, 1.0)
        model_var = prob * z2 / np.maximum(p, 1.0) - lam * lam
    else:  # plain Croston
        lam = z / np.maximum(p, 1.0)
        model_var = (z2 / np.maximum(p, 1.0)) - lam * lam
    model_var = np.maximum(model_var, 0.0)

    # online multiplicative level bias: how revealed demand ran vs forecast
    bias = np.ones(n)
    if USE_BIAS and w > 0:
        used = lam_hist[:w].sum(axis=0)
        actual = revealed.sum(axis=0)
        global_ratio = actual.sum() / max(used.sum(), 1e-9)
        global_ratio = float(np.clip(global_ratio, BIAS_MIN, BIAS_MAX))
        exposed = used >= 1.0
        ratio = np.where(exposed, actual / np.maximum(used, 1e-9), global_ratio)
        bias = np.clip(ratio, BIAS_MIN, BIAS_MAX)
        bias = np.where(exposed, bias, global_ratio)

    lam = lam * bias
    model_var = model_var * (bias * bias)

    if var_emp is not None:
        week_var = VAR_BLEND * model_var + (1.0 - VAR_BLEND) * var_emp
    else:
        week_var = model_var
    week_var = np.maximum(week_var, 1e-6)

    sigma_week = np.sqrt(week_var)
    sigma_horizon = np.sqrt(HORIZON) * np.maximum(sigma_week, SAFE_FRACTION * lam)
    target = HORIZON * lam + K_SAFETY * sigma_horizon

    position = state[:, 0] + state[:, 1] + state[:, 2]
    order = target - position
    return np.maximum(0.0, order)


def _decide_fallback(n, state, revealed):
    """Plain revealed-demand base-stock for unknown series sets."""
    w = revealed.shape[0]
    if w > 0:
        lam = revealed.mean(axis=0)
        week_var = revealed.var(axis=0)
    else:
        lam = np.full(n, 0.5)
        week_var = np.full(n, 0.25)
    week_var = np.maximum(week_var, 1e-6)
    sigma_week = np.sqrt(week_var)
    sigma_horizon = np.sqrt(HORIZON) * np.maximum(sigma_week, SAFE_FRACTION * lam)
    target = HORIZON * lam + K_SAFETY * sigma_horizon
    position = state[:, 0] + state[:, 1] + state[:, 2]
    return np.maximum(0.0, target - position)


def decide(week_index, state, revealed):
    """Base-stock order for the week about to start (see TASK3.md contract)."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[1] != 3:
        state = np.asarray(state, dtype=float).reshape(-1, 3)
    n = state.shape[0]
    revealed = np.asarray(revealed, dtype=float)
    if revealed.size == 0:
        revealed = np.empty((0, n), dtype=float)
    elif revealed.ndim == 1:
        revealed = revealed.reshape(1, -1)
    if revealed.shape[1] != n:
        if revealed.shape[1] > n:
            revealed = revealed[:, :n]
        else:
            revealed = np.pad(revealed, ((0, 0), (0, n - revealed.shape[1])))

    week_index = int(week_index)
    if week_index >= 6 or n == 0:
        # Weeks 7-8 have no order slot.
        return np.zeros(n, dtype=float)

    if _NL == n and _HIST is not None:
        return _decide_history_path(n, state, revealed)
    return _decide_fallback(n, state, revealed)


if __name__ == "__main__":  # pragma: no cover - smoke test / local backtest
    import helper_backtest as hb

    st = None
    if _SERIES_IDS:
        for d in _candidate_data_dirs():
            try:
                p = d / "initial_state.csv"
                if not p.exists():
                    continue
                st_df = pd.read_csv(p)
                need = {"series_id", "on_hand", "arrival_w1", "arrival_w2"}
                if not need <= set(st_df.columns):
                    continue
                pos = {sid: i for i, sid in enumerate(_SERIES_IDS)}
                idx = [pos.get(s) for s in st_df["series_id"]]
                if any(i is None for i in idx):
                    continue
                order = np.argsort(idx, kind="mergesort")
                st = st_df.iloc[order][
                    ["on_hand", "arrival_w1", "arrival_w2"]
                ].to_numpy(dtype=float)
                break
            except Exception:
                continue

    if _HIST is not None:
        for start in (100, 130):
            r = hb.backtest(decide, _HIST, start_week=start, initial=st)
            print(f"start={start} score={r['score']:.1f}")
    else:
        print("no history loaded; cannot backtest")
