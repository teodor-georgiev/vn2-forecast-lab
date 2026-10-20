"""VN2 arena policy (w02): empirical per-series newsvendor quantiles + reorder-up-to.

Method
------
1. Censoring-corrected history: sales.csv + availability.csv; weeks with
   availability == 0 are censored (true demand > 0) and are imputed from the
   series' own same-season (day-of-year +- 3 weeks) uncensored median, with a
   level x portfolio-seasonal fallback.
2. Seasonality: portfolio weekly profile from year-normalised totals, pooled
   over a +-POOL day window around each day-of-year; per-series own ratio
   (year-normalised, same pooling) shrunk in log space towards the portfolio.
3. Level: EWMA of the deseasonalised demand (half-life `hl` weeks), blended
   with a raw EWMA level (`raw_hl`) - the weekly retail season is strong but
   the level drifts, so both views are used.
4. Demand-shape pool: for every series the empirical distribution of its
   relative weekly demand deviation (its newsvendor distribution), taken from
   the censoring-corrected history.
5. Decision - arrival-week order-up-to (lead time 2 weeks):
       proj   = max(0, IP - forecast(week w) - forecast(week w+1))
       target = quantile_q1(series' own 1-week demand distribution)
                x forecast(week w+2)
       order  = max(0, target - proj)
   with IP = on_hand + delivery this week + delivery next week.  The fractile
   q1 is the newsvendor critical ratio for 1.00 EUR per missed unit versus
   0.20 EUR per held unit-week, chosen on backtested simulated cost; it is
   raised for the last two orders (q4, q5), whose leftovers are held for only
   a short time, so a miss there is relatively more expensive.
6. Feedback: the in-game revealed demand (true, uncensored) updates a
   per-series level factor.  Trust in the observed ratio grows with the
   observed volume (Poisson-like precision), so large series react quickly
   while noisy small series are barely moved.
7. Leakage guard: if the revealed game weeks match a column of the history
   (i.e. the game is replayed inside sales.csv), every week after the game
   start is dropped from the model, so no future information can leak in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

POOL = 7  # day-of-year pooling half-window (days)

# ----------------------------------------------------------------- tunables
PARAMS = {
    # ---- shipped configuration (tuned on 10 historical cutoffs) -----------
    "mode": "deficit",  # arrival-week order-up-to rule
    "q1": 0.70,  # newsvendor fractile for the arrival week (per-series quantile)
    "q4": 0.80,  # fractile for the order that lands in the penultimate week
    "q5": 0.90,  # fractile for the order that lands in the final week
    "beta": 0.0,  # weight of the week after the arrival week
    "blend": 0.90,  # weight of the raw (non-seasonal) level in the forecast
    "alpha": 0.40,  # strength of the in-game level correction
    "fb_c": 2.0,  # volume scale of the per-series trust in the observed ratio
    "hl": 8.0,  # EWMA half-life (weeks) for the deseasonalised level
    "raw_hl": 13.0,  # EWMA half-life for the raw level
    "level_scale": 1.0,  # multiplicative bias correction of the level
    "damp": 1.0,  # damping exponent on the in-game level factor
    "fb_cap": 0.6,  # max |log| of the in-game level factor
    "shrink_k": 8.0,  # shrinkage of per-series seasonality to the portfolio
    "pool_years": 3,  # history years used for the demand-shape pool
    # ---- alternative modes / knobs kept for reproducibility ---------------
    "W": 3,  # protection window length (weeks) of the order-up-to mode
    "start_off": 0,  # protection window start offset from the decision week
    "q": 0.80,  # fractile of the order-up-to mode
    "k_over": 0.5,  # expected unit-weeks an excess unit is carried (news mode)
    "k4": 1.0,  # tail multiplier of k_over for the decision in week 5
    "k5": 1.0,  # tail multiplier of k_over for the decision in week 6
}

# Backtest hooks (the manager's replay leaves them at None).
HISTORY_END = None  # number of usable history columns (int) or None
GAME_DOY = None  # explicit list of 8 day-of-year values for the game weeks

_CACHE: dict = {}


# ------------------------------------------------------------------ data io
def _find_data_dir():
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        cand = p / "data"
        if (cand / "sales.csv").exists():
            return cand
    fixed = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")
    return fixed if (fixed / "sales.csv").exists() else None


def _raw():
    """Return (sales, avail, series_ids, week dates)."""
    d = _find_data_dir()
    if d is None:
        return None
    s = pd.read_csv(d / "sales.csv", parse_dates=["week"])
    a = pd.read_csv(d / "availability.csv", parse_dates=["week"])
    sp = s.pivot(index="series_id", columns="week", values="y").sort_index(axis=1)
    ap = a.pivot(index="series_id", columns="week", values="is_available").astype(float)
    ap = ap.reindex(index=sp.index, columns=sp.columns)
    return (
        np.nan_to_num(sp.to_numpy(float), nan=0.0),
        (np.nan_to_num(ap.to_numpy(float), nan=1.0) > 0.5).astype(float),
        sp.index.to_numpy(),
        sp.columns,
    )


# ------------------------------------------------------------- preparation
def _prepare(history_end=None, game_doy=None):
    key = (history_end, None if game_doy is None else tuple(game_doy))
    if key in _CACHE:
        return _CACHE[key]

    raw = _raw()
    if raw is None:
        _CACHE[key] = None
        return None
    sales, avail, sids, weeks = raw
    n, T = sales.shape
    dates = pd.DatetimeIndex(weeks)
    doy = np.asarray(dates.dayofyear, dtype=int)  # 1..366
    year = np.asarray(dates.year, dtype=int)
    T_use = T if history_end is None else int(min(history_end, T))
    D = 366

    # ---- portfolio seasonal profile (year-normalised weekly totals) ------
    tot = sales[:, :T_use].sum(axis=0)
    ts = tot.copy()
    for yv in np.unique(year[:T_use]):
        cc = np.where(year[:T_use] == yv)[0]
        ts[cc] = tot[cc] / max(tot[cc].mean(), 1e-9)
    acc = np.zeros(D)
    cnt = np.zeros(D)
    for t in range(T_use):
        for o in range(-POOL, POOL + 1):
            j = (doy[t] - 1 + o) % D
            acc[j] += ts[t]
            cnt[j] += 1.0
    prof = acc / np.maximum(cnt, 1.0)
    prof = np.maximum(prof, 1e-9)
    prof /= prof.mean()

    # ---- impute censored weeks -------------------------------------------
    ok = avail[:, :T_use] > 0.5
    dem = sales[:, :T_use].copy()
    lvl0 = np.where(
        ok.sum(axis=1) > 0,
        (dem * ok).sum(axis=1) / np.maximum(1, ok.sum(axis=1)),
        dem.mean(axis=1),
    )
    sacc = np.zeros((n, D))
    scnt = np.zeros(D)
    for t in range(T_use):
        for o in (-21, -14, -7, 0, 7, 14, 21):
            j = (doy[t] - 1 + o) % D
            sacc[:, j] += np.where(ok[:, t], dem[:, t], 0.0)
            scnt[j] += 1.0
    smed = np.where(scnt[None, :] > 0, sacc / np.maximum(scnt[None, :], 1.0), np.nan)
    fill = np.where(np.isnan(smed), lvl0[:, None] * prof[None, :], smed)
    rows, cols = np.where(~ok)
    dem[rows, cols] = fill[rows, doy[cols] - 1]
    dem = np.maximum(dem, 0.0)

    # ---- per-series seasonality (portfolio x shrunk own ratio) -----------
    y = dem / prof[doy[:T_use] - 1][None, :]
    yrs_all = np.unique(year[:T_use])
    l_year = np.zeros((n, len(yrs_all)))
    for i, yv in enumerate(yrs_all):
        cc = np.where(year[:T_use] == yv)[0]
        l_year[:, i] = y[:, cc].mean(axis=1) if cc.size else y.mean(axis=1)
    yn = y / np.maximum(l_year[:, np.searchsorted(yrs_all, year[:T_use])], 1e-9)
    oacc = np.zeros((n, D))
    ocnt = np.zeros(D)
    for t in range(T_use):
        for o in range(-POOL, POOL + 1):
            j = (doy[t] - 1 + o) % D
            oacc[:, j] += yn[:, t]
            ocnt[j] += 1.0
    own = np.where(ocnt[None, :] > 0, oacc / np.maximum(ocnt[None, :], 1.0), 1.0)
    nobs = np.broadcast_to(ocnt[None, :], own.shape)
    own = np.clip(own, 0.05, 20.0)
    k = PARAMS["shrink_k"]
    log_own = (nobs * np.log(own)) / (nobs + k)
    seas = prof[None, :] * np.exp(log_own)

    # ---- recent level: EWMA of deseasonalised demand ---------------------
    dese = dem / np.maximum(seas[:, doy[:T_use] - 1], 1e-9)
    age = np.arange(T_use - 1, -1, -1, dtype=float)
    wts = 0.5 ** (age / max(1e-6, PARAMS["hl"]))
    level = (dese * wts[None, :]).sum(axis=1) / wts.sum()
    rts = 0.5 ** (age / max(1e-6, PARAMS["raw_hl"]))
    raw = (dem * rts[None, :]).sum(axis=1) / rts.sum()

    # ---- shape pool: deviation of the W-week aggregate -------------------
    yrs = np.unique(year[:T_use])
    keep = yrs[-int(PARAMS["pool_years"]) :] if len(yrs) > 1 else yrs
    mask = np.isin(year[:T_use], keep)
    l_yr = np.zeros((n, len(keep)))
    for i, yv in enumerate(keep):
        cc = np.where(year[:T_use] == yv)[0]
        l_yr[:, i] = dese[:, cc].mean(axis=1) if cc.size else dese.mean(axis=1)
    yidx2 = np.searchsorted(keep, year[:T_use])
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(dese, axis=1)], axis=1)
    pools = {}
    for W in (1, 2, 3, 4):
        cols = np.where(mask)[0]
        cols = cols[cols >= W - 1]
        if cols.size == 0:
            pools[W] = np.ones((n, 1))
            continue
        agg = csum[:, cols + 1] - csum[:, cols + 1 - W]  # W-week sum ending at col
        base = W * l_yr[:, yidx2[cols]]
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = agg / np.maximum(base, 1e-9)
        dev = np.where(np.isfinite(dev), dev, 1.0)
        dev = np.clip(dev, 0.0, 40.0)
        pools[W] = dev

    # ---- game weeks: calendar position -----------------------------------
    if game_doy is None:
        last = dates[-1]
        gd = pd.DatetimeIndex([last + pd.Timedelta(days=7 * (g + 1)) for g in range(8)])
        gdoy = np.asarray(gd.dayofyear, dtype=int)
    else:
        gdoy = np.asarray(game_doy, dtype=int)

    out = {
        "dem": dem,
        "seas": seas,
        "level": level,
        "raw": raw,
        "dese": dese,
        "doy": doy[:T_use],
        "gdoy": gdoy,
        "pools": pools,
        "n": n,
        "T": T_use,
        "sids": sids,
    }
    _CACHE[key] = out
    return out


# ---------------------------------------------------------------- the policy
def _forecast(P, week_index, revealed):
    """Weekly forecast matrix (8, n) and the in-game level factor (n,)."""
    n = P["n"]
    seas, level, gdoy = P["seas"], P["level"], P["gdoy"]
    raw = P["raw"]
    b = float(PARAMS["blend"])
    ls = float(PARAMS["level_scale"])
    f = np.zeros((8, n))
    for g in range(8):
        f[g] = ((1.0 - b) * level * seas[:, gdoy[g] - 1] + b * raw) * ls
    factor = np.ones(n)
    if revealed is not None and week_index > 0:
        rev = np.asarray(revealed, dtype=float)
        if rev.ndim == 2 and rev.shape[0] > 0 and rev.shape[1] == n:
            rev_sum = rev.sum(axis=0)
            pri = f[: rev.shape[0]].sum(axis=0)
            raw = np.where(pri > 1e-9, rev_sum / np.maximum(pri, 1e-9), 1.0)
            # trust in the observed ratio grows with the volume observed
            # (Poisson-like precision): big series react fast, noisy small
            # series are barely moved.
            c = float(PARAMS["fb_c"])
            trust = pri / (pri + c)
            a = float(PARAMS["alpha"])
            factor = 1.0 + a * trust * (raw - 1.0)
            factor = np.clip(
                factor, np.exp(-PARAMS["fb_cap"]), np.exp(PARAMS["fb_cap"])
            )
            factor = np.where(np.isfinite(factor), factor, 1.0)
    return f, factor


def _match_start(revealed, n):
    """If the game lives inside the history, find its start column."""
    try:
        raw = _raw()
    except Exception:
        return None
    if raw is None or revealed is None:
        return None
    sales = raw[0]
    rev = np.asarray(revealed, dtype=float)
    if rev.ndim != 2 or rev.shape[0] == 0 or rev.shape[1] != sales.shape[0]:
        return None
    r = rev[0]
    if r.shape[0] != sales.shape[0]:
        return None
    x = sales - sales.mean(axis=0, keepdims=True)
    y = r - r.mean()
    num = x.T @ y
    den = np.sqrt((x * x).sum(axis=0)) * (np.sqrt(float((y * y).sum())) + 1e-9)
    corr = num / (den + 1e-9)
    b = int(np.argmax(corr))
    return b if corr[b] > 0.95 else None


def _news_levels(P, g, factor, k_over):
    """Per-series newsvendor stock level for the game week `g`.

    The predictive demand sample of series s in week g is the forecast for
    that week times the series' own empirical relative weekly demand
    deviation (its newsvendor distribution).  The level minimises the
    expected arena cost
        1.00 * missed  +  0.20 * k_over * leftover
    (k_over = expected number of unit-weeks an excess unit is carried), so
    every series gets its own critical fractile, not a rule of thumb.
    """
    n = P["n"]
    rel = P["pools"][1]
    base = P["_fcache"][g] * factor if "_fcache" in P else None
    samples = base[:, None] * rel  # (n, m)
    grid = np.linspace(0.0, 1.0, 21)
    cand = np.quantile(samples, grid, axis=1)  # (21, n)
    diff = samples[None, :, :] - cand[:, :, None]  # (21, n, m)
    cost = (np.maximum(0.0, -diff) + 0.20 * k_over * np.maximum(0.0, diff)).mean(axis=2)
    best = cand[np.argmin(cost, axis=0), np.arange(n)]
    return np.maximum(0.0, best)


def decide(week_index, state, revealed):
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    P = _prepare(HISTORY_END, GAME_DOY)
    if P is None or P["n"] != n:
        return _fallback(week_index, state, revealed)

    # if the game sits inside the history, drop the future weeks from the model
    if HISTORY_END is None and GAME_DOY is None and week_index == 1:
        s = _match_start(revealed, n)
        if s is not None and s > 0:
            P2 = _prepare(s, None)
            if P2 is not None and P2["n"] == n:
                P = P2

    f, factor = _forecast(P, week_index, revealed)
    seas, gdoy = P["seas"], P["gdoy"]
    ip = state.sum(axis=1)  # on_hand + delivery this week + next week
    W = int(PARAMS["W"])
    off = int(PARAMS["start_off"])
    g0 = int(np.clip(week_index + off, 0, 7))
    g1 = int(np.clip(g0 + W - 1, 0, 7))
    wlen = max(1, g1 - g0 + 1)
    pool = P["pools"].get(wlen, P["pools"][1])

    if str(PARAMS.get("mode", "base")) in ("deficit", "news"):
        # ---- arrival-week deficit rule -----------------------------------
        # the order lands in week w+2; the stock still on hand then is the
        # inventory position minus the demand of the two weeks in between.
        # cover the arrival week with a per-series newsvendor level and add a
        # share of the following week.
        g_prev = int(np.clip(week_index, 0, 7))
        g_mid = int(np.clip(week_index + 1, 0, 7))
        g_arr = int(np.clip(week_index + 2, 0, 7))
        g_nxt = int(np.clip(week_index + 3, 0, 7))
        proj = np.maximum(0.0, ip - (f[g_prev] + f[g_mid]) * factor)
        if str(PARAMS.get("mode")) == "news":
            k_over = float(PARAMS["k_over"])
            if week_index >= 5:
                k_over *= float(PARAMS["k5"])
            elif week_index == 4:
                k_over *= float(PARAMS["k4"])
            P["_fcache"] = f
            arr = _news_levels(P, g_arr, factor, k_over)
        else:
            q1 = float(PARAMS["q1"])
            if week_index >= 5:
                q1 = float(PARAMS["q5"])
            elif week_index == 4:
                q1 = float(PARAMS["q4"])
            kap1 = np.quantile(P["pools"][1], q1, axis=1)
            arr = kap1 * f[g_arr] * factor
        target = arr + float(PARAMS["beta"]) * f[g_nxt] * factor
        return np.rint(np.maximum(0.0, target - proj))

    # newsvendor fractile: the last orders cover the tail of the game, where a
    # leftover is only held for a short time (cheap) while a miss costs 1.00.
    if week_index >= 5:
        q = float(PARAMS["q5"])
    elif week_index == 4:
        q = float(PARAMS["q4"])
    else:
        q = float(PARAMS["q"])
    kap = np.quantile(pool, q, axis=1)  # (n,)

    prot = np.zeros(n)  # seasonal protection demand over the window
    for g in range(g0, g1 + 1):
        prot += f[g] * factor
    s_level = prot * kap

    ip = state.sum(axis=1)  # on_hand + delivery this week + next week
    qty = np.maximum(0.0, s_level - ip)
    return np.rint(qty)


def _fallback(week_index, state, revealed):
    """Safety net used only if the arena data cannot be read."""
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    if revealed is not None and np.asarray(revealed).size:
        rev = np.asarray(revealed, dtype=float)
        base = rev.mean(axis=0) if rev.ndim == 2 else rev
    else:
        base = np.full(n, 1.0)
    target = 3.0 * base + 0.75 * np.sqrt(np.maximum(3.0 * base, 0.0))
    return np.rint(np.maximum(0.0, target - state.sum(axis=1)))
