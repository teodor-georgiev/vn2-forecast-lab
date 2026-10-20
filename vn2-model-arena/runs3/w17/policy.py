"""VN2 adaptive ordering policy - family-level hierarchical forecasting.

Method
------
1. Build a censoring-corrected weekly demand panel from sales.csv /
   availability.csv (censored weeks are interpolated along time, then filled
   with the series mean).
2. Estimate a shared yearly seasonal shape `g[week % 52]` on the deseasonalised
   aggregate demand, then a family-level (DepartmentGroup x StoreFormat) level
   which is allocated bottom-up to member series via shrunk recent demand
   shares; this family-implied level is blended with each series' own trailing
   level.
3. Adapt every series' level with the demand actually revealed inside this game
   (per-series mismatch multiplier, clipped), then order up to the per-series
   newsvendor quantile of week w+2 demand - the only week this order can still
   serve - net of the stock already on hand or in transit minus the expected
   demand of the two weeks in between.

The policy only reads the arena `data/` folder, writes nothing, handles any n,
and replays in well under a second.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# --------------------------------------------------------------------- config
_CFG = {
    "hist_weeks_level": 13,
    "halflife": 8.0,
    "q": 0.88,
    "q_mode": "level",
    "q_beta": 1.0,
    "q_lo": 0.5,
    "q_hi": 0.99,
    "group_cols": ("DepartmentGroup", "StoreFormat"),
    "seasonal_power": 1.0,
    "seasonal_smooth": 3,
    "level_shrink_a": 0.5,
    "share_blend": 0.5,
    "kappa_shrink": 2.0,
    "m_k": 1.0,
    "m_clip": (0.2, 5.0),
    "proj_w": 1.0,
    "fill_censored": True,
}

_PROFILE_CACHE: dict = {}


def _data_dir():
    here = Path(__file__).resolve().parent
    for c in (
        here.parent.parent / "data",
        here.parent / "data",
        here / "data",
        here.parent.parent.parent / "data",
    ):
        if (c / "sales.csv").exists():
            return c
    return None


def _load_raw():
    """Return (S, A, attrs) or None; S (n,h) sales, A (n,h) availability bool."""
    try:
        import pandas as pd
    except Exception:
        return None
    d = _data_dir()
    if d is None:
        return None
    try:
        s = pd.read_csv(d / "sales.csv")
        piv = s.pivot(index="series_id", columns="week", values="y").sort_index()
        S = piv.to_numpy(dtype=float)
        try:
            a = pd.read_csv(d / "availability.csv")
            ap = a.pivot(
                index="series_id", columns="week", values="is_available"
            ).sort_index()
            ap = ap.reindex(index=piv.index, columns=piv.columns)
            A = (
                np.where(
                    np.isnan(ap.to_numpy(dtype=float)), 1.0, ap.to_numpy(dtype=float)
                )
                > 0.5
            )
        except Exception:
            A = np.ones_like(S, dtype=bool)
        try:
            attrs = pd.read_csv(d / "product_attributes.csv").set_index("series_id")
            attrs = attrs.reindex(piv.index)
        except Exception:
            attrs = None
        return S, A, attrs
    except Exception:
        return None


def _profile(cfg):
    key = (
        cfg["group_cols"],
        cfg["hist_weeks_level"],
        cfg["halflife"],
        cfg["seasonal_power"],
        cfg["seasonal_smooth"],
        cfg["fill_censored"],
        cfg["level_shrink_a"],
        cfg["share_blend"],
        cfg["kappa_shrink"],
    )
    prof = _PROFILE_CACHE.get(key)
    if prof is not None:
        return prof
    raw = _load_raw()
    if raw is None:
        _PROFILE_CACHE[key] = None
        return None
    S, A, attrs = raw
    n, h = S.shape

    # --- censoring correction ---------------------------------------------
    if cfg["fill_censored"] and A.any():
        import pandas as pd

        Df = pd.DataFrame(S).where(pd.DataFrame(A))
        Df = Df.interpolate(axis=1, limit_direction="both")
        rowmean = pd.Series(S.mean(axis=1), index=Df.index)
        rowmean = Df.mean(axis=1, skipna=True).fillna(rowmean).fillna(0.0)
        Df = Df.apply(lambda r: r.fillna(rowmean), axis=1)
        D = Df.to_numpy(dtype=float)
        D = np.where(np.isnan(D), 0.0, D)
    else:
        D = np.array(S, dtype=float)

    mean_i = np.maximum(D.mean(axis=1), 1e-6)

    # --- shared seasonal shape --------------------------------------------
    phase = np.arange(h) % 52
    if cfg["seasonal_power"] > 0:
        g = np.zeros(52)
        cnt = np.zeros(52)
        np.add.at(g, phase, (D / mean_i[:, None]).sum(axis=0))
        np.add.at(cnt, phase, 1.0)
        g = g / np.maximum(cnt, 1.0)
        wgt = cnt / cnt.sum()
        g = g / max(float((g * wgt).sum()), 1e-6)
        sw = int(cfg["seasonal_smooth"])
        if sw > 1:
            pad = sw // 2
            ext = np.r_[g[-pad:], g, g[:pad]]
            g = np.convolve(ext, np.ones(sw) / sw, mode="same")[pad : pad + 52]
            g = g / max(float((g * wgt).sum()), 1e-6)
        g = np.maximum(g, 0.05) ** float(cfg["seasonal_power"])
    else:
        g = np.ones(52)

    Dd = D / g[phase]
    Dd = np.where(np.isfinite(Dd), Dd, 0.0)

    # --- trailing level per series ----------------------------------------
    L = max(2, min(int(cfg["hist_weeks_level"]), h))
    wts = 0.5 ** (np.arange(L)[::-1] / float(cfg["halflife"]))
    wts = wts / wts.sum()
    level = np.maximum((Dd[:, -L:] * wts).sum(axis=1), 1e-6)

    # --- family structure + bottom-up allocation --------------------------
    gid = np.zeros(n, dtype=int)
    if attrs is not None and cfg["group_cols"]:
        cols = [c for c in cfg["group_cols"] if c in attrs.columns]
        if cols:
            kdf = attrs[cols].astype(str).agg("|".join, axis=1)
            codes, uniq = np.unique(kdf.to_numpy(), return_inverse=True)
            gid = uniq.astype(int)
    n_g = int(gid.max()) + 1 if n else 1
    size = np.bincount(gid, minlength=n_g).astype(float)
    rec = Dd[:, -L:].sum(axis=1)
    a = float(cfg["level_shrink_a"])
    rec_g = np.bincount(gid, weights=rec, minlength=n_g)[gid]
    share = (rec + a) / np.maximum(rec_g + a * size[gid], 1e-6)
    Lsum_g = np.bincount(gid, weights=level, minlength=n_g)[gid]
    level_hier = share * np.maximum(Lsum_g, 1e-6)
    wvol = np.clip(
        np.clip(rec / np.maximum(rec + 1.0, 1e-6), 0.0, 1.0)
        ** float(cfg["share_blend"])
        * 2.0,
        0.0,
        0.95,
    )
    level_eff = np.maximum(wvol * level + (1.0 - wvol) * level_hier, 1e-6)

    # --- dispersion: normalised single-week demand, series and family ------
    norm = Dd / mean_i[:, None]
    order = np.argsort(norm, axis=1)
    sorted_norm = np.take_along_axis(norm, order, axis=1)
    kap_series = sorted_norm  # quantile lookup done at call time
    kap_pool = {}
    for gi in range(n_g):
        m = gid == gi
        if m.any():
            kap_pool[gi] = np.sort(norm[m].ravel())
    kappa_shrink = float(cfg["kappa_shrink"])
    wkp = rec / np.maximum(rec + kappa_shrink, 1e-6)

    prof = {
        "n": n,
        "h": h,
        "g": g,
        "phase_last": (h - 1) % 52,
        "level_eff": level_eff,
        "level": level,
        "gid": gid,
        "sorted_norm": kap_series,
        "kap_pool": kap_pool,
        "wkp": wkp,
    }
    _PROFILE_CACHE[key] = prof
    return prof


def _quantiles(prof, q):
    """Per-series q-quantile of normalised single-week demand (volume shrunk)."""
    n = prof["n"]
    sn = prof["sorted_norm"]
    h = sn.shape[1]
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    idx = np.clip(np.rint(q * (h - 1)).astype(int), 0, h - 1)
    kap_i = sn[np.arange(n), idx]
    gid = prof["gid"]
    kap_g = np.empty(n)
    pool = prof["kap_pool"]
    for gi, arr in pool.items():
        m = gid == gi
        if not m.any():
            continue
        j = np.clip(np.rint(q[m] * (arr.size - 1)).astype(int), 0, arr.size - 1)
        kap_g[m] = arr[j]
    wkp = prof["wkp"]
    return np.clip(wkp * kap_i + (1.0 - wkp) * kap_g, 0.02, 60.0)


def _fallback(week_index, n, state, revealed):
    """No/incompatible history: crude mean-based rule (passes the contract)."""
    state = np.asarray(state, dtype=float)
    if revealed is not None and np.size(revealed):
        lvl = np.asarray(revealed, dtype=float).reshape(-1, n).mean(axis=0)
    else:
        lvl = np.full(n, 1.0)
    avail = state[:, 0] + state[:, 1] + state[:, 2]
    return np.maximum(0.0, 1.6 * lvl + 0.5 - avail)


def decide(week_index, state, revealed):
    """week_index: int 0..7 - index of the week about to start (0 = before week 1).
    state:        np.ndarray (n_series, 3): on_hand, delivery_during_this_week,
                  delivery_during_next_week.
    revealed:     np.ndarray (week_index, n_series): actual demand observed so far
                  within this game; empty (0, n) array at week_index 0.
    returns:      np.ndarray (n_series,) of non-negative order quantities.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim != 2:
        state = np.atleast_2d(state)
    n = state.shape[0]
    if revealed is None:
        R = np.empty((0, n), dtype=float)
    else:
        R = np.asarray(revealed, dtype=float)
        R = R.reshape(-1, n) if R.size else np.empty((0, n), dtype=float)
    w = int(week_index)

    cfg = _CFG
    prof = _profile(cfg)
    if prof is None or prof["n"] != n:
        return _fallback(w, n, state, R)

    g = prof["g"]
    p0 = prof["phase_last"]
    ph = [(p0 + 1 + w + k) % 52 for k in range(3)]
    hh = np.array([float(g[p]) for p in ph])  # seasonal factors for w, w+1, w+2

    level = prof["level_eff"]

    # --- adapt the level with the demand revealed inside this game --------
    if w > 0 and R.shape[0] > 0:
        wr = min(w, R.shape[0])
        expected = level * sum(float(g[(p0 + 1 + t) % 52]) for t in range(wr))
        k = float(cfg["m_k"])
        m = (R[:wr].sum(axis=0) + k) / (expected + k)
        lo, hi = cfg["m_clip"]
        level = level * np.clip(m, lo, hi)

    # --- newsvendor quantile for week w+2 --------------------------------
    if cfg["q_mode"] == "level":
        co = 0.2 * float(cfg["q_beta"]) / np.maximum(level, 1e-6)
        q = 1.0 / (1.0 + co)
        q = np.clip(q, float(cfg["q_lo"]), float(cfg["q_hi"]))
    else:
        q = np.full(n, float(cfg["q"]))
    kap = _quantiles(prof, q)

    target = kap * level * hh[2]  # q-quantile of week w+2 demand
    avail = state[:, 0] + state[:, 1] + state[:, 2]
    proj = np.maximum(0.0, avail - float(cfg["proj_w"]) * level * (hh[0] + hh[1]))
    order = np.maximum(0.0, target - proj)
    order = np.where(np.isfinite(order), order, 0.0)
    return order
