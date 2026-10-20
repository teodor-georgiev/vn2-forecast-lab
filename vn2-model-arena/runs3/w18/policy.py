"""VN2 adaptive ordering policy — family-level hierarchical forecasting.

Method (3 lines)
----------------
1. Censoring correction: weeks in which a product was out of stock report zero
   sales; those weeks are imputed from a trailing availability-weighted level.
2. Hierarchical forecast: a weekly seasonal *shape* is pooled per product family,
   a *level* is estimated per series and per family, and the family total forecast
   is allocated back down to the series (bottom-up allocation by recent demand
   share) and reconciled with the own-series forecast.
3. Ordering: a simulation-based newsvendor on the *critical* week (the week the
   order arrives in) — the order covers the simulated shortage quantile of that
   week's demand net of the stock that will still be on hand, so safety stock is
   distribution-aware (per series, not a rule of thumb).

The in-game revealed demand re-anchors the remaining forecasts (adaptive).
Everything is recomputed from the arena `data/` folder (read-only); no file writes.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------
# configuration (selected on historical backtest cutoffs)
# --------------------------------------------------------------------------
LEVEL_K = 26  # weeks for the trailing level estimate
IMPUT_K = 8  # weeks for the censored-week imputation window
SEAS_K = 8  # weeks for the trailing level used by the seasonal shape
SHARE_K = 26  # weeks used for the bottom-up allocation shares
FAMILY_COL = "ProductGroup"  # attribute column used as the family level
SEAS_W = 0.25  # weight of the seasonal shape inside the own-series forecast
FAMILY_W = 0.25  # weight of the family-allocated forecast
SHRINK_LAM = 4.0  # pooling strength of the family seasonal shape
FRACTILE = 0.70  # newsvendor fractile on the critical week
N_PATHS = 256  # simulation paths for the newsvendor
ADAPT_LAM_UP = 0.50  # adaptation strength when revealed demand > forecast
ADAPT_LAM_DN = 0.25  # adaptation strength when revealed demand < forecast
ADAPT_MIN = 3.0  # minimum revealed units before trusting the adaptation
ADAPT_CLIP = (0.65, 1.75)

HISTORY_END = None  # None -> use the full history (real game). Backtests set this.

_DATA_CACHE = {}
_FC_CACHE = {}


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
def _data_dir() -> Path:
    cands = []
    try:
        cands.append(Path(__file__).resolve().parents[2] / "data")
    except Exception:
        pass
    cands.append(Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"))
    for p in cands:
        if (p / "sales.csv").exists():
            return p
    raise FileNotFoundError("arena data folder not found")


def _load():
    if "raw" in _DATA_CACHE:
        return _DATA_CACHE["raw"]
    import pandas as pd

    d = _data_dir()
    s = pd.read_csv(d / "sales.csv")
    a = pd.read_csv(d / "availability.csv")
    at = pd.read_csv(d / "product_attributes.csv")
    order = s.series_id.unique()
    sv = (
        s.pivot(index="series_id", columns="week", values="y")
        .loc[order]
        .values.astype(float)
    )
    av = (
        a.pivot(index="series_id", columns="week", values="is_available")
        .loc[order]
        .values.astype(float)
    )
    at = at.set_index("series_id").loc[order]
    _DATA_CACHE["raw"] = (sv, av, at)
    return _DATA_CACHE["raw"]


# --------------------------------------------------------------------------
# 1. censoring correction
# --------------------------------------------------------------------------
def _correct(sv, av, H):
    """Replace out-of-stock weeks (observed sales == 0) by a trailing local level."""
    D = sv[:, :H].copy()
    A = av[:, :H]
    H = D.shape[1]
    w = np.exp(-np.arange(IMPUT_K - 1, -1, -1) / (IMPUT_K / 2.0))
    for t in range(H):
        lo = max(0, t - IMPUT_K + 1)
        vals = D[:, lo : t + 1]
        msk = A[:, lo : t + 1] == 1
        ww = w[t - lo : t + 1][None, :]
        den = (ww * msk).sum(axis=1)
        num = (vals * ww * msk).sum(axis=1)
        glob = vals.sum() / max(msk.sum(), 1)
        lvl = np.where(den > 0, num / np.maximum(den, 1e-9), glob)
        D[:, t] = np.where(A[:, t] == 1, D[:, t], lvl)
    return D


def _trailing_mean(x, K, weighted=True):
    k = min(K, x.shape[1])
    v = x[:, -k:]
    if weighted:
        w = np.exp(-np.arange(k - 1, -1, -1) / (k / 2.0))
        return (v * w).sum(axis=1) / w.sum()
    return v.mean(axis=1)


def _trailing_means(x, K, weighted=False):
    n, T = x.shape
    k = min(K, T)
    out = np.zeros((n, T))
    for t in range(T):
        lo = max(0, t - k + 1)
        v = x[:, lo : t + 1]
        if weighted:
            w = np.exp(-np.arange(v.shape[1] - 1, -1, -1) / (v.shape[1] / 2.0))
            out[:, t] = (v * w).sum(axis=1) / w.sum()
        else:
            out[:, t] = v.mean(axis=1)
    return out


# --------------------------------------------------------------------------
# 2. hierarchical forecast components
# --------------------------------------------------------------------------
def _components(H):
    key = ("cmp", H, FAMILY_COL, LEVEL_K, IMPUT_K, SEAS_K, SHARE_K, SHRINK_LAM)
    if key in _FC_CACHE:
        return _FC_CACHE[key]
    sv, av, at = _load()
    n = sv.shape[0]
    D = _correct(sv, av, H)

    fam_key = at[FAMILY_COL].astype(str).values
    _, fam_idx = np.unique(fam_key, return_inverse=True)
    n_fam = int(fam_idx.max()) + 1

    lvl_i = _trailing_mean(D, LEVEL_K)  # (n,)   series level
    DF = np.zeros((n_fam, H))
    np.add.at(DF, fam_idx, D)
    lvl_f = _trailing_mean(DF, LEVEL_K)  # (n_fam,) family level

    # --- weekly seasonal shape: demand / trailing level, averaged per week-of-year
    lvl_i_hist = _trailing_means(D, SEAS_K)
    lvl_f_hist = _trailing_means(DF, SEAS_K)
    eps = 1e-6
    sh_i = D / np.maximum(lvl_i_hist, eps)
    sh_f = DF / np.maximum(lvl_f_hist, eps)
    woy = np.arange(H) % 52
    idx_f = np.ones((n_fam, 52))
    idx_i = np.ones((n, 52))
    for w in range(52):
        m = woy == w
        if m.sum() == 0:
            continue
        idx_f[:, w] = sh_f[:, m].mean(axis=1)
        idx_i[:, w] = sh_i[:, m].mean(axis=1)
    idx_f = 1.0 + (idx_f - 1.0) * (n_fam / (n_fam + SHRINK_LAM))
    idx_f = np.clip(idx_f, 0.15, 3.0)
    volume = D.sum(axis=1)
    wgt = (volume / (volume + SHRINK_LAM * max(volume.mean(), 1e-6)))[:, None]
    idx_i = np.clip(wgt * idx_i + (1 - wgt) * idx_f[fam_idx], 0.15, 3.0)

    # --- bottom-up allocation shares ---------------------------------
    K_SH = max(SHARE_K, 4)
    sum_i = D[:, -K_SH:].sum(axis=1)
    sum_f = DF[:, -K_SH:].sum(axis=1)
    cnt = np.bincount(fam_idx, minlength=n_fam).astype(float)
    prior_f = 0.15 * sum_f / np.maximum(cnt, 1)
    share = (sum_i + prior_f[fam_idx]) / (sum_f[fam_idx] + prior_f[fam_idx])

    comp = dict(
        D=D,
        lvl_i=lvl_i,
        lvl_f=lvl_f,
        idx_i=idx_i,
        idx_f=idx_f,
        share=share,
        fam_idx=fam_idx,
        n_fam=n_fam,
        n=n,
        H=H,
    )
    _FC_CACHE[key] = comp
    return comp


def _assemble(comp, seas_w=None, family_w=None):
    """Build the (8, n) forecast of the 8 game weeks from the components."""
    seas_w = SEAS_W if seas_w is None else seas_w
    family_w = FAMILY_W if family_w is None else family_w
    H, n = comp["H"], comp["n"]
    lvl_i, lvl_f = comp["lvl_i"], comp["lvl_f"]
    idx_i, idx_f = comp["idx_i"], comp["idx_f"]
    share, fam_idx = comp["share"], comp["fam_idx"]
    F = np.zeros((8, n))
    for g in range(8):
        w = (H + g) % 52
        own = (1 - seas_w) * lvl_i + seas_w * (lvl_i * idx_i[:, w])
        alloc = (lvl_f * idx_f[:, w])[fam_idx] * share
        F[g] = (1 - family_w) * own + family_w * alloc
    return np.maximum(F, 0.0)


# --------------------------------------------------------------------------
# 3. per-series demand dispersion (negative binomial size)
# --------------------------------------------------------------------------
def _dispersion(D):
    k = min(26, D.shape[1])
    v = D[:, -k:]
    m = v.mean(axis=1)
    t4 = np.empty_like(v)
    for i in range(v.shape[1]):
        t4[:, i] = v[:, max(0, i - 3) : i + 1].mean(axis=1)
    resid = v - t4
    v_d = 1.25 * (resid * resid).mean(axis=1)
    r = np.where(v_d > m * 1.05, m * m / np.maximum(v_d - m, 1e-6), 40.0)
    r = np.where(m <= 0.02, 40.0, r)
    return np.clip(r, 0.3, 30.0)


# --------------------------------------------------------------------------
# 4. adaptation to revealed in-game demand
# --------------------------------------------------------------------------
def _adapt(F, w, revealed):
    if w <= 0 or revealed is None:
        return F
    rev = np.asarray(revealed, dtype=float)
    if rev.ndim != 2 or rev.shape[0] == 0:
        return F
    k = min(w, 4)
    obs = rev[-k:].sum(axis=0)
    pred = F[:w][-k:].sum(axis=0)
    ok = (obs >= ADAPT_MIN) & (pred > 0.3)
    ratio = np.where(ok, obs / np.maximum(pred, 1e-9), 1.0)
    ratio = np.clip(ratio, ADAPT_CLIP[0], ADAPT_CLIP[1])
    lam = np.where(ratio > 1.0, ADAPT_LAM_UP, ADAPT_LAM_DN)
    adj = 1.0 + lam * (ratio - 1.0)
    out = F.copy()
    out[w:] = out[w:] * adj[None, :]
    return out


# --------------------------------------------------------------------------
# 5. the policy
# --------------------------------------------------------------------------
def _orders(weekly, r, state, w, rng, fractile=None, n_paths=None):
    """Simulation-based newsvendor on the critical week (w+2).

    `weekly` is the (3, n) array of mean demands for weeks w, w+1, w+2.
    The stock that will still be on hand when the order arrives is simulated
    together with the critical week's demand, so the order is the fractile of
    (demand - stock still available) -> never negative.
    """
    fractile = FRACTILE if fractile is None else fractile
    n_paths = N_PATHS if n_paths is None else n_paths
    n = state.shape[0]
    on_hand, arr1, arr2 = state[:, 0], state[:, 1], state[:, 2]

    def draw(mean):
        p = r / (r + np.maximum(mean, 1e-9))
        return rng.negative_binomial(r, p, size=(n_paths, n)).astype(float)

    d0 = draw(weekly[0])
    d1 = draw(weekly[1])
    d2 = draw(weekly[2])
    end0 = np.maximum(0.0, on_hand + arr1 - d0)
    end1 = np.maximum(0.0, end0 + arr2 - d1)
    need = d2 - end1
    q = np.quantile(need, fractile, axis=0)
    return np.maximum(0.0, q)


def decide(week_index, state, revealed):
    """week_index: int 0..7, state: (n,3), revealed: (week_index, n) -> (n,) orders."""
    w = int(week_index)
    state = np.asarray(state, dtype=float)
    revealed = np.asarray(revealed, dtype=float) if revealed is not None else None
    n = state.shape[0]

    if w > 5:
        # weeks 6-8 have no order slot (an order would arrive after the game ends)
        return np.zeros(n, dtype=float)

    H = HISTORY_END
    if H is None:
        sv, av, at = _load()
        H = sv.shape[1]
    comp = _components(int(H))
    F = _assemble(comp)
    if F.shape[1] != n:  # synthetic game with a different number of series
        F = F[:, :n] if F.shape[1] > n else np.repeat(F[:, :1], n, axis=1)
    F = _adapt(F, w, revealed)

    r = _dispersion(comp["D"])
    if r.shape[0] != n:
        r = np.full(n, 5.0)

    rng = np.random.default_rng(1234 + w)
    q = _orders(F[w : w + 3], r, state, w, rng)
    q = np.where(np.isfinite(q), q, 0.0)
    return q
