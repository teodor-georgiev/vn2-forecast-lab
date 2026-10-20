"""VN2 round-3 adaptive ordering policy (worker w05).

Method, three lines
-------------------
1. TSB (Teunter-Syntetos-Babaie, i.e. Croston-family) intermittent-demand
   smoothing per series: occurrence probability p and positive-demand size z,
   initialised from the censoring-corrected history in data\\sales.csv and
   re-run over the demand revealed inside the game every round; rate = p * z.
2. Predictive distribution of a k-week demand sum: negative binomial with mean
   k * rate and dispersion fitted from the empirical variance of k-week rolling
   sums of the (censoring-corrected) history.
3. Base-stock (order-up-to) with the competition's cost ratio: the order makes
   the inventory position reach the P_TARGET fractile of the *scored* demand
   inside the position's 3-week window (weeks w, w+1, w+2); the unscored
   settling weeks 1-2 may consume the position for free (depletion taken at
   fractile P_DEP), so they only reduce what is left for the scored weeks.

Series-order robustness: the manager may pass series in any row order, so the
order is recovered from the week-0 state against initial_state.csv (exact match)
or, failing that, from a rank correlation between revealed demand and each
candidate history row order.  Unknown n (interface check) falls back to a
pooled prior, still fully adaptive through the revealed demand.
"""

from __future__ import annotations

import numpy as np

# ---------------- tunables (tuned by tune.py on historical cutoffs) ----------
P_TARGET = 0.60  # fractile of the scored 3-week demand sum (cost ratio 1.00/0.20)
P_DEP = 0.80  # fractile used for free depletion by the unscored weeks 1-2
ALPHA = 0.15  # TSB occurrence-probability smoothing
BETA = 0.30  # TSB positive-size smoothing
SCALE = 1.15  # multiplier on the order-up-to target
CENS_IMPUTE = True  # impute stock-out (censored) weeks at the series' 75th pct
RECENT_W = 52  # recency window mixed into the historical level prior
R_MIN, R_MAX = 0.05, 500.0

_MODEL = None
_LOCK = None  # per-series history mapping confirmed at week_index 0


# ---------------------------------------------------------------- data -----
def _build_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import warnings
    from pathlib import Path

    import pandas as pd

    m = {"ok": False, "n": -1, "cands": [np.arange(0)], "pooled": True}
    try:
        d = Path(__file__).resolve().parents[2] / "data"
        sales = pd.read_csv(d / "sales.csv")
        avail = pd.read_csv(d / "availability.csv")
        y = sales.pivot_table(
            index="series_id", columns="week", values="y", aggfunc="first"
        ).sort_index(axis=1)
        a = (
            avail.pivot_table(
                index="series_id",
                columns="week",
                values="is_available",
                aggfunc="first",
            )
            .reindex(index=y.index, columns=y.columns)
            .fillna(True)
        )
        Y = np.nan_to_num(y.values.astype(float), nan=0.0)
        AV = a.values.astype(bool)
        n, T = Y.shape

        # censoring correction: a week with availability==0 was a stock-out, so
        # unobserved demand was positive; impute it at the series' 75th
        # percentile of observed positive demand (stock-outs skew high).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q75 = np.nanquantile(np.where(AV & (Y > 0), Y, np.nan), 0.75, axis=1)
        q75 = np.nan_to_num(q75, nan=0.0)
        Yadj = np.where(AV, Y, q75[:, None]) if CENS_IMPUTE else np.where(AV, Y, 0.0)

        # --- level prior: full-history mean mixed with the recent window -----
        lam_full = Yadj.mean(1)
        lam_rec = Yadj[:, max(0, T - RECENT_W) :].mean(1)
        lam = 0.5 * (lam_full + lam_rec)

        # --- TSB initial state: occurrence and positive size -----------------
        occ = (Yadj > 0).mean(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zmean = np.nanmean(np.where(Yadj > 0, Yadj, np.nan), axis=1)
        zmean = np.nan_to_num(zmean, nan=0.0)
        pos_pool = Yadj[Yadj > 0]
        z_pool = float(pos_pool.mean()) if pos_pool.size else 1.0
        p_pool = float((Yadj > 0).mean()) if Yadj.size else 0.5
        p0 = np.where(lam > 0, np.clip(occ, 0.02, 1.0), 0.05)
        z0 = np.where(zmean > 0, zmean, z_pool)

        # --- dispersion of k-week demand sums (negative binomial size) -------
        cs = np.concatenate([np.zeros((n, 1)), np.cumsum(Yadj, axis=1)], axis=1)
        rks = {}
        for k in (1, 2, 3):
            S = cs[:, k:] - cs[:, :-k]  # (n, T-k+1) rolling k-sums
            mk = S.mean(1)
            den = S.var(1) - mk
            rk = np.where(den > 1e-9, mk**2 / np.maximum(den, 1e-9), R_MAX)
            rk = np.where(mk > 1e-9, rk, R_MAX)
            rks[k] = np.clip(rk, R_MIN, R_MAX)

        # --- initial state file (per series, same sorted order as the pivot) --
        ini = pd.read_csv(d / "initial_state.csv").set_index("series_id")
        INIT = np.nan_to_num(
            ini.reindex(y.index)[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(
                float
            ),
            nan=0.0,
        )

        # --- candidate series orders the manager might use -------------------
        rank = {s: i for i, s in enumerate(y.index)}
        cands = [np.arange(n)]

        def _add(ids):
            if len(ids) == n and all(s in rank for s in ids):
                perm = np.array([rank[s] for s in ids])
                if not any(np.array_equal(perm, c) for c in cands):
                    cands.append(perm)

        _add(sales["series_id"].drop_duplicates().tolist())
        try:
            att = pd.read_csv(d / "product_attributes.csv")
            if "series_id" in att.columns:
                _add(att["series_id"].drop_duplicates().tolist())
        except Exception:
            pass

        m.update(
            ok=True,
            n=n,
            Y=Y,
            AV=AV,
            lam=lam,
            p0=p0,
            z0=z0,
            rks=rks,
            INIT=INIT,
            cands=cands,
            pooled=False,
            lam_pool=float(np.mean(lam)),
            p_pool=p_pool,
            z_pool=z_pool,
            r_pool={k: float(np.median(rks[k])) for k in (1, 2, 3)},
        )
    except Exception:
        m["pooled"] = True
    _MODEL = m
    return m


# ------------------------------------------------------------ selection ----
def _spearman(a, b):
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return -1.0
    try:
        from scipy.stats import spearmanr

        r = spearmanr(a, b)
        r = getattr(r, "statistic", None)
        if r is None:
            r = spearmanr(a, b).correlation
    except Exception:
        return -1.0
    return float(r) if np.isfinite(r) else -1.0


def _select(m, w, st, revealed, n):
    """Return (mapping game-index -> history row, use_pooled_prior)."""
    global _LOCK
    if w == 0:
        _LOCK = None
        if m["ok"] and n == m["n"] and st is not None and st.shape == m["INIT"].shape:
            for cand in m["cands"]:
                if np.all(np.abs(st - m["INIT"][cand]) <= 1e-6, axis=1).mean() > 0.95:
                    _LOCK = cand
                    break
    if _LOCK is not None and n == m["n"]:
        return _LOCK, False
    if m["ok"] and n == m["n"] and revealed is not None:
        D = np.asarray(revealed, dtype=float)
        if D.ndim == 2 and D.shape[0] > 0 and D.shape[1] == n:
            rv = np.log1p(np.clip(D.mean(axis=0), 0.0, None))
            best, bestsc = m["cands"][0], -9.0
            for cand in m["cands"]:
                sc = _spearman(rv, np.log1p(m["lam"][cand]))
                if sc > bestsc + 0.03:
                    best, bestsc = cand, sc
            return best, False
    return (m["cands"][0], False) if (m["ok"] and n == m["n"]) else (None, True)


def _q_nb(mu, r, q):
    """q-quantile of a negative binomial with the given mean and size r."""
    out = np.zeros_like(mu)
    good = mu > 1e-9
    if np.any(good):
        from scipy.stats import nbinom

        rr = np.clip(r[good], R_MIN, R_MAX)
        out[good] = nbinom.ppf(q, rr, rr / (rr + mu[good]))
    return np.nan_to_num(out, nan=0.0)


# -------------------------------------------------------------- decide -----
def decide(week_index, state, revealed):
    """Order-up-to policy; see module docstring for the method."""
    m = _MODEL if _MODEL is not None else _build_model()
    st = np.asarray(state, dtype=float).reshape(-1, 3)
    n = st.shape[0]
    w = int(week_index)
    mapping, pooled = _select(m, w, st, revealed, n)

    # ---- per-series TSB state, re-run over the revealed game demand ---------
    if pooled:
        p = np.full(n, m["p_pool"])
        z = np.full(n, m["z_pool"])
        rk = {k: np.full(n, m["r_pool"][k]) for k in (1, 2, 3)}
    else:
        p = m["p0"][mapping].copy()
        z = m["z0"][mapping].copy()
        rk = {k: m["rks"][k][mapping] for k in (1, 2, 3)}
    D = np.asarray(revealed, dtype=float) if revealed is not None else None
    if D is not None and D.ndim == 2 and D.shape[0] > 0:
        D = D.reshape(D.shape[0], -1)
        a, b = ALPHA, BETA
        for t in range(D.shape[0]):
            d = D[t]
            nz = d > 0
            p = (1.0 - a) * p + a * nz
            z = np.where(nz, (1.0 - b) * z + b * np.maximum(d, 0.0), z)
    lam = np.clip(p * z, 0.0, None)  # TSB rate: expected demand/week

    # ---- scored weeks covered by this position -----------------------------
    k = max(0, min(w + 2, 7) - max(w, 2) + 1)  # scored weeks in [w, w+2]
    u = max(0, min(w + 2, 1) - w + 1)  # unscored weeks in window
    Q = SCALE * _q_nb(k * lam, rk[k] if k in rk else rk[1], P_TARGET)

    # ---- stock surviving into the scored weeks -----------------------------
    A = st[:, 0] + st[:, 1]  # on hand + delivery during this week
    B = st[:, 2]  # delivery during next week
    if u > 0:
        Qd = _q_nb(lam, rk[1], P_DEP)
    else:
        Qd = None
    if w == 0:
        surv = np.maximum(0.0, np.maximum(0.0, A - Qd) + B - Qd)
    elif w == 1:
        surv = np.maximum(0.0, A - Qd) + B
    else:
        surv = A + B

    q = np.maximum(0.0, Q - surv)
    return np.clip(np.nan_to_num(q, nan=0.0, posinf=0.0), 0.0, 1e6)
