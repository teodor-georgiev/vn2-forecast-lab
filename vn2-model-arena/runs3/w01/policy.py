"""VN2 adaptive ordering policy - empirical per-series newsvendor quantiles.

Method
------
1. Censoring: history (157 weekly sales per series, availability flagged) is
   reduced to a per-series empirical demand sample using only weeks with
   availability == 1, so weeks that were sold out (unobserved higher demand) do
   not drag the level down.
2. Just-in-time base stock.  An order placed at week_index w arrives at week
   w+2 and is the LAST chance to cover week w+2 (every later week can still be
   ordered later, cheaper and with more information revealed).  So each week we
   raise the expected stock position for week w+2 up to a newsvendor quantile
   q_tau of that series' weekly demand:
       order_w = max(0, q_tau_i - E[stock left for week w+2 | no new order])
   where the expectation is taken over the series' own empirical demand sample.
3. tau is chosen per series on simulated cost: the 8-week game is replayed on
   demand paths drawn from that series' empirical distribution and the tau with
   the lowest simulated cost (1.00 per missed unit, 0.20 per held unit-week,
   weeks 3-8) is kept.

Series alignment: the manager's series order is unknown.  At week_index 0 the
incoming state is matched against data/initial_state.csv (exact permutation
when possible, otherwise parameters averaged over all history series sharing
the same (on_hand, arrival_w1, arrival_w2) triple).  For a series count that
does not match the arena history, a generic empirical rule from the revealed
demand is used instead.

Reads only data/ (read-only), writes nothing.
"""

from __future__ import annotations

import numpy as np

CFG = {
    "tau_grid": (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    "tau_mode": "sim",  # "sim" (per series) or "global"
    "tau_global": 0.85,
    "sim_paths": 256,
    "sim_pairs": 32,
    "sample_cap": 96,
    "censor": "avail_only",  # or "as_is"
    "leftover": "exact",  # "exact" | "mean"
    "adapt_k": 0.0,  # >0 shrinks the target towards the in-game level
    "seed": 20261019,
}

_PARAMS = None
_MAP = None
_RAW = None
_Z90 = 1.2816


def _data_dir():
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for cand in (here.parent.parent / "data", here.parent.parent.parent / "data"):
        if cand.is_dir():
            return cand
    raise FileNotFoundError("arena data/ not found")


def _load_history(cutoff=None):
    global _RAW
    if _RAW is None:
        _RAW = _read_csvs()
    S, A, init, ids = _RAW
    if cutoff is not None:
        S, A = S[:, :cutoff], A[:, :cutoff]
    return S, A, init, ids


def _read_csvs():
    import pandas as pd

    d = _data_dir()
    sales = pd.read_csv(d / "sales.csv")
    avail = pd.read_csv(d / "availability.csv")
    S = sales.pivot(index="series_id", columns="week", values="y").sort_index()
    A = avail.pivot(
        index="series_id", columns="week", values="is_available"
    ).sort_index()
    ids = list(S.index)
    A = A.reindex(ids).to_numpy(dtype=bool)
    S_arr = S.to_numpy(dtype=float)
    try:
        ini = pd.read_csv(d / "initial_state.csv").set_index("series_id").reindex(ids)
        init = ini[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)
        init = np.nan_to_num(init, nan=0.0)
    except Exception:
        init = np.zeros((S_arr.shape[0], 3))
    return S_arr, A, init, ids


def _sample_matrix(D, cap, rng):
    """(n, cap) bootstrap draws from each series' uncensored demand sample."""
    n = D.shape[0]
    out = np.zeros((n, cap))
    for i in range(n):
        v = D[i][np.isfinite(D[i])]
        if v.size == 0:
            continue
        if v.size >= cap:
            out[i] = v[rng.choice(v.size, size=cap, replace=False)]
        else:
            out[i] = v[rng.integers(0, v.size, size=cap)]
    return out


def _leftover_mean(X, arr2, samp_a, samp_b, mode):
    """E[stock left for week w+2] given current deterministic state."""
    if mode == "mean":
        mu2 = np.nanmean(samp_a, axis=1) + np.nanmean(samp_b, axis=1)
        return np.maximum(0.0, X + arr2 - mu2)
    lo1 = np.maximum(0.0, X[:, None] - samp_a)
    lo2 = np.maximum(0.0, lo1 + arr2[:, None] - samp_b)
    out = np.nanmean(lo2, axis=1)
    return np.nan_to_num(out, nan=0.0)


def _simulate_taus(samp_a, samp_b, init_units, taus, paths, pairs, rng):
    """Average scored cost (weeks 3-8) per (series, tau) for the JIT policy."""
    n, cap = samp_a.shape
    Dp = np.empty((n, paths, 8))
    for p in range(8):
        idx = rng.integers(0, cap, size=(n, paths))
        vals = np.take_along_axis(np.where(np.isnan(samp_a), 0.0, samp_a), idx, axis=1)
        Dp[:, :, p] = vals
    # scenario pairs for the expectation, shared across paths and taus
    ia = rng.integers(0, cap, size=(n, pairs))
    ib = rng.integers(0, cap, size=(n, pairs))
    Sa = np.take_along_axis(samp_a, ia, axis=1)
    Sb = np.take_along_axis(samp_b, ib, axis=1)
    Sa = np.where(np.isnan(Sa), np.nanmean(samp_a, axis=1)[:, None], Sa)
    Sb = np.where(np.isnan(Sb), np.nanmean(samp_b, axis=1)[:, None], Sb)

    qmat = np.empty((n, taus.size))
    for k, t in enumerate(taus):
        qmat[:, k] = np.nanquantile(samp_a, t, axis=1)
    qmat = np.nan_to_num(qmat, nan=0.0)

    on_hand = np.repeat(init_units[:, None, None], paths, axis=1)
    on_hand = np.repeat(on_hand, taus.size, axis=2)
    arr1 = np.zeros((n, paths, taus.size))
    arr2 = np.zeros((n, paths, taus.size))
    cost = np.zeros((n, taus.size))
    for w in range(8):
        d = Dp[:, :, w][:, :, None]
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - d)
        missed = np.maximum(0.0, d - supply)
        if w >= 2:
            cost += (0.20 * end + 1.00 * missed).sum(axis=1)
        order = np.zeros((n, paths, taus.size))
        if w < 6:
            X = end
            lo1 = np.maximum(0.0, X[:, :, :, None] - Sa[:, None, None, :])
            lo2 = np.maximum(0.0, lo1 + arr2[:, :, :, None] - Sb[:, None, None, :])
            El = np.nanmean(lo2, axis=3)
            order = np.maximum(0.0, qmat[:, None, :] - El)
            order = np.nan_to_num(order, nan=0.0)
        on_hand, arr1, arr2 = end, arr2, order
    return cost / paths


def fit(cutoff=None, cfg=None):
    """Fit per-series empirical demand distributions (and per-series tau)."""
    global _PARAMS, _MAP
    c = dict(CFG)
    if cfg:
        c.update(cfg)
    S, A, init, ids = _load_history(cutoff)
    n = S.shape[0]
    rng = np.random.default_rng(int(c["seed"]))

    if c["censor"] == "avail_only":
        D = np.where(A, S, np.nan)
    else:
        D = np.where(A, S, 0.0)
    cap = int(c["sample_cap"])
    samp_a = _sample_matrix(D, cap, rng)
    samp_b = _sample_matrix(D, cap, rng)

    taus = np.asarray(c["tau_grid"], dtype=float)
    init_units = np.nansum(np.nan_to_num(init), axis=1)

    if c["tau_mode"] == "sim":
        paths = int(c["sim_paths"])
        cost = _simulate_taus(
            samp_a, samp_b, init_units, taus, paths, int(c["sim_pairs"]), rng
        )
        tau_idx = np.argmin(cost, axis=1)
    else:
        tau_idx = np.full(n, int(np.argmin(np.abs(taus - float(c["tau_global"])))))

    mu = np.nanmean(samp_a, axis=1)
    mu = np.nan_to_num(mu, nan=0.0)
    qmat = np.empty((n, taus.size))
    for k, t in enumerate(taus):
        qmat[:, k] = np.nanquantile(samp_a, t, axis=1)
    qmat = np.nan_to_num(qmat, nan=0.0)

    groups = {}
    r0 = np.round(init, 6)
    for i in range(n):
        groups.setdefault(tuple(r0[i]), []).append(i)

    _PARAMS = {
        "cutoff": cutoff,
        "cfg": c,
        "n": n,
        "ids": ids,
        "init": init,
        "mu": mu,
        "samp_a": samp_a,
        "samp_b": samp_b,
        "taus": taus,
        "qmat": qmat,
        "tau_idx": tau_idx,
        "groups": groups,
        "cost": cost if c["tau_mode"] == "sim" else None,
    }
    _MAP = None
    return _PARAMS


def _resolve(state0, n_game):
    """Per-game-series parameter arrays (mu, samp_a, samp_b, q_tau)."""
    global _MAP
    P = _PARAMS
    tri = np.round(np.asarray(state0, dtype=float)[:, :3], 6)
    if _MAP is not None and _MAP["n_game"] == n_game and np.allclose(_MAP["tri"], tri):
        return _MAP

    n = P["n"]
    idx = None
    if n == n_game and np.allclose(np.round(P["init"], 6), tri):
        idx = np.arange(n)  # file order already matches
    if idx is None:
        # row-wise match on the (on_hand, arr1, arr2) triple
        rows_a = np.full(n_game, -1)
        rows_b = np.full(n_game, -1)
        for j in range(n_game):
            f = np.flatnonzero(np.all(np.round(P["init"], 6) == tri[j], axis=1))
            if f.size:
                rows_a[j], rows_b[j] = f[0], f[-1]
        idx = rows_a
        alt = rows_b

    if np.all(idx >= 0):
        mu = P["mu"][idx]
        sa = P["samp_a"][idx]
        sb = P["samp_b"][idx]
        qi = P["tau_idx"][idx]
        taus = P["taus"]
        q = P["qmat"][idx][np.arange(n_game), qi]
        # same-triple ambiguity: for duplicated groups average the two ends we found
        if not (n == n_game and np.array_equal(idx, np.arange(n))):
            dup = alt != idx
            if dup.any():
                mu = mu.copy()
                q = q.copy()
                mu[dup] = 0.5 * (P["mu"][idx[dup]] + P["mu"][alt[dup]])
                q[dup] = 0.5 * (q[dup] + P["qmat"][alt[dup], qi[dup]])
        out = {
            "n_game": n_game,
            "tri": tri,
            "mu": mu,
            "sa": sa,
            "sb": sb,
            "q": q,
            "tau_i": taus[qi],
            "ok": True,
        }
    else:
        out = {"n_game": n_game, "tri": tri, "ok": False}
    _MAP = out
    return out


_Z_TABLE = (
    (0.50, 0.0),
    (0.60, 0.2533),
    (0.70, 0.5244),
    (0.75, 0.6745),
    (0.80, 0.8416),
    (0.85, 1.0364),
    (0.90, 1.2816),
    (0.95, 1.6449),
)


def _zfac(tau):
    """Nearest normal quantile for the fallback rule."""
    best, bd = 0.8416, 9.9
    for t, z in _Z_TABLE:
        if abs(t - tau) < bd:
            best, bd = z, abs(t - tau)
    return best


def decide(week_index, state, revealed):
    global _PARAMS
    if _PARAMS is None:
        fit()
    P = _PARAMS
    state = np.asarray(state, dtype=float)
    n_game = state.shape[0]
    taus = P["taus"]
    cfg = P["cfg"]
    if n_game != P["n"]:
        return _generic_orders(state, revealed, taus, cfg, week_index)
    R = _resolve(np.asarray(state, dtype=float), n_game)
    if not R["ok"]:
        return _generic_orders(state, revealed, taus, cfg, week_index)
    return _orders(week_index, state, revealed, R["q"], R["mu"], R["sa"], R["sb"], cfg)


def _orders(week_index, state, revealed, q, mu, sa, sb, cfg):
    on_hand, arr1, arr2 = state.T
    X = on_hand + arr1
    target = q.copy()
    k = float(cfg.get("adapt_k", 0.0))
    if k > 0 and week_index > 0:
        r = np.asarray(revealed, dtype=float).reshape(week_index, -1)
        mult = (r.sum(axis=0) + k * mu * week_index) / (
            np.maximum(mu, 1e-9) * (week_index + k)
        )
        target = target * np.clip(mult, 0.25, 4.0)
    left = _leftover_mean(X, arr2, sa, sb, cfg.get("leftover", "exact"))
    order = np.maximum(0.0, target - left)
    return np.nan_to_num(order, nan=0.0, posinf=0.0, neginf=0.0)


def _generic_orders(state, revealed, taus, cfg, week_index):
    """Fallback for an unknown series count: empirical rule from revealed demand."""
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    on_hand, arr1, arr2 = state.T
    X = on_hand + arr1
    r = (
        np.asarray(revealed, dtype=float).reshape(week_index, n)
        if week_index
        else np.zeros((0, n))
    )
    tau = float(cfg.get("tau_global", 0.85))
    if r.shape[0] >= 2:
        mu = r.mean(axis=0)
        sd = r.std(axis=0)
    elif r.shape[0] == 1:
        mu = r[0]
        sd = 0.5 * mu
    else:
        mu = np.maximum((on_hand + arr1 + arr2) / 8.0, 0.25)
        sd = mu
    q = np.maximum(mu + _zfac(tau) * np.maximum(sd, 1e-9), 0.0)
    left = np.maximum(0.0, X + arr2 - 2.0 * mu)
    order = np.maximum(0.0, q - left)
    return np.nan_to_num(order, nan=0.0)
