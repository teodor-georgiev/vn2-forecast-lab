"""VN2 adaptive ordering policy.

Strategy theme
--------------
Per-series stochastic demand-scenario optimiser with a rolling horizon and
per-week coordinate descent:

* every series gets its own demand model: a *level* taken from a recent
  window, a *shape* (intermittency / spread) bootstrapped from a longer
  empirical pool, availability-corrected for censored (sold-out) weeks,
* a seasonal shape is estimated per series (own deviation shrunk towards the
  pooled seasonal profile) and used to move the level from the recent window
  to the weeks we are about to play,
* at each decision week we draw a small ensemble of demand paths for the weeks
  that are still ahead and optimise the *vector* of remaining order quantities
  by coordinate descent (each order lands exactly two weeks later) against the
  simulated cost  1.00 * missed + 0.20 * end-inventory,
* only the first slot (the order that is actually placed now) is committed; the
  whole plan is re-optimised next week with the freshly revealed demand, so the
  policy adapts to whatever the revealed history says.

The policy is self-contained: it reads only the arena `data/` folder, writes no
files and uses no network.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- config
SHORT = 1.00  # EUR per missed unit
HOLD = 0.20  # EUR per unit held at the end of a week
N_PATHS = int(os.environ.get("VN2_NPATHS", 32))
POOL_WEEKS = int(os.environ.get("VN2_POOL_WEEKS", 52))
LEVEL_WEEKS = int(os.environ.get("VN2_LEVEL_WEEKS", 13))
LEVEL_SHRINK = float(os.environ.get("VN2_LEVEL_SHRINK", 13.0))
SEASON_POW = float(os.environ.get("VN2_SEASON_POW", 0.7))
OWN_POW = float(os.environ.get("VN2_OWN_POW", 1.0))
SEASON_CLIP = (0.4, 2.5)
CAND_P = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
N_PASS = int(os.environ.get("VN2_NPASS", 2))
REFINE = int(os.environ.get("VN2_REFINE", 1))
CORRECT_THETA = float(os.environ.get("VN2_THETA", 0.6))
SHRINK_PRIOR = float(os.environ.get("VN2_SHRINK", 1.0))
LEVEL_CLIP = (
    float(os.environ.get("VN2_FB_LO", 0.5)),
    float(os.environ.get("VN2_FB_HI", 2.0)),
)
FEEDBACK_POW = float(os.environ.get("VN2_FBPOW", 1.0))

_MODEL = None


# ------------------------------------------------------------------- helpers
def _data_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent.parent / "data", here.parent / "data", here / "data"):
        if cand.exists():
            return cand
    raise FileNotFoundError("arena data folder not found")


def _correct_censored(Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Replace availability-censored sales with a plausible true demand.

    A week with availability == 0 is a stock-out: the recorded sales are the
    amount that was on the shelf, the true demand was (slightly) higher.  We
    impute the upper quartile of the series' observed (available) demand and
    shrink the correction back towards the raw observation.
    """
    n = Y.shape[0]
    q75 = np.zeros(n)
    for i in range(n):
        vals = Y[i][A[i]]
        if vals.size == 0:
            q75[i] = 0.0
        else:
            s = np.sort(vals)
            q75[i] = s[int(np.floor(0.75 * (s.size - 1)))]
    imp = np.where(A, Y, np.maximum(Y, q75[:, None]))
    return np.where(A, Y, Y + CORRECT_THETA * (imp - Y))


def _seasonal(D: np.ndarray, A: np.ndarray, woy: np.ndarray):
    """Per-series seasonal shape (mean 1 over the year) + pooled profile."""
    n, T = D.shape
    # pooled profile over series (available weeks only)
    num = np.zeros(54)
    den = np.zeros(54)
    for k in range(1, 54):
        m = woy == k
        num[k] = np.where(A[:, m], D[:, m], 0.0).sum()
        den[k] = A[:, m].sum()
    prof = np.ones(54)
    prof[1:54] = np.where(den[1:54] > 0, num[1:54] / np.maximum(den[1:54], 1), 1.0)
    prof[1:54] /= max(prof[1:54].mean(), 1e-9)
    prof[0] = 1.0

    # per-series raw ratio to its own annual mean
    s_num = np.zeros((n, 54))
    s_den = np.zeros((n, 54))
    for k in range(1, 54):
        m = woy == k
        s_num[:, k] = np.where(A[:, m], D[:, m], 0.0).sum(axis=1)
        s_den[:, k] = A[:, m].sum(axis=1)
    ann = np.where(A, D, 0.0).sum(axis=1) / np.maximum(A.sum(axis=1), 1)
    den_ok = np.maximum(s_den, 1.0)
    raw = np.where(
        s_den > 0,
        s_num / den_ok / np.maximum(ann, 1e-9)[:, None],
        1.0,
    )
    # shrink the noisy own estimate towards 1 (3 years of data per week-of-year)
    wgt = s_den / (s_den + 3.0)
    own = 1.0 + (raw - 1.0) * (wgt * OWN_POW)
    shape = own * prof[None, :]
    shape[:, 0] = 1.0
    return shape, prof


def _build_model(cutoff: int | None = None):
    import pandas as pd

    d = _data_dir()
    sales = pd.read_csv(d / "sales.csv")
    avail = pd.read_csv(d / "availability.csv")
    ids = np.array(sorted(sales["series_id"].astype(str).unique()))
    weeks = np.array(sorted(sales["week"].astype(str).unique()))
    n, T = len(ids), len(weeks)
    si = {k: i for i, k in enumerate(ids)}
    wi = {k: i for i, k in enumerate(weeks)}

    Y = np.zeros((n, T), dtype=float)
    A = np.zeros((n, T), dtype=bool)
    r = sales["series_id"].astype(str).map(si).to_numpy()
    c = sales["week"].astype(str).map(wi).to_numpy()
    Y[r, c] = sales["y"].to_numpy(dtype=float)
    r = avail["series_id"].astype(str).map(si).to_numpy()
    c = avail["week"].astype(str).map(wi).to_numpy()
    A[r, c] = avail["is_available"].to_numpy(dtype=bool)

    D = _correct_censored(Y, A)

    # calendar (week-of-year), extended by 8 weeks for the game horizon
    dates = pd.to_datetime(pd.Series(weeks), format="%Y-%m-%d")
    ext = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=7), periods=8, freq="7D")
    all_dates = pd.concat([dates, pd.Series(ext)], ignore_index=True)
    woy_all = all_dates.dt.isocalendar().week.to_numpy()

    if cutoff is None:
        cutoff = int(os.environ.get("VN2_CUTOFF", T))
    cutoff = int(max(1, min(cutoff, T)))

    shape, prof = _seasonal(D, A, woy_all[:T])
    # seasonal strength knob (0 = ignore the yearly pattern, 1 = full)
    shape = 1.0 + SEASON_POW * (shape - 1.0)
    sh = np.maximum(shape[:, woy_all[:T]], 0.2)  # (n, T) seasonal divisor

    # ---- seasonally normalised demand history
    dnorm = D / sh

    # ---- level: recent window, shrunk towards the longer pool mean
    lw = max(1, min(LEVEL_WEEKS, cutoff))
    pw = max(2, min(POOL_WEEKS, cutoff))
    recent = dnorm[:, cutoff - lw : cutoff].mean(axis=1)
    pool = dnorm[:, cutoff - pw : cutoff].copy()
    pool_mean = pool.mean(axis=1)
    kappa = max(0.0, LEVEL_SHRINK)
    lvl = (lw * recent + kappa * pool_mean) / (lw + kappa)

    # ---- seasonal move to the 8 game weeks
    game_woy = woy_all[cutoff : cutoff + 8]
    gshape = np.maximum(shape[:, game_woy], 0.2)  # (n, 8)
    F = lvl[:, None] * gshape  # expected demand level per game week (n, 8)

    scale_pool = np.where(pool_mean > 1e-9, 1.0 / np.maximum(pool_mean, 1e-9), 0.0)

    return {
        "n": n,
        "T": T,
        "cutoff": cutoff,
        "pool": pool,
        "scale_pool": scale_pool,
        "F": F,
        "Y": Y,
        "A": A,
        "D": D,
    }


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


# ------------------------------------------------------------------ policy
def _feedback(model, revealed, w: int) -> np.ndarray:
    """Multiplicative level update from the demand revealed inside the game."""
    n = model["n"]
    if w <= 0:
        return np.ones(n)
    rev = np.asarray(revealed, dtype=float)
    if rev.ndim != 2 or rev.shape[0] == 0:
        return np.ones(n)
    obs = rev[:w].sum(axis=0)
    exp = model["F"][:, :w].sum(axis=1)
    tiny = exp < 0.05
    ratio = np.where(
        tiny,
        np.where(obs > 0, 3.0, 1.0),
        (obs + 0.5 * exp) / (1.5 * np.maximum(exp, 1e-9)),
    )
    lam = (w / (w + SHRINK_PRIOR)) ** FEEDBACK_POW
    fb = 1.0 + lam * (np.clip(ratio, 0.3, 3.0) - 1.0)
    return np.clip(fb, LEVEL_CLIP[0], LEVEL_CLIP[1])


def _optimise(state, Dsim, w: int):
    """Coordinate descent over the remaining order slots (returns slot 0)."""
    n, S, H = Dsim.shape
    K = max(1, min(6 - w, 8 - w))  # order slots left (weeks w..5)
    on_hand, a1, a2 = np.asarray(state, dtype=float).T
    a1 = np.nan_to_num(a1)
    a2 = np.nan_to_num(a2)
    on_hand = np.nan_to_num(on_hand)

    def carry_chain(cur_q):
        """Inventory at the start of each order-slot week (index in Dsim)."""
        carry = np.repeat(on_hand[:, None], S, axis=1)
        for t, arr in ((w, a1), (w + 1, a2)):
            j = t - w
            if 0 <= j < H:
                carry = np.maximum(0.0, carry + arr[:, None] - Dsim[:, :, j])
        starts = []
        for k in range(K):
            starts.append(carry)
            j = 2 + k  # slot k is served in absolute week w+2+k
            if j < H:
                carry = np.maximum(0.0, carry + cur_q[k][:, None] - Dsim[:, :, j])
        return starts

    cur_q = [np.zeros(n) for _ in range(K)]
    ar = np.arange(n)

    def tail_cost(k, cands, starts):
        """Expected cost of weeks (w+2+k)..7 given candidate orders for slot k."""
        C = cands.shape[0]
        carry = starts[k][None, :, :]
        cost = np.zeros((C, n, S), dtype=float)
        for kk in range(k, K):
            j = 2 + kk
            if j >= H:
                break
            q = cands[:, :, None] if kk == k else cur_q[kk][None, :, None]
            sup = carry + q
            d = Dsim[None, :, :, j]
            end = np.maximum(0.0, sup - d)
            cost += np.maximum(0.0, d - sup) + HOLD * end
            carry = end
        return cost.mean(axis=2)  # (C, n)

    for _ in range(N_PASS):
        for k in range(K):
            j = 2 + k
            if j >= H:
                continue
            starts = carry_chain(cur_q)
            d_k = Dsim[:, :, j]
            qbase = np.quantile(d_k, CAND_P, axis=1)  # (P, n)
            cmed = np.median(starts[k], axis=1)  # (n,)
            cands = np.concatenate(
                [qbase, np.maximum(0.0, qbase - cmed[None, :])], axis=0
            )  # (C, n)
            exp_cost = tail_cost(k, cands, starts)
            best = exp_cost.argmin(axis=0)
            cur_q[k] = np.maximum(0.0, cands[best, ar])

    # local refinement around the coarse optimum (unit-level precision)
    off = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    for _ in range(REFINE):
        for k in range(K):
            j = 2 + k
            if j >= H:
                continue
            starts = carry_chain(cur_q)
            cands = np.maximum(0.0, cur_q[k][None, :] + off[:, None])  # (9, n)
            exp_cost = tail_cost(k, cands, starts)
            best = exp_cost.argmin(axis=0)
            cur_q[k] = np.maximum(0.0, cands[best, ar])

    return cur_q[0]


def _generic_decide(state, w: int, revealed) -> np.ndarray:
    """Fallback for an unknown series layout.

    Uses only the demand revealed inside the game (order invariant): each
    series' own level from its revealed weeks, and the pooled shape of the
    revealed demand to pick a newsvendor-style quantile.
    """
    st = np.asarray(state, dtype=float)
    n = st.shape[0]
    try:
        rev = np.asarray(revealed, dtype=float) if revealed is not None else None
        if rev is None or rev.ndim != 2 or rev.shape[0] == 0:
            return np.zeros(n)
        rev = rev[:w]
        if rev.shape[0] == 0:
            return np.zeros(n)
        pooled = rev.ravel()
        pooled = pooled[pooled > 0]
        if pooled.size == 0:
            return np.zeros(n)
        lvl = rev.mean(axis=0)
        ratio = np.quantile(pooled, 0.6) / max(pooled.mean(), 1e-9)
        want = np.maximum(0.0, lvl * ratio)
        on_hand, arr1, _arr2 = st.T
        proj = np.nan_to_num(on_hand) + np.nan_to_num(arr1)
        return np.maximum(0.0, want - proj)
    except Exception:
        return np.zeros(n)


def decide(week_index, state, revealed):
    """week_index: int 0..7, state: (n,3), revealed: (week_index, n)."""
    st = np.asarray(state, dtype=float)
    if st.ndim == 1:
        st = st.reshape(1, -1)
    n = st.shape[0]
    w = int(week_index)
    try:
        model = _get_model()
    except Exception:
        return np.zeros(n)
    if model["n"] != n:
        # Unknown series layout (no per-series model available): use a generic,
        # order-invariant rule built from the demand revealed inside the game.
        return _generic_decide(st, w, revealed)

    rev = (
        np.asarray(revealed, dtype=float) if revealed is not None else np.empty((0, n))
    )
    if rev.ndim != 2:
        rev = rev.reshape(0, n)

    fb = _feedback(model, rev, w)

    H = 8 - w
    pool = model["pool"]
    L = pool.shape[1]
    rng = np.random.default_rng(9173 + 31 * w)
    idx = rng.integers(0, L, size=(n, N_PATHS * H))
    sample = pool[np.arange(n)[:, None], idx].reshape(n, N_PATHS, H)
    # put the sample on the forecast level of the corresponding game week
    scale = (
        model["F"][:, w:][:, None, :]
        * model["scale_pool"][:, None, None]
        * fb[:, None, None]
    )
    Dsim = sample * scale

    q = _optimise(st, Dsim, w)
    q = np.nan_to_num(np.maximum(np.asarray(q, dtype=float), 0.0), nan=0.0)
    return q
