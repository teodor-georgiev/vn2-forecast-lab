"""VN2 adaptive ordering policy.

Per-series stochastic demand-scenario optimizer with a weekly coordinate search
over the order-up-to quantile.

Interface (TASK3.md):

    decide(week_index, state, revealed) -> (n_series,) non-negative orders

How it works
------------
1. Demand prior.  From the arena history (`data/sales.csv` + `data/availability.csv`)
   each series gets a censoring-aware weekly rate `m` and weekly variance `v`,
   which define a negative-binomial (Gamma-Poisson) weekly demand law.
2. Adaptation.  Every decision the level is shrunk toward the demand actually
   revealed inside this game, and the variance is rescaled with the level.
3. Scenario optimizer.  At decision week w we sample many demand paths for the
   weeks still to come, and for each series search over order-up-to quantiles
   `tau` on a grid.  For every `tau` the "order up to the tau-quantile of demand
   from arrival to the end of the game" rule is simulated through the whole
   remaining game (weeks 3-8 scored, 1.00 per missed unit, 0.20 per held
   unit-week); the tau with the lowest simulated expected cost wins for that
   series.  The engine returns the order implied by each series' chosen tau.

The whole thing is vectorised over series, so a 599-series replay stays well
under the compute budget.  No file writes, no network, read-only arena data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SHORT = 1.00
HOLD = 0.20
CENSOR_WEIGHT = 0.65  # weight on the availability-corrected rate
LEVEL_PRIOR_WEEKS = 8.0  # pseudo-weeks of prior when blending revealed demand
TAU_GRID = np.array([0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95])
N_SCEN = 64  # scenarios used for the cost evaluation
N_TARGET = 600  # paths used to estimate the demand quantiles
HIST_WINDOW = 26  # recent history window for the level estimate


# --------------------------------------------------------------------------- #
# history prior (computed once at import)
# --------------------------------------------------------------------------- #
def _find_data_dir():
    here = Path(__file__).resolve()
    for base in here.parents[:5]:
        cand = base / "data"
        if (cand / "sales.csv").exists():
            return cand
    return None


def _load_history():
    data = _find_data_dir()
    if data is None:
        return None, None
    try:
        import pandas as pd

        sales = pd.read_csv(data / "sales.csv")
        avail = pd.read_csv(data / "availability.csv")
        s = sales.pivot(index="series_id", columns="week", values="y")
        a = avail.pivot(index="series_id", columns="week", values="is_available")
        a = a.reindex(index=s.index, columns=s.columns)
        return s.to_numpy(dtype=float), a.to_numpy(dtype=float)
    except Exception:
        return None, None


def build_prior(y, av, cols=None, window=HIST_WINDOW):
    """Censoring-aware per-series weekly mean/variance -> (m, v)."""
    if cols is not None:
        y = y[:, cols]
        av = av[:, cols]
    y = np.nan_to_num(y, nan=0.0)
    av = np.nan_to_num(av, nan=1.0)
    n = y.shape[0]
    if y.shape[1] == 0:
        return np.full(n, 0.5), np.full(n, 1.0)

    def rates(sy, sa):
        cnt = np.maximum(sa.sum(axis=1), 1e-9)
        r_av = (sy * sa).sum(axis=1) / cnt
        return r_av, sy.mean(axis=1)

    r_av_w, r_all_w = rates(y[:, -window:], av[:, -window:])
    r_av_n, r_all_n = rates(y, av)
    r_av = np.where(np.isfinite(r_av_w), r_av_w, r_av_n)
    r_all = np.where(np.isfinite(r_all_w), r_all_w, r_all_n)
    r_av = np.where(np.isfinite(r_av), r_all, 0.0)
    r_all = np.where(np.isfinite(r_all), 0.0, 0.0)
    m = np.maximum(CENSOR_WEIGHT * r_av + (1.0 - CENSOR_WEIGHT) * r_all, 1e-3)

    yc = np.where(av > 0, y, m[:, None])
    var_w = yc[:, -window:].var(axis=1, ddof=1)
    var_n = yc.var(axis=1, ddof=1)
    v = np.where(np.isfinite(var_w) & (var_w > 0), var_w, var_n)
    v = np.where(np.isfinite(v) & (v > 0), v, (1.3 * m) ** 2)
    return m, np.maximum(v, 1.05 * m)


_HIST_Y, _HIST_A = _load_history()
if _HIST_Y is not None:
    _PRIOR_M, _PRIOR_V = build_prior(_HIST_Y, _HIST_A)
else:  # pragma: no cover
    _PRIOR_M = _PRIOR_V = None


# --------------------------------------------------------------------------- #
# demand model
# --------------------------------------------------------------------------- #
def _nb_params(m, v):
    """Gamma-Poisson parameters matched to mean m and variance v."""
    m = np.maximum(np.nan_to_num(np.asarray(m, dtype=float), nan=1e-3), 1e-6)
    v = np.nan_to_num(np.asarray(v, dtype=float), nan=0.0)
    v = np.maximum(v, 1.05 * m)
    r = m * m / np.maximum(v - m, 1e-6)
    return np.clip(r, 0.05, 1e6), m / np.clip(r, 0.05, 1e6)


def _sample_paths(m, v, horizon, rng, n_scen):
    r, scale = _nb_params(m, v)
    if horizon <= 0:
        return np.zeros((n_scen, m.shape[0], 0))
    lam = rng.gamma(
        shape=r[None, :, None],
        scale=scale[None, :, None],
        size=(n_scen, m.shape[0], horizon),
    )
    return rng.poisson(lam).astype(float)


def _tail_targets(m, v, horizon, taus, rng, n_paths=N_TARGET):
    """targets[k, step, i] = tau_k-quantile of demand over weeks step+2..end."""
    paths = _sample_paths(m, v, horizon, rng, n_paths)
    # cumulative sums from the end: cs[:, :, j] = sum of weeks j..horizon-1
    cs = np.cumsum(paths[:, :, ::-1], axis=2)[:, :, ::-1]
    targets = np.zeros((taus.shape[0], horizon, m.shape[0]))
    for k, t in enumerate(taus):
        for step in range(horizon):
            j = step + 2
            if j < horizon:
                targets[k, step] = np.quantile(cs[:, :, j], t, axis=0)
    return targets


# --------------------------------------------------------------------------- #
# core decision
# --------------------------------------------------------------------------- #
def _update_model(w, revealed, m, v):
    if w > 0 and revealed is not None and revealed.size:
        rev = np.asarray(revealed, dtype=float)
        mr = rev.mean(axis=0)
        k = LEVEL_PRIOR_WEEKS
        m_eff = (k * m + w * mr) / (k + w)
        scale = np.where(m > 0, m_eff / m, 1.0)
        v_eff = v * np.clip(scale, 0.05, 20.0) ** 2
        return np.maximum(m_eff, 1e-3), np.maximum(v_eff, 1.05 * m_eff)
    return m, v


def _choose(w, state, m, v, rng, taus=TAU_GRID, n_scen=N_SCEN):
    """Return (orders, chosen_tau_index) for the remaining-horizon rule."""
    n = state.shape[0]
    horizon = 8 - w  # weeks w..7 still to come
    targets = _tail_targets(m, v, horizon, taus, rng)

    demand = _sample_paths(m, v, horizon, rng, n_scen)  # (S, n, H)
    st = np.broadcast_to(state[None, None, :, :], (taus.shape[0], n_scen, n, 3)).copy()

    total = np.zeros((taus.shape[0], n))
    for step in range(horizon):
        wk = w + step
        on_hand, arr1, arr2 = st[:, :, :, 0], st[:, :, :, 1], st[:, :, :, 2]
        if wk < 6:  # orders may be placed in weeks 0..5
            ip = on_hand + arr1 + arr2
            order = np.rint(np.maximum(0.0, targets[:, step, :][:, None, :] - ip))
        else:
            order = np.zeros((taus.shape[0], n_scen, n))
        supply = on_hand + arr1
        d = demand[:, :, step]
        end = np.maximum(0.0, supply - d)
        missed = np.maximum(0.0, d - supply)
        if wk >= 2:
            total += (SHORT * missed + HOLD * end).sum(axis=1)
        st = np.stack([end, arr2, order], axis=3)

    best = np.argmin(total, axis=0)  # (n,)
    ip_now = state[:, 0] + state[:, 1] + state[:, 2]
    q = np.maximum(0.0, targets[best, 0, np.arange(n)] - ip_now)
    return np.rint(q), best


def decide(week_index, state, revealed):
    """week_index: int 0..7.  state: (n,3).  revealed: (week_index, n)."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 2:
        state = state.reshape(-1, 3)
    n = state.shape[0]
    if week_index >= 6 or n == 0:
        return np.zeros(n)

    if _PRIOR_M is not None and _PRIOR_M.shape[0] == n:
        m, v = _PRIOR_M.copy(), _PRIOR_V.copy()
    else:
        m, v = _prior_from_revealed(week_index, revealed, n)

    m, v = _update_model(week_index, revealed, m, v)
    rng = np.random.default_rng(1000 + 17 * week_index)
    orders, _ = _choose(week_index, state, m, v, rng)
    return np.maximum(0.0, orders)


def _prior_from_revealed(w, revealed, n):
    """Fallback prior when the arena history does not match the state length."""
    if w > 0 and revealed is not None and np.asarray(revealed).size:
        rev = np.asarray(revealed, dtype=float)
        m = np.maximum(np.nan_to_num(rev.mean(axis=0), nan=1.0), 1e-3)
        var = rev.var(axis=0, ddof=1) if rev.shape[0] > 1 else np.zeros(n)
        var = np.nan_to_num(var, nan=0.0)
        v = np.maximum(var, 1.05 * m)
        return np.minimum(m, 1e5), v
    return np.full(n, 1.0), np.full(n, 2.0)
