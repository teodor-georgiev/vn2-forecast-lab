"""Parametrised policy model + sweep harness (research only)."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import hb2  # noqa: E402

DATA = {}


def load(decensor=True):
    key = ("data", decensor)
    if key in DATA:
        return DATA[key]
    import pandas as pd

    root = HERE.parents[1]
    sales = pd.read_csv(root / "data" / "sales.csv", parse_dates=["week"])
    avail = pd.read_csv(root / "data" / "availability.csv", parse_dates=["week"])
    init = pd.read_csv(root / "data" / "initial_state.csv")
    Y = (
        sales.pivot(index="series_id", columns="week", values="y")
        .sort_index()
        .values.astype(float)
    )
    A = (
        avail.pivot(index="series_id", columns="week", values="is_available")
        .sort_index()
        .values.astype(float)
    )
    A = (A > 0.5).astype(float)
    D = decensor_matrix(Y, A) if decensor else Y.copy()
    init_s = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
    DATA[key] = (Y, A, D, init_s)
    return DATA[key]


def decensor_matrix(y, a, wnd=11):
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
    local = num / den
    smean = av.sum(1) / np.maximum(a.sum(1), 1.0)
    est = np.where(den > 1.5, local, smean[:, None]) * prof[wk]
    return np.where(a > 0.5, y, np.maximum(y, est))


def seasonal_profiles(d, blocks=52):
    """Detrended global profile + per-series profiles (year-over-year normalised)."""
    n, T = d.shape
    idx, blk = np.arange(T) % 52, np.arange(T) // 52
    tot = d.sum(0)
    num = np.zeros(52)
    cnt = np.zeros(52)
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

    # per-series yearly profile (deseasonalised by global): profile of d / glob
    dd = d / glob[idx]
    pnum = np.zeros((n, 52))
    pcnt = np.zeros((n, 52))
    for b in np.unique(blk):
        sel = blk == b
        m = dd[:, sel].mean(1, keepdims=True)
        m = np.maximum(m, 1e-9)
        for k in np.unique(idx[sel]):
            pnum[:, k] += (dd[:, sel][:, idx[sel] == k] / m).mean(1)
            pcnt[:, k] += 1
    per = np.where(pcnt > 0, pnum / np.maximum(pcnt, 1), 1.0)
    per = np.clip(per, 0.2, 5.0)
    return glob, per


def nb_quantile(mu, var, q):
    from scipy import stats

    mu = np.maximum(mu, 1e-9)
    var = np.maximum(var, mu * (1 + 1e-9))
    pois = (var - mu) <= 1e-9
    r = np.where(pois, 1e12, mu * mu / np.maximum(var - mu, 1e-9))
    p = r / (r + mu)
    out = stats.nbinom.ppf(q, r, p)
    return np.maximum(np.where(np.isfinite(out), out, mu), 0.0)


class Model:
    def __init__(self, decensor=True, **kw):
        self.Y, self.A, self.D, self.init = load(decensor)
        self.glob, self.per = seasonal_profiles(self.D)
        self.p = dict(
            win_short=13,
            win_long=52,
            w_recent=0.85,
            use_trend=True,
            trend_shrink=0.5,
            trend_lo=0.85,
            trend_hi=1.20,
            seas_mode="global",
            seas_k=4.0,
            flat_C=False,
            c_win=None,
            seas_lo=0.4,
            seas_hi=3.0,
            q_A=0.80,
            q_B=0.80,
            q_C=0.80,
            mult_A=1.0,
            mult_B=1.0,
            mult_C=1.0,
            phi_shrink=0.5,
            phi_cap=6.0,
            add_A=0.0,
            add_B=0.0,
            add_C=0.0,
            a_thr=3.0,
            c_thr=1.0,
            base=None,
            phi_mode="shrunk",
        )
        self.p.update(kw)
        self._prep()

    def _prep(self):
        p = self.p
        d = self.D
        mu = d.mean(1)
        phi = np.clip((d.var(1) - mu) / np.maximum(mu * mu, 1e-9), 0, p["phi_cap"])
        med = np.median(mu[mu > 0]) if (mu > 0).any() else 1.0
        hi = mu >= max(2.0, med)
        phi_hi = np.median(phi[hi]) if hi.any() else 1.0
        phi_lo = np.median(phi[~hi]) if (~hi).any() else 1.5
        w = p["phi_shrink"]
        self.phi = np.where(hi, w * phi + (1 - w) * phi_hi, w * phi + (1 - w) * phi_lo)
        self.mu_all = mu

    # ---- forecast -------------------------------------------------------
    def forecast(self, w, revealed):
        p = self.p
        base = self.Y.shape[1] if p["base"] is None else p["base"]
        d = self.D[:, :base]
        a = self.A[:, :base]
        if w and revealed is not None and len(revealed):
            d = np.column_stack((d, np.asarray(revealed, float).T))
            a = np.column_stack((a, np.ones((d.shape[0], w))))
        T = d.shape[1]
        n = d.shape[0]
        is_C = self.mu_all < p["c_thr"]
        if p["seas_mode"] == "blend":
            k0 = p["seas_k"]
            S = (3.0 * self.per + k0 * self.glob[None, :]) / (3.0 + k0)
        else:
            S = np.tile(self.glob, (n, 1))
        if p["flat_C"]:
            S = np.where(is_C[:, None], self.glob[None, :], S)
        S = np.clip(np.maximum(S, 1e-6), p["seas_lo"], p["seas_hi"])
        dt = d / S[:, np.arange(T) % 52]  # de-seasonalised demand

        def lvl(k):
            k = min(k, T)
            return dt[:, T - k :].mean(axis=1)

        l_short = lvl(p["win_short"])
        l_long = lvl(p["win_long"])
        lv = p["w_recent"] * l_short + (1 - p["w_recent"]) * l_long
        if p["c_win"]:
            lv_c = p["w_recent"] * lvl(p["c_win"]) + (1 - p["w_recent"]) * l_long
            lv = np.where(is_C, lv_c, lv)
        tr = np.ones(d.shape[0])
        if p["use_trend"]:
            h = max(3, p["win_short"] // 2)
            m2 = lvl(h)
            k = min(2 * h, T)
            m1 = dt[:, T - k : T - h].mean(axis=1)
            rr = np.where((m1 > 0.15) & (m2 > 0.15), m2 / np.maximum(m1, 1e-9), 1.0)
            tr = np.clip(rr, p["trend_lo"], p["trend_hi"])
            tr = 1.0 + p["trend_shrink"] * (tr - 1.0)
        lv = np.maximum(lv, 0.01)
        out = np.zeros((3, n))
        for hh in range(3):
            out[hh] = lv * (tr ** (hh + 1)) * S[:, (T + hh) % 52]
        return out

    # ---- decision -------------------------------------------------------
    def decide(self, w, state, revealed):
        p = self.p
        fc = self.forecast(w, revealed)
        mu = self.mu_all
        mu3 = fc.sum(0)
        var3 = (fc * (1.0 + self.phi[None, :] * fc)).sum(0)
        is_A = mu >= p["a_thr"]
        is_C = mu < p["c_thr"]
        q = np.where(is_A, p["q_A"], np.where(is_C, p["q_C"], p["q_B"]))
        mult = np.where(is_A, p["mult_A"], np.where(is_C, p["mult_C"], p["mult_B"]))
        add = np.where(is_A, p["add_A"], np.where(is_C, p["add_C"], p["add_B"]))
        target = nb_quantile(mu3, var3, q) * mult + add
        ip = np.asarray(state, float).sum(1)
        return np.rint(np.maximum(0.0, target - ip))


def score(model, cutoffs, history_key="D", initial=True, verbose=False):
    Y, A, D, init_s = load()
    hist = D if history_key == "D" else Y
    out = {}
    for c in cutoffs:
        model.p["base"] = c
        r = hb2.backtest(
            model.decide, hist, start_week=c, initial=init_s if initial else None
        )
        out[c] = r["score"]
        if verbose:
            wk = r["weekly"][2:]
            print(
                f"  cut{c}: {r['score']:8.1f} hold {sum(x['holding'] for x in wk):7.0f} "
                f"short {sum(x['shortage'] for x in wk):7.0f}"
            )
    return out


if __name__ == "__main__":
    m = Model()
    for hk in ("D", "Y"):
        s = score(m, [105, 130, 149], history_key=hk)
        print(hk, {k: round(v) for k, v in s.items()})
