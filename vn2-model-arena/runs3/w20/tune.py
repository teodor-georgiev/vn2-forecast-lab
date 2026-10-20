"""Backtest harness for the w20 policy core (authoring only)."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")

sales = pd.read_csv(DATA / "sales.csv")
avail = pd.read_csv(DATA / "availability.csv")
init = pd.read_csv(DATA / "initial_state.csv")
attrs = pd.read_csv(DATA / "product_attributes.csv")

order_ids = init["series_id"].tolist()
piv = sales.pivot_table(index="series_id", columns="week", values="y").reindex(
    index=order_ids
)
Y = piv.to_numpy(dtype=float)
WEEKS = list(piv.columns)
A = (
    avail.pivot_table(index="series_id", columns="week", values="is_available")
    .reindex(index=order_ids, columns=WEEKS)
    .to_numpy(dtype=bool)
)
atts = attrs.set_index("series_id").reindex(order_ids)
# group key: Department (fallback ProductGroup)
gkey = (
    atts["Department"].astype(str) + "|" + atts["ProductGroup"].astype(str)
).to_numpy()
uniq, gidx = np.unique(gkey, return_inverse=True)
NG = len(uniq)
# calendar week-of-year index (0..51) per historical week
wofy = pd.to_datetime(WEEKS).isocalendar().week.to_numpy().astype(int) - 1
print("groups", NG, "T", Y.shape[1])


def group_seasonal(Y, A, gidx, t0, half=26.0):
    """Pooled within-group multiplicative week-of-year index from weeks < t0."""
    T = Y.shape[1]
    w = wofy[:t0]
    num = np.zeros((NG, 52))
    cnt = np.zeros((NG, 52))
    for g in range(NG):
        sel = gidx == g
        if not sel.any():
            continue
        Yg = Y[sel][:, :t0]
        # per-series level (avail weeks only, censoring-adjusted)
        lvl = np.where(A[sel][:, :t0], Yg, np.nan)
        lvl = np.nanmean(lvl, axis=1)
        lvl = np.where(np.isfinite(lvl) & (lvl > 1e-6), lvl, np.nanmean(Yg, axis=1))
        lvl = np.where(np.isfinite(lvl) & (lvl > 1e-6), lvl, 1.0)
        R = Yg / lvl[:, None]
        for k in range(52):
            m = w == k
            if m.sum() == 0:
                continue
            v = R[:, m].ravel()
            num[g, k] = np.median(v)
            cnt[g, k] = m.sum()
    idx = np.ones((NG, 52))
    ok = cnt >= 3
    idx[ok] = num[ok]
    # shrink toward 1 by sample size and clip
    shr = np.clip(cnt / (cnt + half), 0, 1)
    idx = 1 + shr * (idx - 1)
    return np.clip(idx, 0.5, 2.0)


class Policy:
    def __init__(self, t0, cfg):
        self.t0 = t0
        self.cfg = cfg
        c = cfg
        hist = Y[:, :t0]
        histA = A[:, :t0]
        W = c["W"]
        recent = hist[:, -W:]
        recentA = histA[:, -W:]
        # avail-only level over recent window
        av = np.where(recentA, recent, np.nan)
        with np.errstate(all="ignore"):
            lvl_av = np.nanmean(av, axis=1)
        mean_all = recent.mean(axis=1)
        lvl_av = np.where(np.isfinite(lvl_av) & (lvl_av > 1e-9), lvl_av, mean_all)
        lvl_av = np.where(np.isfinite(lvl_av) & (lvl_av > 1e-9), lvl_av, 0.0)
        self.lvl_prior = lvl_av * c["u"]
        self.lvl_prior = np.maximum(self.lvl_prior, 1e-6)
        # empirical pool: available weeks kept, censored weeks imputed
        pool = recent.copy()
        imp = np.where(recentA, recent, np.nan)
        with np.errstate(all="ignore"):
            lav2 = np.nanmean(imp, axis=1)
        lav2 = np.where(np.isfinite(lav2) & (lav2 > 1e-9), lav2, mean_all)
        lav2 = np.where(np.isfinite(lav2) & (lav2 > 1e-9), lav2, 0.0)
        fill = np.where(recentA, recent, lav2[:, None] * c["u"])
        self.pool = np.where(np.isnan(fill), 0.0, fill)  # (n, W)
        self.seas = group_seasonal(Y, A, gidx, t0)  # (NG, 52)
        self.p = c["p"]
        self.k = c["k"]

    def week_pred(self, scale, seas_week):
        # per-game-week (mean, quantile) using scaled pool
        pool = self.pool * scale[:, None] * seas_week[:, None]
        mean = pool.mean(axis=1)
        q = np.quantile(pool, self.p, axis=1)
        return mean, q

    def decide(self, week_index, state, revealed):
        c = self.cfg
        n = state.shape[0]
        scale = np.ones(n)
        if revealed is not None and len(revealed) > 0:
            w = len(revealed)
            scale = (
                (self.k * self.lvl_prior + revealed.sum(axis=0))
                / (self.k + w)
                / self.lvl_prior
            )
            scale = np.clip(scale, 0.25, 4.0)
        # seasonal index per game week (calendar continuation assumption)
        cal = [(self.t0 + j) % 52 for j in range(8)]
        if n == len(gidx):
            s = self.seas[gidx][:, cal].T  # (8, n)
        else:
            s = np.ones((8, n))
        on_hand, arr1, arr2 = state.T
        order = np.zeros(n)
        # expected end inventory at week_index+1 via Monte Carlo
        mdraw = c["mc"]
        rng = np.random.default_rng(1000 + week_index)
        Wp = self.pool.shape[1]
        idx = rng.integers(0, Wp, size=(mdraw, n))
        seas0 = s[week_index]
        pool0 = self.pool * (scale * seas0)[:, None]
        d0 = pool0[np.arange(n)[None, :], idx]  # (mdraw, n)
        S0 = on_hand + arr1
        end0 = np.maximum(0.0, S0[None, :] - d0)
        idx2 = rng.integers(0, Wp, size=(mdraw, n))
        seas1 = s[week_index + 1]
        pool1 = self.pool * (scale * seas1)[:, None]
        d1 = pool1[np.arange(n)[None, :], idx2]
        S1 = end0 + arr2[None, :]
        end1 = np.maximum(0.0, S1 - d1)
        proj = end1.mean(axis=0)
        # target for week_index+2
        q2 = np.quantile(
            self.pool * (scale * s[week_index + 2])[:, None], self.p, axis=1
        )
        order = np.clip(q2 - proj, 0.0, None)
        if c.get("cap"):
            # cap: don't buy so much that committed supply exceeds cap-quantile of
            # total remaining demand
            tot = np.zeros(n)
            for j in range(week_index + 2, 8):
                tot = tot + np.quantile(
                    self.pool * (scale * s[j])[:, None],
                    min(c["capq"], 1.0),
                    axis=1,
                )
            committed = proj + arr2 * 0  # proj already includes arr2 carryover
            order = np.minimum(order, np.clip(tot - proj, 0.0, None))
        return np.rint(np.maximum(0.0, order))


def reference_decide(history, forecast_window=13, buffer=0.75):
    """Manager's reference bar rule with 13-week-mean forecasts."""

    def fn(week_index, state, revealed):
        n = state.shape[0]
        h = history.shape[1]
        t0 = h - (8 - 0)  # not used; caller sets history up to t0
        return None

    return fn


def backtest(decide, hist, start_week, initial=None, weeks=8):
    """Local mirror of tools/inventory_sim.order_plan_from_policy (helper is buggy)."""
    hist = np.asarray(hist, dtype=float)
    n = hist.shape[0]
    demand = hist[:, start_week : start_week + weeks].T
    state = np.zeros((n, 3)) if initial is None else np.asarray(initial, float).copy()
    revealed = []
    orders = np.zeros((6, n))
    weekly = []
    for w in range(weeks):
        order = np.zeros(n)
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(0.0, np.asarray(decide(w, state.copy(), rev), float))
            )
            orders[w] = order
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {
                "week": w + 1,
                "holding": float(0.2 * end.sum()),
                "shortage": float(1.0 * missed.sum()),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return {"score": float(score), "weekly": weekly, "orders": orders}


def run_reference(hist, start_week, window=13, buffer=0.75, initial=None):
    """Reference deficit rule, forecasts = trailing window mean of revealed/history."""
    n = hist.shape[0]

    def forecast_fn(w, state, revealed):
        rows = []
        for i in range(n):
            past = list(hist[i, :start_week])
            if revealed is not None and len(revealed):
                past = list(np.asarray(revealed)[:, i])
            ww = min(window, len(past))
            rows.append(np.mean(past[-ww:]) if ww else 0.0)
        f = np.array(rows)
        return np.vstack([f, f, f])

    def policy(forecast, state):
        a, b, c = [np.maximum(0.0, np.asarray(f, float)) for f in forecast]
        on_hand, arr1, arr2 = state.T
        proj = np.maximum(0.0, on_hand + arr1 + arr2 - (a + b))
        return np.rint(np.clip(c + buffer * np.sqrt(c) - proj, 0, None))

    return backtest(
        lambda w, st, rev: policy(forecast_fn(w, st, rev), st),
        hist,
        start_week,
        initial=initial,
    )


def run_reference2(hist, start_week, window=13, buffer=0.75, initial=None):
    return run_reference(
        hist, start_week, window=window, buffer=buffer, initial=initial
    )


def run_policy(cfg, hist, start_week, initial=None):
    pol = Policy(start_week, cfg)
    return backtest(pol.decide, hist, start_week, initial=initial)


if __name__ == "__main__":
    import itertools

    INIT = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)
    cuts = [100, 130]
    print("== reference (real initial state) ==")
    for W in (13, 26):
        sc = [run_reference(Y, s, window=W, initial=INIT)["score"] for s in cuts]
        print(
            f"  window={W}: "
            + " ".join("%.0f" % x for x in sc)
            + f"  mean {np.mean(sc):.0f}"
        )

    print("== grid (real initial state) ==")
    best = []
    for W, u, p, k, seas in itertools.product(
        (13, 26), (1.0, 1.25), (0.8, 0.9), (8, 13), (True, False)
    ):
        cfg = {
            "W": W,
            "u": u,
            "p": p,
            "k": k,
            "seas": seas,
            "cap": True,
            "capq": 0.95,
            "mc": 64,
        }
        sc = [run_policy(cfg, Y, s, initial=INIT)["score"] for s in cuts]
        best.append((np.mean(sc), cfg, sc))
    for m, cfg, sc in sorted(best, key=lambda x: x[0])[:8]:
        shown = " ".join(f"{kk}={vv}" for kk, vv in cfg.items() if kk != "mc")
        print("  mean %6.0f  %s  [%s]" % (m, shown, " ".join("%.0f" % x for x in sc)))
