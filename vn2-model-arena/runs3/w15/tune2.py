"""Per-series config selection from rolling backtests, with holdout validation."""

from __future__ import annotations

import itertools
import time

import numpy as np

import mybt
import tune as T  # Y, YC, INIT, ref_orders, censor_correct

Y, YC, INIT = T.Y, T.YC, T.INIT
N, WKS = Y.shape
SHORT, HOLD = 1.0, 0.20


def backtest_ps(decide, hist, start_week, initial):
    """Like mybt.backtest but returns per-series scored cost (n,)."""
    n = hist.shape[0]
    demand = hist[:, start_week : start_week + 8].T
    state = np.asarray(initial, float).copy()
    revealed = []
    orders = np.zeros((6, n))
    cost = np.zeros(n)
    for w in range(8):
        if w < 6:
            orders[w] = np.rint(
                np.maximum(
                    0.0,
                    np.asarray(
                        decide(w, state.copy(), np.array(revealed).reshape(w, n)),
                        dtype=float,
                    ),
                )
            )
        order = orders[w] if w < 6 else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        if w >= 2:
            cost += HOLD * end + SHORT * missed
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w].copy())
    return cost


CONFIGS = [
    dict(scale=s, buffer=b, projection=p, base=base, adapt=a, trend=t)
    for s, b, p, base, a, t in itertools.product(
        [0.95, 1.1, 1.25],
        [0.0, 0.6],
        [1.0, 1.15],
        ["corr", "raw"],
        [0.0, 6.0],
        [False, True],
    )
]


def make_decide(hist, cfg):
    Nh, L0 = hist.shape
    scale, buffer, projection = cfg["scale"], cfg["buffer"], cfg["projection"]
    adapt, trend = cfg["adapt"], cfg["trend"]
    base = cfg["base"]

    def lvl(avail, upto):
        if upto <= 0:
            return np.zeros(avail.shape[0])
        return np.maximum(avail[:, max(0, upto - 13) : upto].mean(axis=1), 0.0)

    def decide(w, state, revealed):
        rev = np.asarray(revealed, dtype=float).reshape(w, Nh).T
        avail = np.concatenate([hist, rev], axis=1) if w else hist
        level_now = lvl(avail, L0 + w)
        if trend and L0 >= 26:
            prev = lvl(avail, L0 + w - 13)
            level_now = level_now * np.clip(
                level_now / np.maximum(prev, 1e-9), 1.0, 1.25
            )
        mult = 1.0
        if adapt > 0 and w > 0:
            pred = np.zeros(Nh)
            for j in range(w):
                pred += lvl(avail, L0 + j)
            obs = rev.sum(axis=1)
            m = np.clip(
                (obs + adapt * level_now) / np.maximum(pred + adapt * level_now, 1e-9),
                0.4,
                2.5,
            )
            mult = np.where(np.isfinite(m), m, 1.0)
        f = [level_now * mult] * 3
        return T.ref_orders(f, state, scale, buffer, projection)

    return decide


def evaluate(cfgs, windows, mats):
    """Return (n_cfg, n_windows) mean total scores and (n_cfg, n_windows, n) costs."""
    tot = np.zeros((len(cfgs), len(windows)))
    per = np.zeros((len(cfgs), len(windows), N))
    for i, cfg in enumerate(cfgs):
        Hmat = YC if cfg["base"] == "corr" else Y
        for j, s in enumerate(windows):
            c = backtest_ps(make_decide(Hmat[:, :s], cfg), Y, s, INIT)
            per[i, j] = c
            tot[i, j] = c.sum()
    return tot, per


if __name__ == "__main__":
    t0 = time.time()
    H = Y
    tune_windows = [95, 115, 130, 145]
    hold_windows = [100, 120, 140, 149]

    tot_tune, per_tune = evaluate(CONFIGS, tune_windows, None)
    mean_tune = tot_tune.mean(1)
    order = np.argsort(mean_tune)
    print("=== tune windows", tune_windows, "===")
    for i in order[:8]:
        print(f"{mean_tune[i]:9.1f}  {CONFIGS[i]}")
    ref_cfg = dict(
        scale=1.0, buffer=0.75, projection=1.0, base="raw", adapt=0.0, trend=False
    )
    tot_ref, _ = evaluate([ref_cfg], tune_windows, None)
    print("reference config:", round(float(tot_ref.mean()), 1))

    # holdout for global best and per-series selection among top-K
    topK = 8
    top_idx = list(order[:topK])
    tot_hold, per_hold = evaluate(
        [CONFIGS[i] for i in top_idx],
        hold_windows,
        None,
    )
    print("=== holdout windows", hold_windows, "=== (top-8 configs)")
    for k, i in enumerate(top_idx):
        print(f"{tot_hold[k].mean():9.1f}  {CONFIGS[i]}")

    # per-series selection on tune windows, validated on holdout
    per_tune_sum = per_tune[top_idx].sum(1)  # (K, n)
    sel = np.argmin(per_tune_sum, axis=0)
    ps_tune_total = np.array(
        [per_tune[top_idx[sel[i]], :, i].sum() for i in range(N)]
    ).mean()
    tot_ref_hold, _ = evaluate([ref_cfg], hold_windows, None)
    print("reference holdout:", round(float(tot_ref_hold.mean()), 1))
    print("per-series tune mean:", round(float(ps_tune_total) / len(tune_windows), 1))
    ps_hold = np.array([per_hold[sel[i], :, i].sum() for i in range(N)]).mean()
    print("per-series holdout mean:", round(float(ps_hold) / len(hold_windows), 1))
    print("best global holdout:", round(float(tot_hold.min(1).mean()), 1))
    # how often each config selected
    vals, cnts = np.unique(sel, return_counts=True)
    print("selection mix:", [(CONFIGS[top_idx[v]], int(c)) for v, c in zip(vals, cnts)])
    np.save("sel.npy", np.array([top_idx[v] for v in sel]))
    np.save("top_idx.npy", np.array(top_idx))
    print("elapsed", round(time.time() - t0, 1))
