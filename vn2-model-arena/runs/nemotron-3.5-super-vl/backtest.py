"""Backtest harness for round-2 ordering policy.
Uses only arena data/tools + this run directory.
"""

import sys
import numpy as np
import pandas as pd

ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
sys.path.insert(0, ARENA + r"\tools")
from inventory_sim import simulate, reference_orders  # noqa: E402

DATA = ARENA + r"\data"
OUT = ARENA + r"\runs\nemotron-3.5-super-vl"

SALES = pd.read_csv(f"{DATA}/sales.csv")
INIT = (
    pd.read_csv(f"{DATA}/INITIAL_STATE.csv")
    if False
    else pd.read_csv(f"{DATA}/initial_state.csv")
)

# Align everything to initial_state series order
series_order = INIT["series_id"].tolist()
n = len(series_order)

# sales: week is a date string; treat as ordered weekly periods
piv = SALES.pivot(index="series_id", columns="week", values="y")
piv = piv.reindex(index=series_order)
weeks = list(piv.columns)  # 157 dates, oldest -> newest
W = len(weeks)
H = np.asarray(piv.values, dtype=float)  # (n, W)
H = np.nan_to_num(H, nan=0.0)

init = INIT.set_index("series_id").reindex(series_order)
STATE0 = np.column_stack(
    [
        init["on_hand"].values.astype(float),
        init["arrival_w1"].values.astype(float),
        init["arrival_w2"].values.astype(float),
    ]
)

WEEKS = 8
ORDER_WEEKS = 6
SCORED_FROM = 3  # 1-based


# ---------------------------------------------------------------- forecasts
def recent_mean_forecast(train, k=8):
    """Weighted mean of last k valid (nonzero) observations, flat across horizons."""
    n, _ = train.shape
    mu = np.zeros((n, 8))
    sigma = np.zeros(n)
    for i in range(n):
        y = train[i]
        nz = y[y > 0]
        m = len(nz)
        if m == 0:
            mu[i, :] = 0.0
            sigma[i] = 0.0
            continue
        kk = min(k, m)
        recent = nz[-kk:]
        wts = np.arange(1, kk + 1, dtype=float)
        mu[i, :] = np.dot(recent, wts) / wts.sum()
        sigma[i] = np.std(nz) if m > 1 else max(0.5, mu[i, 0])
    return mu, sigma


def croston_forecast(train):
    n, _ = train.shape
    mu = np.zeros((n, 8))
    sigma = np.zeros(n)
    for i in range(n):
        y = train[i]
        nz_idx = np.flatnonzero(y > 0)
        if len(nz_idx) == 0:
            continue
        sizes = y[nz_idx]
        inter = np.diff(
            np.concatenate([[0], nz_idx + 1])
        )  # weeks since previous positive
        inter = inter[1:]
        if len(inter) == 0:
            inter = np.array([1.0])
        avg_size = sizes.mean()
        avg_inter = inter.mean()
        fc = avg_size / max(avg_inter, 1.0)
        mu[i, :] = fc
        sigma[i] = sizes.std() if len(sizes) > 1 else max(0.5, fc)
    return mu, sigma


# ---------------------------------------------------------------- policy
def greedy_orders(mu, sigma, state, k=1.0):
    """Static greedy base-stock: order each week to cover inflated demand 2 weeks out.
    mu, sigma: (n, 8), (n,) effective weekly demand and volatility.
    state: (n, 3) initial (on_hand, arr_w1, arr_w2).
    Returns orders (6, n) for game weeks 1..6.
    """
    d = np.maximum(0.0, mu + k * sigma[:, None])  # (n, 8) effective demand
    on_hand, arr1, arr2 = state.T.copy()
    orders = np.zeros((ORDER_WEEKS, n))
    end = None
    for t in range(WEEKS):  # t = 0-based game week 1..8
        supply = on_hand + (arr1 if t == 0 else arr2 if t == 1 else 0)
        # handle state properly
        if t == 0:
            supply = on_hand + arr1
        elif t == 1:
            supply = end + arr2
        else:
            supply = end + prev_order
        d_t = d[:, t]
        end = np.maximum(0.0, supply - d_t)
        if t < ORDER_WEEKS:
            # order placed week t+1 arrives week t+3 -> targets game week t+3 (1-based) = index t+2
            target_idx = t + 2
            if target_idx < WEEKS:
                # projected end inventory at week t+2 (1-based t+3?) -- compute next week end without this order
                # week t+1 (index t+1) end inventory without order:
                if t + 1 < WEEKS:
                    supply_t1 = end + (
                        arr2 if t == 0 else 0.0
                    )  # arr2 arrives week 2 only (t==1)... messy below
                orders[t] = 0.0  # placeholder
            prev_order = orders[t]
    return orders


def greedy_orders_v2(mu, sigma, state, k=1.0):
    """Clean forward simulation:
    state = (on_hand, arr_next, arr_after_next). At week t: supply = on_hand+arr_next.
    After demand: end. New state = (end, arr_after_next, order_t).
    Order o_t targets supply at week t+2 (0-based index t+2): o_t = max(0, d'[t+2] - end_{t+1})
    where end_{t+1} computed without o_t.
    """
    d = np.maximum(0.0, mu + k * sigma[:, None])
    on_hand, arr1, arr2 = state.T.copy()
    orders = np.zeros((ORDER_WEEKS, n))
    # week 1
    end = np.maximum(0.0, on_hand + arr1 - d[:, 0])
    prev_order = 0.0  # order placed in week 0 (none)
    # state after week 1: (end, arr2, o_1)
    # week 2:
    # We'll iterate, keeping (end_prev, arrival_this_week, arrival_next_week)
    arr_this = arr2  # arrival in week 2
    arr_next = np.zeros(n)  # arrival in week 3 = o_1 (decided now)
    # decide o_1 targeting week 3 (index 2) supply:
    # end_2 (index1) = max(0, end_1 + arr_this - d[1]); supply_3 = end_2 + o_1
    end2 = np.maximum(0.0, end + arr_this - d[:, 1])
    o1 = np.maximum(0.0, d[:, 2] - end2)
    orders[0] = o1
    # now walk weeks 2..8
    cur_end = end2  # end inventory after week 2
    # state entering week 3: on_hand = cur_end, arr_next = o1, arr_after = o_2 (decide at week 2... no)
    # Let's restructure: loop t from 1 (week 2 index) ... simpler to redo in one loop below.
    return orders


def greedy_orders_final(mu, sigma, state, k=1.0):
    d = np.maximum(0.0, mu + k * sigma[:, None])  # (n,8)
    on_hand = np.asarray(state)[:, 0].copy()
    arr1 = np.asarray(state)[:, 1].copy()
    arr2 = np.asarray(state)[:, 2].copy()
    orders = np.zeros((ORDER_WEEKS, n))
    # state = (on_hand, arr_next, arr_after_next)
    oh, an1, an2 = on_hand, arr1, arr2
    end = np.zeros(n)
    for t in range(WEEKS):
        supply = oh + an1
        end = np.maximum(0.0, supply - d[:, t])
        if t < ORDER_WEEKS:
            # order o_t arrives at week t+2 (0-based), targets supply of week t+2
            # need end inventory after week t+1 (without o_t):
            # week t+1 supply = end + an2 (an2 arrives week t+1? careful)
            # At week t, state=(oh, an1, an2): an1 arrives this week(t), an2 arrives week t+1.
            # After week t: new state would be (end, an2, o_t).
            # Week t+1: supply = end + an2. end_t1 = max(0, end+an2 - d[t+1]).
            # Week t+2: supply = end_t1 + o_t. Target d[t+2].
            if t + 1 < WEEKS:
                end_t1 = np.maximum(0.0, end + an2 - d[:, t + 1])
            else:
                end_t1 = end.copy()
            if t + 2 < WEEKS:
                o_t = np.maximum(0.0, d[:, t + 2] - end_t1)
            else:
                o_t = np.zeros(n)
            orders[t] = o_t
            # advance state: new oh=end, an1=an2, an2=o_t
            oh, an1, an2 = end, an2, o_t
        else:
            oh, an1, an2 = end, an2, np.zeros(n)
    return orders


# ---------------------------------------------------------------- reference
def reference_plan(mu, sigma, state, buffer=0.75, scale=1.0, projection=1.0):
    """Static version of reference rule: iterate forward with forecasts, deciding
    o_t via reference_orders using (h1,h2,h3) = demand forecasts at t,t+1,t+2."""
    orders = np.zeros((ORDER_WEEKS, n))
    oh = np.asarray(state)[:, 0].copy()
    an1 = np.asarray(state)[:, 1].copy()
    an2 = np.asarray(state)[:, 2].copy()
    end = np.zeros(n)
    for t in range(WEEKS):
        if t < ORDER_WEEKS:
            fc = [
                np.maximum(0.0, mu[:, t + h]) if t + h < WEEKS else np.zeros(n)
                for h in range(3)
            ]
            st = np.column_stack([oh, an1, an2])
            o_t = reference_orders(
                fc, st, scale=scale, projection=projection, buffer=buffer
            )
            orders[t] = o_t
        supply = oh + an1
        end = np.maximum(0.0, supply - mu[:, t])  # simulate with point forecast
        if t < ORDER_WEEKS:
            oh, an1, an2 = end, an2, orders[t]
        else:
            oh, an1, an2 = end, an2, np.zeros(n)
    return orders


# ---------------------------------------------------------------- backtest
def run_backtest(g, forecast_fn, policy_fn, policy_kwargs, k_safety=1.0):
    """Game weeks = history[g:g+8]; train on history[:g]."""
    train = H[:, :g]
    demand = H[:, g : g + WEEKS].T  # (8, n)
    mu, sigma = forecast_fn(train)
    orders = policy_fn(mu, sigma, STATE0, **policy_kwargs)
    res = simulate(demand, STATE0, orders)
    return res


def compare(cutoffs, k_values):
    results = []
    for g in cutoffs:
        for k in k_values:
            res = run_backtest(
                g, recent_mean_forecast, greedy_orders_final, {"k": k}, k_safety=k
            )
            results.append(("greedy_recent", g, k, res["score"], res["setup"]))
        res = run_backtest(g, recent_mean_forecast, reference_plan, {}, k_safety=None)
        results.append(("reference", g, None, res["score"], res["setup"]))
    return results


def compare_grid(cutoffs, k_values, forecast_fn=recent_mean_forecast):
    """Average scored cost over cutoffs for each (forecast, k); also reference."""
    summary = {}
    for g in cutoffs:
        for k in k_values:
            res = run_backtest(g, forecast_fn, greedy_orders_final, {"k": k})
            summary.setdefault((forecast_fn.__name__, k), []).append(res["score"])
        res = run_backtest(g, forecast_fn, reference_plan, {})
        summary.setdefault(("reference", None), []).append(res["score"])
    rows = []
    for key, scores in summary.items():
        rows.append((key[0], key[1], np.mean(scores), np.std(scores)))
    rows.sort(key=lambda r: r[2])
    return rows


def recent_mean_forecast_k(train, kk=8):
    return recent_mean_forecast_kk(train, kk)


def recent_mean_forecast_kk(train, kk=8):
    n, _ = train.shape
    mu = np.zeros((n, 8))
    sigma = np.zeros(n)
    for i in range(n):
        y = train[i]
        nz = y[y > 0]
        m = len(nz)
        if m == 0:
            continue
        kk2 = min(kk, m)
        recent = nz[-kk2:]
        wts = np.arange(1, kk2 + 1, dtype=float)
        mu[i, :] = np.dot(recent, wts) / wts.sum()
        sigma[i] = nz.std() if m > 1 else max(0.5, mu[i, 0])
    return mu, sigma


def rm_forecast(train, kk):
    return recent_mean_forecast_kk(train, kk)


if __name__ == "__main__":
    cutoffs = [135, 140, 145, 149]
    # forecast comparison at greedy k=0
    print("=== forecast comparison (greedy k=0), averaged over cutoffs ===")
    for kk in [4, 6, 8, 10, 12]:
        scores = []
        for g in cutoffs:
            res = run_backtest(
                g,
                lambda tr, kk=kk: rm_forecast(tr, kk),
                greedy_orders_final,
                {"k": 0.0},
            )
            scores.append(res["score"])
        print(f"recent_mean kk={kk:2d}  avg_score={np.mean(scores):10.2f}")
    scores = []
    for g in cutoffs:
        res = run_backtest(g, croston_forecast, greedy_orders_final, {"k": 0.0})
        scores.append(res["score"])
    print(f"croston             kk=--  avg_score={np.mean(scores):10.2f}")
    scores = []
    for g in cutoffs:
        res = run_backtest(g, recent_mean_forecast, reference_plan, {})
        scores.append(res["score"])
    print(f"reference           avg_score={np.mean(scores):10.2f}")

    print("=== croston with safety stock ===")
    for k in [0.0, 0.25, 0.5, 0.75]:
        scores = []
        for g in cutoffs:
            res = run_backtest(g, croston_forecast, greedy_orders_final, {"k": k})
            scores.append(res["score"])
        print(f"croston  k={k}  avg_score={np.mean(scores):10.2f}")

    # ---------------------------------------------------------------- lightgbm
    print("=== lightgbm panel forecast ===")
    from lightgbm import LGBMRegressor

    def lgbm_forecast(train):
        n, wt = train.shape
        feat_list, targ_list = [], []
        for t in range(4, wt):
            seg = train[:, t - 4 : t]
            base = np.column_stack(
                [
                    train[:, t - 1],
                    train[:, t - 2],
                    train[:, t - 3],
                    train[:, t - 4],
                    seg.mean(axis=1),
                    seg.std(axis=1),
                    seg.max(axis=1),
                    (seg > 0).mean(axis=1),
                ]
            )
            for h in range(1, 9):
                if t + h > wt - 1:
                    continue
                feat_list.append(np.column_stack([np.full((n, 1), float(h)), base]))
                targ_list.append(train[:, t + h])
        X = np.vstack(feat_list)
        y = np.concatenate(targ_list)
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=1,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(X, y)
        # forecast from last observed week, horizons 1..8
        seg = train[:, wt - 4 : wt]
        base = np.column_stack(
            [
                train[:, wt - 2],
                train[:, wt - 3],
                train[:, wt - 4],
                train[:, wt - 5],
                seg.mean(axis=1),
                seg.std(axis=1),
                seg.max(axis=1),
                (seg > 0).mean(axis=1),
            ]
        )
        mu = np.column_stack(
            [
                model.predict(np.column_stack([np.full(n, float(h)), base]))
                for h in range(1, 9)
            ]
        )
        # volatility: std of recent positive demand
        sigma = np.zeros(n)
        for i in range(n):
            nz = train[i][train[i] > 0]
            sigma[i] = nz.std() if len(nz) > 1 else max(0.5, mu[i, 0])
        return np.maximum(mu, 0.0), sigma

    scores = []
    for g in cutoffs:
        res = run_backtest(g, lgbm_forecast, greedy_orders_final, {"k": 0.0})
        scores.append(res["score"])
    print(f"lgbm k=0.0          avg_score={np.mean(scores):10.2f}")
    scores = []
    for g in cutoffs:
        res = run_backtest(g, lgbm_forecast, greedy_orders_final, {"k": 0.25})
        scores.append(res["score"])
    print(f"lgbm k=0.25         avg_score={np.mean(scores):10.2f}")
