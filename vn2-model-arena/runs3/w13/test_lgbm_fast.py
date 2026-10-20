import numpy as np, pandas as pd, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lightgbm as lgb

D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
y = Y.values.astype(np.float32)
a = A.values.astype(np.float32)
woy = np.array([d.isocalendar().week for d in Y.columns], dtype=np.int32)
n = y.shape[0]
T = y.shape[1]
series_mean = np.nanmean(np.where(a > 0.5, y, np.nan), axis=1).astype(np.float32)
series_std = np.nanstd(np.where(a > 0.5, y, np.nan), axis=1).astype(np.float32)
series_cv = (series_std / (series_mean + 1e-6)).astype(np.float32)


def build_train(horizon, max_t=None):
    ts = list(range(52, (max_t if max_t else T) - horizon))
    # number samples
    X_list = []
    y_list = []
    for t in ts:
        pos = t + horizon
        mask = a[:, pos] > 0.5
        if not mask.any():
            continue

        # features for all series
        # lags
        def lag(k):
            idx = pos - k
            if 0 <= idx < T:
                col = np.where(a[:, idx] > 0.5, y[:, idx], np.nan)
                col = np.where(np.isnan(col), series_mean, col)
            else:
                col = series_mean.copy()
            return col

        F = []
        F.append(series_mean)
        tw = woy[pos % T]  # rough woy for future; for training pos<T so woy[pos]
        # for training pos<T
        tw = woy[pos]
        F.append(np.full(n, np.sin(2 * np.pi * tw / 52), dtype=np.float32))
        F.append(np.full(n, np.cos(2 * np.pi * tw / 52), dtype=np.float32))
        for k in [1, 2, 4, 8, 13, 26, 52]:
            F.append(lag(k).astype(np.float32))
        # rolling mean 13
        cols = []
        for k in range(1, 14):
            idx = pos - k
            if 0 <= idx < T:
                cols.append(np.where(a[:, idx] > 0.5, y[:, idx], np.nan))
        if cols:
            arr = np.column_stack(cols)
            rm = np.nanmean(arr, axis=1)
            rm = np.where(np.isnan(rm), series_mean, rm)
        else:
            rm = series_mean.copy()
        F.append(rm.astype(np.float32))
        X = np.column_stack(F).astype(np.float32)
        X = X[mask]
        yt = y[:, pos][mask]
        X_list.append(X)
        y_list.append(yt)
    Xall = np.vstack(X_list)
    yall = np.concatenate(y_list)
    return Xall, yall


t0 = time.time()
models = {}
for h in [1, 2, 3]:
    Xall, yall = build_train(h)
    dtrain = lgb.Dataset(Xall, label=yall)
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "num_threads": 2,
    }
    m = lgb.train(params, dtrain, num_boost_round=80)
    models[h] = m
    print(f"h{h}: samples {len(yall)} time {time.time() - t0:.1f}")
print("train total", time.time() - t0)

# quantile for h3
t0 = time.time()
Xall, yall = build_train(3)
dtrain = lgb.Dataset(Xall, label=yall)
params = {
    "objective": "quantile",
    "alpha": 0.833,
    "metric": "quantile",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "num_threads": 2,
}
mq = lgb.train(params, dtrain, num_boost_round=80)
print("quantile train", time.time() - t0)


# evaluate
def backtest(decide, history, start_week=0, weeks=8, initial=None):
    history = np.asarray(history, float)
    n = history.shape[0]
    if initial is None:
        state = np.zeros((n, 3))
    else:
        state = np.asarray(initial, float).copy()
    demand = history[:, start_week : start_week + weeks].T
    revealed = []
    orders = np.zeros((6, n))
    weekly = []
    for w in range(weeks):
        if w < 6:
            decision = decide(
                w,
                state.copy(),
                np.array(revealed, dtype=float).reshape(w, n)
                if revealed
                else np.empty((0, n)),
            )
            orders[w] = np.rint(np.maximum(0.0, np.asarray(decision, float)))
        order = orders[w] if w < 6 else np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {"week": w + 1, "holding": 0.2 * end.sum(), "shortage": 1.0 * missed.sum()}
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return float(score)


def make_features_game(hist, av, woyh, pos):
    # hist (n,t)
    t = hist.shape[1]
    woyh = np.asarray(woyh)
    while woyh.shape[0] <= pos:
        woyh = np.append(woyh, (int(woyh[-1]) % 52) + 1)

    def lag(k):
        idx = pos - k
        if 0 <= idx < t:
            col = np.where(av[:, idx] > 0.5, hist[:, idx], np.nan)
            col = np.where(np.isnan(col), series_mean, col)
        else:
            col = series_mean.copy()
        return col.astype(np.float32)

    F = []
    F.append(series_mean)
    tw = int(woyh[pos])
    F.append(np.full(n, np.sin(2 * np.pi * tw / 52), dtype=np.float32))
    F.append(np.full(n, np.cos(2 * np.pi * tw / 52), dtype=np.float32))
    for k in [1, 2, 4, 8, 13, 26, 52]:
        F.append(lag(k))
    cols = []
    for k in range(1, 14):
        idx = pos - k
        if 0 <= idx < t:
            cols.append(np.where(av[:, idx] > 0.5, hist[:, idx], np.nan))
    if cols:
        arr = np.column_stack(cols)
        rm = np.nanmean(arr, axis=1)
        rm = np.where(np.isnan(rm), series_mean, rm)
    else:
        rm = series_mean.copy()
    F.append(rm.astype(np.float32))
    return np.column_stack(F).astype(np.float32)


def make_decide(models, mq):
    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        revealed = np.asarray(revealed, float)
        if n == state.shape[0]:
            hist = y.copy()
            av = a.copy()
            woyh = woy.copy()
        else:
            hist = np.zeros((state.shape[0], 0), dtype=np.float32)
            av = np.ones((state.shape[0], 0), dtype=np.float32)
            woyh = np.array([])
        if revealed.size:
            hist = np.column_stack([hist, revealed.T]) if hist.size else revealed.T
            av = (
                np.column_stack([av, np.ones((state.shape[0], revealed.shape[0]))])
                if av.size
                else np.ones((state.shape[0], revealed.shape[0]))
            )
            if woyh.size:
                woyh = np.concatenate(
                    [
                        woyh,
                        np.array(
                            [
                                (woyh[-1] + i) % 52 + 1
                                for i in range(1, revealed.shape[0] + 1)
                            ]
                        ),
                    ]
                )
        Tcur = hist.shape[1]
        fmeans = []
        for hh in [1, 2]:
            pos = Tcur + hh - 1
            X = make_features_game(hist, av, woyh, pos)
            pred = models[hh].predict(X)
            fmeans.append(np.maximum(pred, 0))
        pos = Tcur + 2
        X = make_features_game(hist, av, woyh, pos)
        c_pred = mq.predict(X)
        c_pred = np.maximum(c_pred, 0)
        fc = np.array([fmeans[0], fmeans[1], c_pred])
        a1, b, c = [f for f in fc]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - 1.0 * (a1 + b))
        deficit = c - projected
        return np.clip(deficit, 0, None)

    return decide


decide = make_decide(models, mq)
for sw in [100, 130]:
    sc = backtest(decide, y, start_week=sw, initial=None)
    print(f"sw={sw} score={sc:.0f}")
