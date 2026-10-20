import numpy as np, pandas as pd, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sklearn.linear_model import Ridge, QuantileRegressor

D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
y = Y.values.astype(float)
a = A.values.astype(float)
woy = np.array([d.isocalendar().week for d in Y.columns])
n = y.shape[0]
T = y.shape[1]
series_mean = np.nanmean(np.where(a > 0.5, y, np.nan), axis=1)


def make_features(hist, av, woyh, pos):
    woyh = np.asarray(woyh)
    while woyh.shape[0] <= pos:
        woyh = np.append(woyh, (int(woyh[-1]) % 52) + 1)

    def lag(k):
        idx = pos - k
        if 0 <= idx < hist.shape[1]:
            col = np.where(av[:, idx] > 0.5, hist[:, idx], np.nan)
            col = np.where(np.isnan(col), series_mean, col)
        else:
            col = series_mean.copy()
        return col

    feats = [series_mean]
    feats.append(np.sin(2 * np.pi * woyh[pos] / 52) * np.ones(n))
    feats.append(np.cos(2 * np.pi * woyh[pos] / 52) * np.ones(n))
    for k in [1, 2, 3, 4, 8, 13, 26, 52]:
        feats.append(lag(k))
    for w in [4, 13]:
        cols = []
        for k in range(1, w + 1):
            idx = pos - k
            if 0 <= idx < hist.shape[1]:
                cols.append(np.where(av[:, idx] > 0.5, hist[:, idx], np.nan))
        if cols:
            arr = np.column_stack(cols)
            rm = np.nanmean(arr, axis=1)
            rm = np.where(np.isnan(rm), series_mean, rm)
        else:
            rm = series_mean.copy()
        feats.append(rm)
    return np.column_stack(feats)


def build_train(horizon):
    Xs = []
    ys = []
    for t in range(52, T - horizon):
        hist = y[:, :t]
        av = a[:, :t]
        woyh = woy[:t]
        pos = t + horizon
        mask = a[:, t + horizon] > 0.5
        X = make_features(hist, av, woyh, pos)[mask]
        yt = y[:, t + horizon][mask]
        Xs.append(X)
        ys.append(yt)
    return np.vstack(Xs), np.concatenate(ys)


# mean models for h1,h2
mean_models = {}
for h in [1, 2]:
    X, yt = build_train(h)
    m = Ridge(alpha=1.0)
    m.fit(X, yt)
    mean_models[h] = m
# quantile model for h3
X, yt = build_train(3)
qmodel = QuantileRegressor(quantile=0.833, alpha=1.0, solver="highs")
qmodel.fit(X, yt)
print("trained")


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


def make_decide(buffer=0.0, projection=1.0):
    def decide(week_index, state, revealed):
        state = np.asarray(state, float)
        revealed = np.asarray(revealed, float)
        if n == state.shape[0]:
            hist = y.copy()
            av = a.copy()
            woyh = woy.copy()
        else:
            hist = np.zeros((state.shape[0], 0))
            av = np.ones((state.shape[0], 0))
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
            X = make_features(hist, av, woyh, pos)
            pred = mean_models[hh].predict(X)
            fmeans.append(np.maximum(pred, 0))
        # quantile for c
        pos = Tcur + 2
        X = make_features(hist, av, woyh, pos)
        c_pred = qmodel.predict(X)
        c_pred = np.maximum(c_pred, 0)
        fc = np.array([fmeans[0], fmeans[1], c_pred])
        a1, b, cc = [f for f in fc]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a1 + b))
        deficit = cc + buffer * np.sqrt(np.maximum(cc, 0)) - projected
        return np.clip(deficit, 0, None)

    return decide


decide = make_decide(buffer=0.0, projection=1.0)
for sw in [100, 130]:
    sc = backtest(decide, y, start_week=sw, initial=None)
    print(f"sw={sw} score={sc:.0f}")
