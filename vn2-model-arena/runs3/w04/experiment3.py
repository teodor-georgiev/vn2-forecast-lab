import sys, time
import numpy as np
import pandas as pd

ARENA = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
sys.path.insert(0, ARENA + "/tools")
import inventory_sim as sim

sys.path.insert(0, "C:/Users/Toni/Documents/Default Project/vn2-model-arena/runs3/w04")
import helper_backtest as hb

t0 = time.time()
sales = pd.read_csv(ARENA + "/data/sales.csv")
avail = pd.read_csv(ARENA + "/data/availability.csv")
initial = pd.read_csv(ARENA + "/data/initial_state.csv")

series_list = sorted(sales["series_id"].unique().tolist())
sid_to_idx = {s: i for i, s in enumerate(series_list)}
n = len(series_list)
weeks = sorted(sales["week"].unique().tolist())
T = len(weeks)
w_to_t = {w: t for t, w in enumerate(weeks)}

D = np.zeros((n, T))
A = np.zeros((n, T), dtype=bool)
for _, r in sales.iterrows():
    D[sid_to_idx[r["series_id"]], w_to_t[r["week"]]] = r["y"]
for _, r in avail.iterrows():
    A[sid_to_idx[r["series_id"]], w_to_t[r["week"]]] = bool(r["is_available"])

init_state = (
    initial.set_index("series_id")
    .loc[series_list][["on_hand", "arrival_w1", "arrival_w2"]]
    .to_numpy(float)
)

Ycorr = D.copy()
woy = np.array([t % 52 for t in range(T)])
for i in range(n):
    for t in range(T):
        if not A[i, t]:
            mask = (woy == woy[t]) & A[i]
            if mask.sum() > 0:
                Ycorr[i, t] = D[i][mask].mean()
            elif A[i].sum() > 0:
                Ycorr[i, t] = D[i][A[i]].mean()
            else:
                Ycorr[i, t] = 0.0

series_mean = Ycorr.mean(axis=1)
series_std = Ycorr.std(axis=1)
series_zero = (Ycorr == 0).mean(axis=1)
series_max = Ycorr.max(axis=1)


def woy_sin_cos(idx):
    ang = 2 * np.pi * (idx % 52) / 52.0
    return np.sin(ang), np.cos(ang)


def make_features_for_series(y, target_idx, series_idx):
    cur = len(y)

    def lag(k):
        if k <= cur:
            return float(y[-k])
        return 0.0

    lags = [
        lag(1),
        lag(2),
        lag(3),
        lag(4),
        lag(5),
        lag(6),
        lag(7),
        lag(8),
        lag(13),
        lag(26),
        lag(52),
    ]

    def roll(m):
        arr = y[-m:] if m <= cur else y
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))

    m4 = roll(4)
    m8 = roll(8)
    m13 = roll(13)
    m26 = roll(26)
    m52 = roll(52)
    trend = float(np.mean(y[-4:]) - np.mean(y[-8:-4])) if cur >= 8 else 0.0
    if cur >= 13:
        slope = float(np.polyfit(np.arange(13), y[-13:], 1)[0])
    else:
        slope = 0.0
    ratio = float(np.mean(y[-4:]) / (series_mean[series_idx] + 1e-6))
    si, ci = woy_sin_cos(target_idx)
    feats = [
        series_mean[series_idx],
        series_std[series_idx],
        series_zero[series_idx],
        series_max[series_idx],
        target_idx / 157.0,
        si,
        ci,
        *lags,
        m4[0],
        m4[1],
        m4[2],
        m8[0],
        m8[1],
        m13[0],
        m13[1],
        m13[2],
        m26[0],
        m26[1],
        m52[0],
        m52[1],
        trend,
        slope,
        ratio,
    ]
    return feats


import lightgbm as lgb

HORIZONS = [1, 2, 3]
models = {}
for h in HORIZONS:
    X = []
    yy = []
    for i in range(n):
        y = Ycorr[i]
        for t in range(max(52, T - 104), T - h):
            if not A[i, t + h]:
                continue
            feats = make_features_for_series(y[: t + 1], t + h, i)
            X.append(feats)
            yy.append(D[i, t + h])
    X = np.array(X, float)
    yy = np.array(yy, float)
    print("horizon", h, X.shape, flush=True)
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.1,
        n_estimators=60,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        verbose=-1,
        max_bin=63,
    )
    model.fit(X, yy)
    models[h] = model
print("trained", round(time.time() - t0, 1), flush=True)


def forecast_matrix(week_index, state, revealed):
    n_series = state.shape[0]
    if n_series != n:
        return np.zeros((3, n_series))
    rev = np.asarray(revealed, dtype=float)
    if rev.size == 0:
        rev2 = np.zeros((0, n))
    else:
        rev2 = rev.reshape(-1, n)
    if rev2.shape[0] > 0:
        ext = np.hstack([Ycorr, rev2.T])
    else:
        ext = Ycorr.copy()
    out = np.zeros((3, n_series))
    cur_len = T + week_index
    Xall = {}
    for hi, h in enumerate([1, 2, 3]):
        rows = []
        for i in range(n_series):
            y = ext[i, :cur_len]
            target_idx = cur_len - 1 + h
            rows.append(make_features_for_series(y, target_idx, i))
        X = np.array(rows, float)
        pred = models[h].predict(X)
        sn = np.zeros(n_series)
        tw = (cur_len - 1 + h) % 52
        for i in range(n_series):
            y = ext[i, :cur_len]
            hist_idx = [t for t in range(cur_len) if t % 52 == tw]
            sn[i] = np.mean(y[hist_idx]) if len(hist_idx) > 0 else np.mean(y)
        out[hi, :] = np.maximum(0.0, 0.8 * pred + 0.2 * sn)
    return out


def decide(week_index, state, revealed):
    fc = forecast_matrix(week_index, state, revealed)
    a, b, c = [np.maximum(0.0, fc[i]) for i in range(3)]
    st = np.asarray(state, float)
    on_hand, arr1, arr2 = st.T
    buffer = 0.75
    projection = 1.0
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
    return np.clip(deficit, 0, None)


for start in [100, 130]:
    res = hb.backtest(decide, D, start_week=start, initial=init_state)
    print(f"start {start}: score {res['score']:.2f}", flush=True)
    for r in res["weekly"]:
        print(r, flush=True)
