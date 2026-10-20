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

# corrected demand
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

# series stats
series_mean = Ycorr.mean(axis=1)
series_std = Ycorr.std(axis=1)
series_zero = (Ycorr == 0).mean(axis=1)
series_max = Ycorr.max(axis=1)


def woy_sin_cos(idx):
    ang = 2 * np.pi * (idx % 52) / 52.0
    return np.sin(ang), np.cos(ang)


def make_features_for_series(y, target_idx, series_idx):
    # y: 1d corrected history up to current (length L), current last index = len(y)-1
    # target_idx: global index of target week
    cur = len(y) - 1

    def lag(k):
        if k <= len(y):
            return y[-k]
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
        arr = y[-m:] if m <= len(y) else y
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))

    m4 = roll(4)
    m8 = roll(8)
    m13 = roll(13)
    m26 = roll(26)
    m52 = roll(52)
    trend = float(np.mean(y[-4:]) - np.mean(y[-8:-4])) if len(y) >= 8 else 0.0
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
        float(y[-1]) if len(y) else 0.0,
    ]
    return feats


# Build training data
import lightgbm as lgb

HORIZONS = [1, 2, 3]
models = {}
for h in HORIZONS:
    X = []
    yy = []
    for i in range(n):
        y = Ycorr[i]
        for t in range(52, T - h):
            # current observed up to t, target t+h
            feats = make_features_for_series(y[: t + 1], t + h, i)
            X.append(feats)
            yy.append(y[t + h])
    X = np.array(X, float)
    yy = np.array(yy, float)
    print(h, X.shape)
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.1,
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(X, yy)
    models[h] = model
print("trained", time.time() - t0)


# Forecast function using models
def forecast_matrix(week_index, state, revealed):
    n_series = state.shape[0]
    # map if n_series==n
    if n_series != n:
        # fallback zeros
        return np.zeros((3, n_series))
    # build extended history
    if week_index == 0:
        pass
    # base Ycorr plus revealed
    rev = np.asarray(revealed, dtype=float)
    if rev.size == 0:
        rev2 = np.zeros((0, n))
    else:
        rev2 = rev.reshape(-1, n)
    # extended
    ext = np.vstack([Ycorr, rev2]) if rev2.shape[0] > 0 else Ycorr.copy()
    # current time offset: before game week week_index, last observed index = T-1 + week_index?
    # revealed has week_index rows (game weeks 0..week_index-1). So last observed global index = T-1+week_index
    out = np.zeros((3, n_series))
    for hi, h in enumerate([1, 2, 3]):
        for i in range(n_series):
            cur_len = T + week_index  # number observed
            y = ext[i, :cur_len]
            target_idx = (
                cur_len - 1 + h
            )  # global index of target (0-based) = (T-1+week_index)+h -> cur_len-1+h = T-1+week_index+h
            feats = np.array(make_features_for_series(y, target_idx, i)).reshape(1, -1)
            pred = models[h].predict(feats)[0]
            out[hi, i] = max(0.0, float(pred))
    return out


# ordering
def decide(week_index, state, revealed):
    fc = forecast_matrix(week_index, state, revealed)  # (3,n)
    # tune buffer/projection
    a, b, c = [np.maximum(0.0, fc[i]) for i in range(3)]
    st = np.asarray(state, float)
    on_hand, arr1, arr2 = st.T
    buffer = 0.97
    projection = 1.0
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0)) - projected
    return np.clip(deficit, 0, None)


# backtest at cutoffs
for start in [100, 130]:
    res = hb.backtest(decide, D, start_week=start, initial=init_state)
    print(f"start {start}: score {res['score']:.2f}")
    for r in res["weekly"]:
        print(r)
