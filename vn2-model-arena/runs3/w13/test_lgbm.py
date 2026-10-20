import numpy as np, pandas as pd, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lightgbm as lgb

D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
init = pd.read_csv(D / "initial_state.csv")
Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
ids = list(Y.index)
y = Y.values.astype(float)
a = A.values.astype(float)
woy = np.array([d.isocalendar().week for d in Y.columns])
init = init.set_index("series_id").loc[ids]
st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
n = y.shape[0]
T = y.shape[1]

# series stats
series_mean = np.nanmean(np.where(a > 0.5, y, np.nan), axis=1)
series_std = np.nanstd(np.where(a > 0.5, y, np.nan), axis=1)
series_cv = series_std / (series_mean + 1e-6)
series_fraczero = np.nanmean((np.where(a > 0.5, y, np.nan) == 0), axis=1)
# product attributes encoded
attr = pd.read_csv(D / "product_attributes.csv").set_index("series_id").loc[ids]
# simple numeric: group by store format? We'll skip attrs for now


def build_features(hist, av, woyh, target_idx):
    # hist: (n, t) demand observed (0 for unavail), av (n,t), woyh (t,)
    # returns feature matrix (n, F) for predicting target at position target_idx (0-based in extended timeline, >= hist len? actually target is future week not in hist)
    # We compute lags relative to target position pos=target_idx. Lag k = hist[:, pos-k] if available.
    pos = target_idx
    t = hist.shape[1]
    # ensure woyh length at least pos+1
    woyh = np.asarray(woyh).copy()
    while woyh.shape[0] <= pos:
        woyh = np.append(woyh, (int(woyh[-1]) % 52) + 1)
    feats = {}
    feats["woy"] = woyh[pos]
    feats["woy_sin"] = np.sin(2 * np.pi * woyh[pos] / 52)
    feats["woy_cos"] = np.cos(2 * np.pi * woyh[pos] / 52)
    # lags
    for k in [1, 2, 3, 4, 8, 13, 26, 52]:
        idx = pos - k
        if 0 <= idx < t:
            col = np.where(av[:, idx] > 0.5, hist[:, idx], np.nan)
            # impute with series mean
            col = np.where(np.isnan(col), series_mean, col)
        else:
            col = np.full(n, np.nan)
            col = np.where(np.isnan(col), series_mean, col)
        feats[f"lag{k}"] = col
    # rolling means
    for w in [4, 13, 26, 52]:
        # mean of last w available weeks before pos
        cols = []
        for k in range(1, w + 1):
            idx = pos - k
            if 0 <= idx < t:
                cols.append(np.where(av[:, idx] > 0.5, hist[:, idx], np.nan))
        if cols:
            arr = np.column_stack(cols)
            rm = np.nanmean(arr, axis=1)
        else:
            rm = series_mean
        rm = np.where(np.isnan(rm), series_mean, rm)
        feats[f"rmean{w}"] = rm
    # series stats
    feats["series_mean"] = series_mean
    feats["series_std"] = series_std
    feats["series_cv"] = series_cv
    feats["series_fraczero"] = series_fraczero
    # availability of pos? unknown; set 1
    X = pd.DataFrame(feats)
    return X


# Build training data
def build_trainset(horizon):
    Xs = []
    ys = []
    for t in range(52, T - horizon):
        target_pos = t + horizon  # position in extended timeline (0-based week index)
        # target week must be available
        avail_target = a[:, t + horizon]  # 0/1
        # features based on history up to t (weeks 0..t-1? hist length t)
        hist = y[:, :t]
        av = a[:, :t]
        woyh = woy[:t]
        X = build_features(
            hist, av, woyh, t + horizon
        )  # target pos = t+horizon, but hist length t; positions 0..t-1 known, target pos beyond
        # Wait build_features uses pos=target_idx and accesses hist[:, pos-k]. If pos>=t, lag1 pos-1 = t+horizon-1 may exceed hist length. Need hist extended? For training, target future position t+horizon; lags should be relative to target, using weeks before target, some may be within hist (pos-k < t). If pos-k >= t, unknown (should be nan). But for target at t+horizon, pos-k = t+horizon-k. For k<horizon, this is >=t (future unknown) -> nan. That's okay; those lags unavailable. But typically we want lags up to target. We'll allow nan->series_mean.
        ytarget = y[:, t + horizon]
        # mask available targets
        mask = avail_target > 0.5
        X = X.loc[mask]
        yt = ytarget[mask]
        Xs.append(X)
        ys.append(yt)
    Xall = pd.concat(Xs, ignore_index=True)
    yall = np.concatenate(ys)
    return Xall, yall


for h in [1, 2, 3]:
    Xall, yall = build_trainset(h)
    print(f"h{h}: samples {len(yall)}")

# Train models
models_mean = {}
models_q = {}
for h in [1, 2, 3]:
    Xall, yall = build_trainset(h)
    # mean
    dtrain = lgb.Dataset(Xall, label=yall)
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
    }
    m = lgb.train(params, dtrain, num_boost_round=200)
    models_mean[h] = m
    # quantile 0.833
    paramsq = {
        "objective": "quantile",
        "alpha": 0.833,
        "metric": "quantile",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
    }
    mq = lgb.train(paramsq, dtrain, num_boost_round=200)
    models_q[h] = mq
    print(f"trained h{h}")


# Evaluate in backtest using full history
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


# feature computation for game: use extended history (base + revealed)
def make_decide(models_mean, models_q):
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
        # forecast h1..h3
        fmeans = []
        fqs = []
        for hh in [1, 2, 3]:
            target_pos = (
                Tcur + hh - 1
            )  # position in extended timeline for target week (0-based)
            # build feature df for all series
            X = build_features(hist, av, woyh, target_pos)
            # align columns to training feature order
            # predict
            fm = models_mean[hh].predict(X)
            fq = models_q[hh].predict(X)
            fmeans.append(np.maximum(fm, 0))
            fqs.append(np.maximum(fq, 0))
        fmeans = np.array(fmeans)
        fqs = np.array(fqs)
        # a,b use mean; c use quantile
        a_fc = fmeans[0]
        b_fc = fmeans[1]
        c_fc = fqs[2]
        on_hand, arr1, arr2 = state.T
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - 1.0 * (a_fc + b_fc))
        deficit = c_fc - projected
        return np.clip(deficit, 0, None)

    return decide


decide = make_decide(models_mean, models_q)
for sw in [100, 130]:
    sc = backtest(decide, y, start_week=sw, initial=st0)
    print(f"sw={sw} score={sc:.0f}")
