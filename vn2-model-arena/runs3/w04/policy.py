import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
DATA = os.path.join(ARENA, "data")

# ---------- load history ----------
_sales = pd.read_csv(os.path.join(DATA, "sales.csv"))
_avail = pd.read_csv(os.path.join(DATA, "availability.csv"))

series_list = sorted(_sales["series_id"].unique().tolist())
_sid_to_idx = {s: i for i, s in enumerate(series_list)}
n = len(series_list)
weeks = sorted(_sales["week"].unique().tolist())
T = len(weeks)
_w2t = {w: t for t, w in enumerate(weeks)}

D = np.zeros((n, T))
A = np.zeros((n, T), dtype=bool)
for _, r in _sales.iterrows():
    D[_sid_to_idx[r["series_id"]], _w2t[r["week"]]] = r["y"]
for _, r in _avail.iterrows():
    A[_sid_to_idx[r["series_id"]], _w2t[r["week"]]] = bool(r["is_available"])

# censoring correction: fill stockout weeks with seasonal mean of available weeks
Ycorr = D.copy()
_woy = np.array([t % 52 for t in range(T)])
for i in range(n):
    for t in range(T):
        if not A[i, t]:
            mask = (_woy == _woy[t]) & A[i]
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


def _woy_sin_cos(idx):
    ang = 2 * np.pi * (idx % 52) / 52.0
    return float(np.sin(ang)), float(np.cos(ang))


def _roll(y, m):
    arr = y[-m:] if m <= len(y) else y
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))


def make_features(y, target_idx, series_idx):
    cur = len(y)

    def lag(k):
        return float(y[-k]) if k <= cur else 0.0

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
    m4 = _roll(y, 4)
    m8 = _roll(y, 8)
    m13 = _roll(y, 13)
    m26 = _roll(y, 26)
    m52 = _roll(y, 52)
    trend = float(np.mean(y[-4:]) - np.mean(y[-8:-4])) if cur >= 8 else 0.0
    if cur >= 13:
        slope = float(np.polyfit(np.arange(13), y[-13:], 1)[0])
    else:
        slope = 0.0
    ratio = float(np.mean(y[-4:]) / (series_mean[series_idx] + 1e-6))
    si, ci = _woy_sin_cos(target_idx)
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


# ---------- train LightGBM per horizon ----------
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
            feats = make_features(y[: t + 1], t + h, i)
            X.append(feats)
            yy.append(D[i, t + h])
    X = np.array(X, dtype=float)
    yy = np.array(yy, dtype=float)
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


def _forecast(week_index, state, revealed):
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
    for hi, h in enumerate([1, 2, 3]):
        rows = []
        for i in range(n_series):
            y = ext[i, :cur_len]
            target_idx = cur_len - 1 + h
            rows.append(make_features(y, target_idx, i))
        X = np.array(rows, dtype=float)
        pred = models[h].predict(X)
        # seasonal naive for same week-of-year
        sn = np.zeros(n_series)
        tw = (cur_len - 1 + h) % 52
        for i in range(n_series):
            y = ext[i, :cur_len]
            hist_idx = [t for t in range(cur_len) if t % 52 == tw]
            sn[i] = (
                float(np.mean(y[hist_idx])) if len(hist_idx) > 0 else float(np.mean(y))
            )
        out[hi, :] = np.maximum(0.0, 0.8 * pred + 0.2 * sn)
    return out


def decide(week_index, state, revealed):
    state = np.asarray(state, dtype=float)
    n_series = state.shape[0]
    if n_series != n:
        # fallback for unknown series (e.g. selfcheck): order zeros
        return np.zeros(n_series)
    fc = _forecast(int(week_index), state, revealed)
    a, b, c = [np.maximum(0.0, fc[i]) for i in range(3)]
    on_hand, arr1, arr2 = state.T
    buffer = 1.2
    projection = 1.0
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0.0)) - projected
    return np.clip(deficit, 0.0, None)
