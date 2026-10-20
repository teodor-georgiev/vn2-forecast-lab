# -*- coding: utf-8 -*-
"""Round 2 ordering plan: latent-demand LightGBM forecast + quantile safety-stock targets."""

import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\mimo-v2.6-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
t0 = time.time()

sales = pd.read_csv(DATA + r"\sales.csv")
avail = pd.read_csv(DATA + r"\availability.csv")
attrs = pd.read_csv(DATA + r"\product_attributes.csv")
init = pd.read_csv(DATA + r"\initial_state.csv")

weeks = sorted(sales["week"].unique())
W = len(weeks)  # 157
widx = {w: i for i, w in enumerate(weeks)}
sids = sorted(sales["series_id"].unique())
N = len(sids)
sidx = {s: i for i, s in enumerate(sids)}

Y = np.zeros((W, N))
AV = np.ones((W, N), dtype=bool)
for sid, w, y in zip(sales["series_id"], sales["week"], sales["y"]):
    Y[widx[w], sidx[sid]] = y
for sid, w, a in zip(avail["series_id"], avail["week"], avail["is_available"]):
    AV[widx[w], sidx[sid]] = bool(a)

# latent demand: unavailable weeks -> NaN, impute with series rolling seasonal guess
Yl = Y.copy()
Yl[~AV] = np.nan
# impute: per series, use median of same week-of-year (or trailing mean of available)
raw = pd.DataFrame(Yl, index=range(W), columns=range(N))
seasonal_fill = raw.rolling(
    53, min_periods=6, center=True
).mean()  # local smoothed fill
Yf = raw.fillna(seasonal_fill).fillna(method="bfill").fillna(method="ffill").fillna(0.0)
Yf = Yf.values

np.save(RUN + r"\Yf.npy", Yf)
np.save(RUN + r"\AV.npy", AV)


# ---------- features ----------
def build_features(Ym, av, t):
    """features up to week index t (inclusive) for predicting week t+1"""
    f = {}
    for lag in (1, 2, 3, 4, 8, 13, 26, 52):
        f[f"lag{lag}"] = Ym[t - lag] if t - lag >= 0 else np.zeros(Ym.shape[1])
    f["rm4"] = Ym[max(0, t - 3) : t + 1].mean(0)
    f["rm13"] = Ym[max(0, t - 12) : t + 1].mean(0)
    f["rm26"] = Ym[max(0, t - 25) : t + 1].mean(0)
    f["rm52"] = Ym[max(0, t - 51) : t + 1].mean(0)
    f["rmax13"] = Ym[max(0, t - 12) : t + 1].max(0)
    f["std13"] = Ym[max(0, t - 12) : t + 1].std(0)
    f["zero_frac26"] = (Ym[max(0, t - 25) : t + 1] == 0).mean(0)
    f["last_avail"] = av[t].astype(float)
    return pd.DataFrame(f, index=range(Ym.shape[1]))


# calendar for future weeks: last week 2022-01-03; game weeks continue Mondays
last_date = pd.Timestamp(weeks[-1])
future_dates = [last_date + pd.Timedelta(weeks=1 + i) for i in range(8)]
dow_sin = lambda d: np.sin(2 * np.pi * d.isocalendar().week / 52.345)

# attrs encoding
attrs = attrs.set_index("series_id").loc[sids]
attr_cols = {}
for c in [
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
]:
    attr_cols[c] = attrs[c].astype("category").cat.codes.values
attr_df = pd.DataFrame(attr_cols, index=range(N))

import lightgbm as lgb


def train_and_forecast(train_end, horizon, seed=0, num_boost=600):
    """train on weeks 0..train_end inclusive, forecast horizon weeks recursively.
    Returns (mu_train_pred, forecasts (horizon,N))."""
    rows_X, rows_y, rows_w = [], [], []
    for t in range(53, train_end + 1):
        X = build_features(Yf, AV, t)
        X["avail_frac4"] = AV[max(0, t - 3) : t + 1].mean(0)
        X["avail_frac13"] = AV[max(0, t - 12) : t + 1].mean(0)
        X["wky"] = np.arange(N) % 7  # weak per-series id proxy (regularity)
        X["wky2"] = np.arange(N) % 13
        for c, v in attr_cols.items():
            X[c] = v
        d = (
            pd.Timestamp(weeks[t + 1])
            if t + 1 < W
            else last_date + pd.Timedelta(weeks=t + 1 - W + 1)
        )
        iso = pd.Timestamp(weeks[min(t, W - 1)]).isocalendar().week
        X["isoweek"] = iso
        X["isoweek_sin"] = np.sin(2 * np.pi * iso / 52.345)
        X["isoweek_cos"] = np.cos(2 * np.pi * iso / 52.345)
        mask = AV[t + 1] if t + 1 < W else np.ones(N, bool)
        rows_X.append(X[mask.values])
        rows_y.append(np.log1p(Yf[t + 1][mask.values]))
    Xtr = pd.concat(rows_X, ignore_index=True)
    ytr = np.concatenate(rows_y)
    dtr = lgb.Dataset(Xtr, label=ytr)
    params = dict(
        objective="regression",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=200,
        seed=seed,
        num_threads=6,
        verbosity=-1,
    )
    model = lgb.train(params, dtr, num_boost_round=num_boost)
    # recursive forecast
    Ym = Yf.copy()
    avm = AV.copy()
    fc = np.zeros((horizon, N))
    for h in range(horizon):
        t = train_end + h
        X = build_features(Ym, avm, t)
        X["avail_frac4"] = np.ones(N)
        X["avail_frac13"] = np.ones(N)
        X["wky"] = np.arange(N) % 7
        X["wky2"] = np.arange(N) % 13
        for c, v in attr_cols.items():
            X[c] = v
        iso = (
            pd.Timestamp(weeks[min(t, W - 1)]).isocalendar().week
            if t < W
            else future_dates[h].isocalendar().week
        )
        X["isoweek"] = iso
        X["isoweek_sin"] = np.sin(2 * np.pi * iso / 52.345)
        X["isoweek_cos"] = np.cos(2 * np.pi * iso / 52.345)
        mu = np.expm1(model.predict(X))
        mu = np.clip(mu, 0, None)
        fc[h] = mu
        Ym = np.vstack([Ym, mu])
        avm = np.vstack([avm, np.ones((1, N), bool)])
    return model, fc


print("train full...", flush=True)
model, fc = train_and_forecast(W - 1, 8)
print("full fc done", time.time() - t0, flush=True)
np.save(RUN + r"\fc_full.npy", fc)

# ---------- validation residuals for quantile factor ----------
print("train valid...", flush=True)
model_v, fc_v = train_and_forecast(W - 9, 8)  # predict weeks W-8..W-1
act = Yf[W - 8 : W]
ratio = (act + 0.5) / (fc_v + 0.5)
lev = fc_v
print("ratio q83 overall", np.quantile(ratio, 0.83), flush=True)
np.save(RUN + r"\fc_valid.npy", fc_v)


# bucketed multiplier at quantile q
def bucket_k(q, nb=6):
    m = lev.ravel()
    r = ratio.ravel()
    qs = np.quantile(m, np.linspace(0, 1, nb + 1)[1:-1])
    out = np.zeros(m.shape[0])
    k = np.searchsorted(qs, m)
    for b in range(nb + 1):
        sel = k == b
        if sel.any():
            out[sel] = np.quantile(r[sel], q)
    return out, qs


print("elapsed", time.time() - t0, flush=True)
