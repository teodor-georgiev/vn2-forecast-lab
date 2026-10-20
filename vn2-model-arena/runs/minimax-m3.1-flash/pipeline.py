import numpy as np, pandas as pd, lightgbm as lgb, time, os, sys, json

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
T0 = time.time()

sales = pd.read_csv(D + r"\sales.csv")
av = pd.read_csv(D + r"\availability.csv")
attrs = pd.read_csv(D + r"\product_attributes.csv")
sales["week"] = pd.to_datetime(sales["week"])
av["week"] = pd.to_datetime(av["week"])
sids = sorted(sales["series_id"].unique())
sid_idx = {s: i for i, s in enumerate(sids)}
weeks = sorted(sales["week"].unique())
widx = {w: i for i, w in enumerate(weeks)}
S, T = len(sids), len(weeks)
Y = np.full((S, T), np.nan)
A = np.zeros((S, T), bool)
sales["ti"] = sales["week"].map(widx)
sales["si"] = sales["series_id"].map(sid_idx)
Y[sales["si"].values, sales["ti"].values] = sales["y"].values
av["ti"] = av["week"].map(widx)
av["si"] = av["series_id"].map(sid_idx)
A[av["si"].values, av["ti"].values] = av["is_available"].values.astype(bool)
assert np.isfinite(Y[A]).all()
wk = pd.DatetimeIndex(weeks)
woy = wk.isocalendar().week.values.astype(int)
yr = wk.year.values
# extend calendar 8 weeks past the end so future target weeks can be indexed
fut = pd.date_range(wk[-1] + pd.Timedelta(days=7), periods=8, freq="7D")
woy = np.concatenate([woy, fut.isocalendar().week.values.astype(int)])
yr = np.concatenate([yr, fut.year.values])


# seasonality: harmonic on week of year
def harm(w, k):
    return np.sin(2 * np.pi * k * w / 52.18), np.cos(2 * np.pi * k * w / 52.18)


Yobs = np.where(A, Y, 0.0)
# ---- imputation of unavailable weeks (local mean of available values) ----
Yimp = Yobs.copy()
for i in range(S):
    row = Y[i]
    m = A[i]
    if m.all():
        continue
    s = pd.Series(np.where(m, row, np.nan))
    roll = s.rolling(5, center=True, min_periods=1).mean()
    roll = roll.ffill().bfill()
    fill = roll.values
    bad = ~m
    if np.isnan(fill[bad]).any():
        fb = np.nanmean(row[m]) if m.any() else 0.0
        fill = np.where(np.isnan(fill), fb, fill)
    Yimp[i, bad] = fill[bad]


def roll_mean(M, w):
    cs = np.cumsum(np.concatenate([np.zeros((M.shape[0], 1)), M], axis=1), axis=1)
    out = np.full(M.shape, np.nan)
    for t in range(M.shape[1]):
        lo = max(0, t - w + 1)
        out[:, t] = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)
    return out


def roll_std(M, w):
    m1 = roll_mean(M, w)
    m2 = roll_mean(M * M, w)
    v = np.maximum(m2 - m1 * m1, 0.0)
    return np.sqrt(v)


MIN_T = 52
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 26, 39, 52]
ROLLW = [4, 8, 13, 26, 52]


def build_features():
    cols, names = [], []
    for k in LAGS:
        c = np.full((S, T), np.nan)
        c[:, k:] = Yimp[:, :-k]
        cols.append(c)
        names.append(f"ilag{k}")
    for k in [1, 2, 4, 8, 13, 26, 52]:
        c = np.full((S, T), np.nan)
        c[:, k:] = Yobs[:, :-k]
        cols.append(c)
        names.append(f"obs_lag{k}")
    for w in ROLLW:
        cols.append(roll_mean(Yimp, w))
        names.append(f"rmean{w}")
    for w in [8, 26, 52]:
        cols.append(roll_std(Yimp, w))
        names.append(f"rstd{w}")
    # min/max
    for w in [13, 52]:
        cs = np.cumsum(np.concatenate([np.zeros((S, 1)), Yimp], axis=1), axis=1)
        mn = np.full((S, T), np.nan)
        mx = np.full((S, T), np.nan)
        for t in range(T):
            lo = max(0, t - w + 1)
            seg = Yimp[:, lo : t + 1]
            mn[:, t] = seg.min(axis=1)
            mx[:, t] = seg.max(axis=1)
        cols.append(mn)
        names.append(f"rmin{w}")
        cols.append(mx)
        names.append(f"rmax{w}")
    # availability features
    Af = A.astype(float)
    for w in [4, 8, 13, 26, 52]:
        cols.append(roll_mean(Af, w))
        names.append(f"avail{w}")
    cols.append(Af)
    names.append("avail_last")
    # expanding (causal) stats on available values
    Yi = np.where(A, Y, np.nan)
    valid = A.astype(float)
    csum = np.cumsum(Yi * valid, axis=1)
    vsum = np.cumsum(valid, axis=1)
    exp_mean = csum / np.maximum(vsum, 1)
    csum2 = np.cumsum(np.where(A, Y**2, 0.0), axis=1)
    var = np.maximum(csum2 / np.maximum(vsum, 1) - exp_mean**2, 0.0)
    cols += [exp_mean, np.sqrt(var), vsum / T]
    names += ["exp_mean", "exp_std", "exp_avail_frac"]
    # horizon-agnostic calendar of origin + target
    tt = np.tile(np.arange(T), (S, 1))
    cols.append(tt.astype(float))
    names.append("t")
    return cols, names


cols, names = build_features()
meta = attrs.set_index("series_id").loc[sids].reset_index()
CATS = [
    "series_id",
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
]


def make_frame(origins, h):
    """origins: array of t indices; h: horizon. returns X (n_rows, n_feat) df, meta, target"""
    rows = []
    for t in origins:
        for s in range(S):
            rows.append((s, t))
    s_arr = np.array([r[0] for r in rows])
    t_arr = np.array([r[1] for r in rows])
    X = np.column_stack([c[s_arr, t_arr] for c in cols]).astype(np.float32)
    df = pd.DataFrame(X, columns=names)
    tw = t_arr + h
    for c in CATS:
        df[c] = meta[c].values[s_arr]
    for c in CATS:
        df[c] = df[c].astype("category")
    # calendar of target week
    w_t = woy[tw]
    for k in (1, 2, 3):
        s_, c_ = (
            np.sin(2 * np.pi * k * w_t / 52.18),
            np.cos(2 * np.pi * k * w_t / 52.18),
        )
        df[f"woy_sin{k}"] = s_.astype(np.float32)
        df[f"woy_cos{k}"] = c_.astype(np.float32)
    df["targ_year"] = yr[tw].astype(np.float32)
    df["h"] = np.full(len(df), h, dtype=np.float32)
    twc = np.minimum(tw, T - 1)
    tgt = Y[s_arr, twc]
    avail_t = A[s_arr, twc]
    return df, s_arr, t_arr, tgt, avail_t


def avail_mult(s_arr, t_arr, w=8):
    """expected future availability for each row, from data up to origin t (exclusive of target week)"""
    tot = np.zeros(len(s_arr))
    cnt = 0
    for t in np.unique(t_arr):
        m = t_arr == t
        lo = max(0, t - w + 1)
        seg = A[:, lo : t + 1].mean(axis=1)
        tot[m] = seg[s_arr[m]]
    return tot


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    return dict(
        rmse=float(np.sqrt(np.mean(e**2))),
        mae=float(np.mean(np.abs(e))),
        wape=float(np.sum(np.abs(e)) / max(np.sum(y), 1e-9)),
        bias=float(np.mean(e)),
    )


PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=int(os.cpu_count() or 4),
)


def run_backtest(cut=149, use_mult=True, n_est=350, log_space=True):
    """train on origins with t+h <= cut-1, predict origins t=cut-1"""
    res = {}
    preds = []
    for h in range(1, 9):
        max_t = cut - 1 - h
        tr_origins = np.arange(MIN_T, max_t + 1)
        if len(tr_origins) == 0:
            continue
        Xtr, s_tr, t_tr, y_tr, av_tr = make_frame(tr_origins, h)
        mask = av_tr & np.isfinite(y_tr)
        Xtr = Xtr[mask]
        y_tr = y_tr[mask]
        yy = np.log1p(y_tr) if log_space else y_tr
        ds = lgb.Dataset(Xtr, label=yy, categorical_feature=CATS, free_raw_data=False)
        m = lgb.train(PARAMS, ds, num_boost_round=n_est)
        Xte, s_te, t_te, y_te, av_te = make_frame(np.array([cut - 1]), h)
        p = m.predict(Xte)
        if log_space:
            p = np.expm1(p)
        mult = avail_mult(s_te, t_te) if use_mult else np.ones(len(p))
        preds.append(
            pd.DataFrame(dict(sid=np.array(sids)[s_te], h=h, y=y_te, p=p * mult, mu=p))
        )
        print(f"  h={h} rows={len(Xtr)} t={round(time.time() - T0, 1)}", flush=True)
    return pd.concat(preds, ignore_index=True)


if __name__ == "__main__":
    print("cpu", os.cpu_count(), "elapsed", round(time.time() - T0, 1), flush=True)
    bt = run_backtest()
    y = bt["y"].values
    p_med = bt["p"].values
    res = np.log1p(np.maximum(bt["mu"].values, 0))
    print("naive last", metrics(y, np.tile(Yobs[:, 148], 8)))
    sn = np.concatenate([Yimp[:, 148 + h - 52] for h in range(1, 9)])
    print("naive s52 ", metrics(y, sn))
    print("MODEL median-ish", metrics(y, p_med))
    # smearing
    r = np.log1p(np.maximum(p_med, 0)) - np.log1p(np.maximum(y, 0))
    s2 = np.var(r)
    sm = np.expm1(np.log1p(np.maximum(p_med, 0)) + 0.5 * s2)
    print("MODEL smearing s2=%.4f" % s2, metrics(y, sm))
    bt.to_csv(RUN + r"\backtest_preds.csv", index=False)
    print("total elapsed", round(time.time() - T0, 1), flush=True)
