import numpy as np
import pandas as pd
import lightgbm as lgb
import time

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
OUT = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\mimo-v2.6-flash\predictions.csv"

t0 = time.time()
sales = pd.read_csv(f"{DATA}\\sales.csv", parse_dates=["week"])
avail = pd.read_csv(f"{DATA}\\availability.csv", parse_dates=["week"])
attrs = pd.read_csv(f"{DATA}\\product_attributes.csv")

sales["is_available"] = True
av = avail.merge(sales[["series_id", "week"]], on=["series_id", "week"], how="left")
av["in_sales"] = av["week"].isin(sales["week"].unique())
# treat availability rows not in sales as y=0 available/unavailable weeks
extra = av[~av["in_sales"]][["series_id", "week", "is_available"]].copy()
extra["y"] = 0.0
df = pd.concat(
    [sales[["series_id", "week", "y"]], extra[["series_id", "week", "y"]]],
    ignore_index=True,
)
df = df.merge(avail, on=["series_id", "week"], how="left")
df["is_available"] = df["is_available"].fillna(True)
df = df.sort_values(["series_id", "week"]).reset_index(drop=True)

weeks = sorted(df["week"].unique())
wmap = {w: i for i, w in enumerate(weeks)}
df["t"] = df["week"].map(wmap)
T = len(weeks)
H = 8
print("weeks:", T, "series:", df["series_id"].nunique(), "rows:", len(df), flush=True)

# attributes
cat_cols = [c for c in attrs.columns if c != "series_id"]
for c in cat_cols:
    attrs[c] = attrs[c].astype("category").cat.codes
df = df.merge(attrs, on="series_id", how="left")
df["is_available"] = df["is_available"].astype(int)

df = df.sort_values(["series_id", "t"]).reset_index(drop=True)
g = df.groupby("series_id")

lag_cols = {}
for L in [1, 2, 3, 4, 5, 8, 13, 26, 52, 78]:
    lag_cols[f"lag{L}"] = g["y"].shift(L)
for L in [1, 2, 3, 4, 13, 26, 52]:
    lag_cols[f"avlag{L}"] = g["is_available"].shift(L)
for name, col in lag_cols.items():
    df[name] = col

for W in [4, 8, 13, 26, 52]:
    df[f"rmean{W}"] = g["y"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )
df["rstd26"] = g["y"].transform(lambda s: s.shift(1).rolling(26, min_periods=2).std())
df["avrate8"] = g["is_available"].transform(
    lambda s: s.shift(1).rolling(8, min_periods=1).mean()
)
df["avrate26"] = g["is_available"].transform(
    lambda s: s.shift(1).rolling(26, min_periods=1).mean()
)
df["cnt_nonzero"] = g["y"].transform(
    lambda s: (
        s.shift(1)
        .rolling(52, min_periods=1)
        .apply(lambda x: np.count_nonzero(x), raw=True)
    )
)

wk = pd.Index(weeks)
dts = pd.Series(weeks)
df["woy"] = df["week"].dt.isocalendar().week.astype(int)
df["wday"] = df["week"].dt.dayofweek
df["month"] = df["week"].dt.month
df["trend"] = df["t"] / T

feat = [
    "lag1",
    "lag2",
    "lag3",
    "lag4",
    "lag5",
    "lag8",
    "lag13",
    "lag26",
    "lag52",
    "lag78",
    "avlag1",
    "avlag2",
    "avlag3",
    "avlag4",
    "avlag13",
    "avlag26",
    "avlag52",
    "rmean4",
    "rmean8",
    "rmean13",
    "rmean26",
    "rmean52",
    "rstd26",
    "avrate8",
    "avrate26",
    "cnt_nonzero",
    "woy",
    "wday",
    "month",
    "trend",
    "is_available",
] + cat_cols

df["y_c"] = df["y"].clip(lower=0)
train_mask = df["t"] <= T - 1 - 8  # hold out last 8 weeks
tr = df[train_mask & df["lag52"].notna()]
va = df[~train_mask & df["lag52"].notna()]
print("train rows:", len(tr), "valid rows:", len(va), flush=True)

dtr = lgb.Dataset(
    tr[feat], label=tr["y_c"], categorical_feature=cat_cols, free_raw_data=False
)
dva = lgb.Dataset(va[feat], label=va["y_c"], free_raw_data=False, reference=dtr)
params = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    verbosity=-1,
    num_threads=4,
    seed=7,
)
model = lgb.train(
    params,
    dtr,
    num_boost_round=800,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)
print("best iter:", model.best_iteration, flush=True)
# refit on all data with best iteration count
iters = max(model.best_iteration, 50)
dfull = lgb.Dataset(
    df[df["lag52"].notna()][feat],
    label=df[df["lag52"].notna()]["y_c"],
    categorical_feature=cat_cols,
)
model = lgb.train(params, dfull, num_boost_round=int(iters * 1.1))

# --- recursive forecast for horizons 1..8 ---
last_avail = df.sort_values("t").groupby("series_id")["is_available"].last()
series_ids = sorted(df["series_id"].unique())

# build frame of last T weeks per series plus future rows
hist = df[["series_id", "t", "y", "is_available"] + cat_cols].copy()
future_rows = []
for sid in series_ids:
    base = df[df["series_id"] == sid].iloc[-1]
    for h in range(1, H + 1):
        r = {
            "series_id": sid,
            "t": T - 1 + h,
            "y": np.nan,
            "is_available": int(last_avail[sid]),
        }
        for c in cat_cols:
            r[c] = base[c]
        future_rows.append(r)
fut = pd.DataFrame(future_rows)
full = (
    pd.concat([hist, fut], ignore_index=True)
    .sort_values(["series_id", "t"])
    .reset_index(drop=True)
)
full["week"] = full["t"].map(
    lambda i: weeks[i] if i < T else weeks[-1] + pd.Timedelta(weeks=int(i - T + 1))
)

gg = full.groupby("series_id")
for L in [1, 2, 3, 4, 5, 8, 13, 26, 52, 78]:
    full[f"lag{L}"] = gg["y"].shift(L)
for L in [1, 2, 3, 4, 13, 26, 52]:
    full[f"avlag{L}"] = gg["is_available"].shift(L)
for W in [4, 8, 13, 26, 52]:
    full[f"rmean{W}"] = gg["y"].transform(
        lambda s: s.shift(1).rolling(W, min_periods=1).mean()
    )
full["rstd26"] = gg["y"].transform(
    lambda s: s.shift(1).rolling(26, min_periods=2).std()
)
full["avrate8"] = gg["is_available"].transform(
    lambda s: s.shift(1).rolling(8, min_periods=1).mean()
)
full["avrate26"] = gg["is_available"].transform(
    lambda s: s.shift(1).rolling(26, min_periods=1).mean()
)
full["cnt_nonzero"] = gg["y"].transform(
    lambda s: (
        s.shift(1)
        .rolling(52, min_periods=1)
        .apply(lambda x: np.count_nonzero(x), raw=True)
    )
)
full["woy"] = full["week"].dt.isocalendar().week.astype(int)
full["wday"] = full["week"].dt.dayofweek
full["month"] = full["week"].dt.month
full["trend"] = full["t"] / T

pred_rows = []
for h in range(1, H + 1):
    sub = full[full["t"] == T - 1 + h]
    sub = sub.copy()
    # fallback where features missing: seasonal naive (t-52)
    p = model.predict(sub[feat])
    sn = sub["lag52"].values
    p = np.where(np.isnan(p), 0.0, p)
    # blend slight shrinkage toward seasonal naive / recent mean for safety
    rm = sub["rmean8"].values
    sn = np.where(np.isnan(sn), rm, sn)
    p = 0.85 * p + 0.15 * np.nan_to_num(sn, nan=0.0)
    p = np.clip(p, 0, None)
    sub["y_hat"] = p
    sub["horizon"] = h
    pred_rows.append(sub[["series_id", "horizon", "y_hat"]])

pred = pd.concat(pred_rows, ignore_index=True)


assert len(pred) == 599 * 8, len(pred)
assert (
    pred["y_hat"].notna().all()
    and np.isfinite(pred["y_hat"]).all()
    and (pred["y_hat"] >= 0).all()
)
assert not pred.duplicated(["series_id", "horizon"]).any()
pred = pred.sort_values(["series_id", "horizon"])[["series_id", "horizon", "y_hat"]]
pred.to_csv(OUT, index=False)
print("wrote", OUT, "rows", len(pred), "time", time.time() - t0, flush=True)
