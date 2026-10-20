import pandas as pd, numpy as np, time, sys

BASE = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
DATA = BASE + "/data"
RUN = BASE + "/runs/beam"
sys.path.insert(0, BASE + "/tools")
from inventory_sim import simulate, reference_orders
from lightgbm import LGBMRegressor

sales = pd.read_csv(DATA + "/sales.csv")
avail = pd.read_csv(DATA + "/availability.csv")
attrs = pd.read_csv(DATA + "/product_attributes.csv")
df = sales.merge(avail, on=["series_id", "week"])
df = df.merge(attrs, on="series_id")
df["week"] = pd.to_datetime(df["week"])
df["is_available"] = df["is_available"].astype(int)
df = df.sort_values(["series_id", "week"]).reset_index(drop=True)
for c in attrs.columns:
    if c == "series_id":
        continue
    df[c] = df[c].astype("category").cat.codes
series_ids = sorted(df.series_id.unique())
n = len(series_ids)
sid_to_idx = {s: i for i, s in enumerate(series_ids)}

LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLLS = [4, 8, 13, 26, 52]


def featurize(panel):
    g = panel.groupby("series_id", group_keys=False)
    out = panel.copy()
    out["woy"] = out["week"].dt.isocalendar().week.astype(int)
    out["month"] = out["week"].dt.month
    out["year"] = out["week"].dt.year
    for l in LAGS:
        out[f"lag_y_{l}"] = g["y"].shift(l)
    shifted = g["y"].shift(1)
    for w in ROLLS:
        out[f"roll_mean_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        out[f"roll_std_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
    nz = (panel["y"] > 0).astype(float)
    out["roll_nz_8"] = (
        nz.groupby(panel["series_id"])
        .shift(1)
        .groupby(panel["series_id"])
        .transform(lambda s: s.rolling(8, min_periods=1).sum())
    )
    for l in [1, 2, 4]:
        out[f"lag_av_{l}"] = g["is_available"].shift(l)
    av_shifted = g["is_available"].shift(1)
    out["roll_av_8"] = av_shifted.groupby(panel["series_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    out["sid_idx"] = out["series_id"].map(sid_to_idx)
    out["exp_mean"] = shifted.groupby(panel["series_id"]).transform(
        lambda s: s.expanding().mean()
    )
    out["trend_4_13"] = out["roll_mean_4"] - out["roll_mean_13"]
    out["cv_26"] = out["roll_std_26"] / (out["roll_mean_26"] + 1e-6)
    tmp = panel[["series_id", "week", "y"]].copy()
    tmp["woy"] = panel["week"].dt.isocalendar().week.astype(int).values
    tmp["y_shift"] = tmp.groupby("series_id")["y"].shift(1)
    tmp["woy_hist_mean"] = tmp.groupby(["series_id", "woy"])["y_shift"].transform(
        lambda s: s.expanding().mean()
    )
    out["woy_hist_mean"] = tmp["woy_hist_mean"].values
    for w in [4, 13]:
        out[f"roll_max_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).max()
        )
        out[f"roll_min_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).min()
        )
    return out


feature_cols = None


def train_model(train_df):
    feat = featurize(train_df)
    fc = [c for c in feat.columns if c not in ["series_id", "week", "y"]]
    global feature_cols
    feature_cols = fc
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(feat[fc], feat["y"].values)
    return model, fc


# Validation window: last 8 weeks
val_weeks = sorted(df.week.unique())[-8:]
cut_week = sorted(df.week.unique())[-9]
train_df = df[df.week <= cut_week].copy()
t0 = time.time()
model, fc = train_model(train_df)
print("trained", time.time() - t0)

# forecast val weeks recursively
hist = train_df.copy()
F = np.zeros((8, n))
attr_lookup = df.drop_duplicates("series_id")[
    ["series_id"] + list(attrs.columns[1:])
].copy()
# ensure attr codes already in df; attr_lookup from df
attr_lookup = df.drop_duplicates("series_id")[
    ["series_id"] + [c for c in df.columns if c in attrs.columns and c != "series_id"]
].copy()
for i, w in enumerate(val_weeks):
    new_rows = pd.DataFrame({"series_id": series_ids, "week": pd.to_datetime(w)})
    new_rows = new_rows.merge(attr_lookup, on="series_id", how="left")
    new_rows["y"] = np.nan
    new_rows["is_available"] = 1
    new_rows = new_rows[hist.columns]
    hist = (
        pd.concat([hist, new_rows], ignore_index=True)
        .sort_values(["series_id", "week"])
        .reset_index(drop=True)
    )
    feat = featurize(hist)
    target_feat = feat[feat.week == pd.to_datetime(w)]
    X = target_feat[fc]
    yhat = np.clip(model.predict(X), 0, None)
    mask = hist.week == pd.to_datetime(w)
    hist.loc[mask, "y"] = yhat
    F[i] = yhat
    print("horizon", i + 1, time.time() - t0)

# actual demand
actual_D = np.zeros((8, n))
for i, w in enumerate(val_weeks):
    sub = df[df.week == pd.to_datetime(w)]
    m = sub.set_index("series_id").y
    actual_D[i] = [m.get(s, 0.0) for s in series_ids]

# seasonal naive forecast F_sn
F_sn = np.zeros((8, n))
for i, w in enumerate(val_weeks):
    wdt = pd.to_datetime(w)
    prev = wdt - pd.Timedelta(weeks=52)
    sub = df[df.week == prev]
    if len(sub) > 0:
        m = sub.set_index("series_id").y
        F_sn[i] = [m.get(s, np.nan) for s in series_ids]
    else:
        F_sn[i] = 0.0
F_sn = np.nan_to_num(F_sn, nan=0.0)
F_sn = np.clip(F_sn, 0, None)

# initial state zeros
init = np.zeros((n, 3))


def gen_orders(F, init, scale, projection, buffer):
    # fixed plan using F as demand propagation
    state = init.copy().astype(float)
    orders = np.zeros((6, n))  # only 6 order weeks; but F has 8
    # We need forecast triples for weeks 1..6: F[w],F[w+1],F[w+2] (0-based)
    for w in range(6):
        fc3 = np.array([F[w], F[w + 1], F[w + 2]])  # (3,n)
        q = reference_orders(
            fc3, state, scale=scale, projection=projection, buffer=buffer
        )
        orders[w] = q
        # propagate state with forecasted demand F[w]
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - F[w])
        state = np.column_stack((end, arr2, q))
    return orders


# Score function
def score_orders(orders, demand, init):
    sim = simulate(demand, init, orders)
    return sim["score"]


scale = 0.9
projection = 1.0
buffer = 0.25
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    Fb = alpha * F + (1 - alpha) * F_sn
    orders = gen_orders(Fb, init, scale, projection, buffer)
    sc = score_orders(orders, actual_D, init)
    print(f"blend alpha(model)={alpha}: score={sc:.1f}")
print("time", time.time() - t0)
