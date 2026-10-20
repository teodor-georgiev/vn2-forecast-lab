import pandas as pd, numpy as np, time
from lightgbm import LGBMRegressor

BASE = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
DATA = BASE + "/data"
RUN = BASE + "/runs/beam"
t0 = time.time()

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
    return out


feature_cols = [c for c in featurize(df).columns if c not in ["series_id", "week", "y"]]

models = []
for seed in [42, 7, 2024]:
    m = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    feat = featurize(df)
    m.fit(feat[feature_cols], feat["y"].values)
    models.append(m)
print("trained", time.time() - t0)

attr_lookup = df.drop_duplicates("series_id")[
    ["series_id"] + list(attrs.columns[1:])
].copy()

hist = df.copy()
last_week = df.week.max()
future_weeks = [last_week + pd.Timedelta(weeks=h) for h in range(1, 9)]

all_preds = []
for h, w in enumerate(future_weeks, 1):
    new_rows = pd.DataFrame({"series_id": series_ids, "week": w})
    new_rows = new_rows.merge(attr_lookup, on="series_id", how="left")
    new_rows["y"] = np.nan
    new_rows["is_available"] = 1
    new_rows = new_rows[hist.columns]
    hist = pd.concat([hist, new_rows], ignore_index=True)
    hist = hist.sort_values(["series_id", "week"]).reset_index(drop=True)
    feat = featurize(hist)
    target_feat = feat[feat.week == w]
    X = target_feat[feature_cols]
    yhat = np.mean([m.predict(X) for m in models], axis=0)
    yhat = np.clip(yhat, 0, None)
    mask = hist.week == w
    hist.loc[mask, "y"] = yhat
    for s, yv in zip(target_feat.series_id, yhat):
        all_preds.append({"series_id": s, "horizon": h, "y_hat": float(yv)})
    print(f"horizon {h} done", time.time() - t0)

out = pd.DataFrame(all_preds, columns=["series_id", "horizon", "y_hat"])
out.to_csv(RUN + "/predictions.csv", index=False)
print("saved", out.shape, time.time() - t0)
assert out.shape == (599 * 8, 3)
assert not out.y_hat.isna().any()
assert (out.y_hat >= 0).all()
assert out.duplicated(["series_id", "horizon"]).sum() == 0
print("checks ok")
