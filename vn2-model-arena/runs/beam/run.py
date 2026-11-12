import pandas as pd, numpy as np, time, os
from lightgbm import LGBMRegressor

BASE = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
DATA = BASE + "/data"
RUN = BASE + "/runs/beam"
t0 = time.time()

# Load
sales = pd.read_csv(DATA + "/sales.csv")
avail = pd.read_csv(DATA + "/availability.csv")
attrs = pd.read_csv(DATA + "/product_attributes.csv")
df = sales.merge(avail, on=["series_id", "week"])
df = df.merge(attrs, on="series_id")
df["week"] = pd.to_datetime(df["week"])
df["is_available"] = df["is_available"].astype(int)
df = df.sort_values(["series_id", "week"]).reset_index(drop=True)

# attribute encoding
for c in attrs.columns:
    if c == "series_id":
        continue
    df[c] = df[c].astype("category").cat.codes

series_ids = sorted(df.series_id.unique())
sid_to_idx = {s: i for i, s in enumerate(series_ids)}

LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLLS = [4, 8, 13, 26, 52]


def featurize(panel):
    # panel: dataframe with series_id, week, y, is_available, attrs; sorted
    g = panel.groupby("series_id", group_keys=False)
    out = panel.copy()
    # calendar of target week
    out["woy"] = out["week"].dt.isocalendar().week.astype(int)
    out["month"] = out["week"].dt.month
    out["year"] = out["week"].dt.year
    # lag y
    for l in LAGS:
        out[f"lag_y_{l}"] = g["y"].shift(l)
    # rolling on shifted y
    shifted = g["y"].shift(1)
    for w in ROLLS:
        out[f"roll_mean_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        out[f"roll_std_{w}"] = shifted.groupby(panel["series_id"]).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
    # nonzero count rolling
    nz = (panel["y"] > 0).astype(float)
    nz_shifted = g_y = nz.groupby(panel["series_id"]).shift(1) if False else None
    # simpler: compute via groupby transform
    out["roll_nz_8"] = (
        nz.groupby(panel["series_id"])
        .shift(1)
        .groupby(panel["series_id"])
        .transform(lambda s: s.rolling(8, min_periods=1).sum())
    )
    # availability lags/rolling
    for l in [1, 2, 4]:
        out[f"lag_av_{l}"] = g["is_available"].shift(l)
    av_shifted = g["is_available"].shift(1)
    out["roll_av_8"] = av_shifted.groupby(panel["series_id"]).transform(
        lambda s: s.rolling(8, min_periods=1).mean()
    )
    # series index
    out["sid_idx"] = out["series_id"].map(sid_to_idx)
    # expanding mean of y (shifted)
    out["exp_mean"] = shifted.groupby(panel["series_id"]).transform(
        lambda s: s.expanding().mean()
    )
    return out


# Build training data: use observed panel only
train_panel = df.copy()
train_feat = featurize(train_panel)
# target is y (already in row). Features use shifted, so okay. But rows where target week's y known; features from past.
# Drop rows with no lag (first 52) to reduce NaNs? LightGBM can handle NaN; keep all but maybe first.
X_cols = [c for c in train_feat.columns if c not in ["series_id", "week", "y"]]
# We'll store feature columns list
feature_cols = [c for c in train_feat.columns if c not in ["series_id", "week", "y"]]
print("n features", len(feature_cols))


def make_xy(feat_df):
    X = feat_df[feature_cols]
    y = feat_df["y"].values
    return X, y


# Validation: hold out last 8 weeks
val_weeks = sorted(df.week.unique())[-8:]
train_mask = ~df.week.isin(val_weeks)


# For validation, train on weeks before val, forecast recursively through val weeks, compare.
def train_model(train_df):
    feat = featurize(train_df)
    X = feat[feature_cols]
    y = feat["y"].values
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(X, y)
    return model


# Validation run
print("Validation training...")
# training panel for validation: observed up to before val
max_train_week = df.week.max() if False else None
cut_week = sorted(df.week.unique())[-9]  # last train week before val
train_df = df[df.week <= cut_week].copy()
model = train_model(train_df)
# forecast val weeks recursively
hist = train_df.copy()
preds_val = {}
for i, w in enumerate(val_weeks):
    # append placeholder
    new_rows = pd.DataFrame(
        {
            "series_id": series_ids,
            "week": pd.to_datetime(w),
        }
    )
    # merge attrs
    new_rows = new_rows.merge(attrs, on="series_id")
    new_rows["y"] = np.nan
    new_rows["is_available"] = 1
    # encode attrs
    for c in attrs.columns:
        if c == "series_id":
            continue
        new_rows[c] = (
            new_rows[c].astype("category").cat.codes
            if new_rows[c].dtype == "object"
            else new_rows[c]
        )
    # Actually attrs already encoded in df; new_rows from attrs raw. Let's instead build from template.
    hist = pd.concat([hist, new_rows[hist.columns]], ignore_index=True)
    hist = hist.sort_values(["series_id", "week"]).reset_index(drop=True)
    feat = featurize(hist)
    target_feat = feat[feat.week == pd.to_datetime(w)]
    X = target_feat[feature_cols]
    yhat = model.predict(X)
    yhat = np.clip(yhat, 0, None)
    # fill
    mask = hist.week == pd.to_datetime(w)
    hist.loc[mask, "y"] = yhat
    preds_val[w] = dict(zip(target_feat.series_id, yhat))

# compute wmape
actual = df[df.week.isin(val_weeks)][["series_id", "week", "y"]]
pred_list = []
for w in val_weeks:
    for s in series_ids:
        pred_list.append({"series_id": s, "week": w, "y_hat": preds_val[w][s]})
pred_df = pd.DataFrame(pred_list)
pred_df["week"] = pd.to_datetime(pred_df["week"])
m = actual.merge(pred_df, on=["series_id", "week"])
wmape = np.abs(m.y - m.y_hat).sum() / m.y.sum()
print("Val WMAPE model:", wmape)
print("time", time.time() - t0)

# Seasonal naive val
# For each val week, y_hat = y at w-52 if available else recent mean
sn_preds = {}
for w in val_weeks:
    wdt = pd.to_datetime(w)
    prev = wdt - pd.Timedelta(weeks=52)
    # map
    hist_map = df[df.week == prev].set_index("series_id").y
    for s in series_ids:
        v = hist_map.get(s, np.nan)
        if pd.isna(v):
            v = train_df[train_df.series_id == s].y.tail(4).mean()
        sn_preds.setdefault(w, {})[s] = max(0.0, float(v))
pred_list = []
for w in val_weeks:
    for s in series_ids:
        pred_list.append({"series_id": s, "week": w, "y_hat": sn_preds[w][s]})
pred_df = pd.DataFrame(pred_list)
pred_df["week"] = pd.to_datetime(pred_df["week"])
m = actual.merge(pred_df, on=["series_id", "week"])
wmape_sn = np.abs(m.y - m.y_hat).sum() / m.y.sum()
print("Val WMAPE SN:", wmape_sn)

# Blend weight search
best = None
for a in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    # blend model and sn
    pl = []
    for w in val_weeks:
        for s in series_ids:
            mv = preds_val[w][s]
            sv = sn_preds[w][s]
            pl.append({"series_id": s, "week": w, "y_hat": a * mv + (1 - a) * sv})
    pdf = pd.DataFrame(pl)
    pdf["week"] = pd.to_datetime(pdf["week"])
    mm = actual.merge(pdf, on=["series_id", "week"])
    w = np.abs(mm.y - mm.y_hat).sum() / mm.y.sum()
    print(f"blend a(model)={a}: WMAPE={w:.4f}")
    if best is None or w < best[1]:
        best = (a, w)
print("best blend", best)
blend_a = best[0]
print("time", time.time() - t0)
