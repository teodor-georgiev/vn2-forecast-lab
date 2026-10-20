import numpy as np
import pandas as pd
import lightgbm as lgb

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
RD = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\deepseek-v4.1-flash"
OUT = RD + r"\predictions.csv"
FALLBACK = RD + r"\pred_baseline.csv"

s = pd.read_csv(D + r"\sales.csv", parse_dates=["week"])
a = pd.read_csv(D + r"\availability.csv", parse_dates=["week"])
p = pd.read_csv(D + r"\product_attributes.csv")

S = s.pivot(index="series_id", columns="week", values="y").sort_index()
A = (
    a.pivot(index="series_id", columns="week", values="is_available")
    .reindex_like(S)
    .fillna(True)
    .astype(bool)
)
Y = S.where(A)  # NaN on unavailable weeks
YT = Y.T  # (week, series)
AT = A.T.astype(float)
series = S.index
weeks = S.columns
nw = len(weeks)
ns = len(series)

# ---- per-series static features ----
attr = p.set_index("series_id").reindex(series)
cats = {}
for c in [
    "ProductGroup",
    "Division",
    "Department",
    "DepartmentGroup",
    "StoreFormat",
    "Format",
]:
    cats[c] = pd.Categorical(attr[c].astype(str)).codes.astype(np.int32)

ser_mean = YT.mean().values
ser_med = YT.median().values
ser_nz = (YT.fillna(0) > 1e-9).mean().values
ar_all = AT.mean().values


# ---- per-origin feature matrices (week, series) ----
def roll(df, w):
    return df.rolling(w, min_periods=1).mean()


R4, R8, R13, R26, R52 = (roll(YT, w).values for w in (4, 8, 13, 26, 52))
STD13 = YT.rolling(13, min_periods=2).std().values
AR4, AR13, AR52 = (roll(AT, w).values for w in (4, 13, 52))
Z13 = (YT.fillna(0) <= 1e-9).rolling(13, min_periods=1).mean().values
lagm = {}
for k in (1, 2, 3, 4, 8, 13, 26, 52):
    v = YT.shift(k).values.copy()
    m = np.isnan(v)
    v[m] = np.where(
        np.isnan(ser_mean)[None, :].repeat(nw, 0)[m],
        0.0,
        np.repeat(ser_mean[None, :], nw, 0)[m],
    )
    lagm[k] = v


def feats(i):
    d = {}
    for k in (1, 2, 4, 8, 13, 26, 52):
        d["lag%d" % k] = lagm[k][i]
    for nm, M in (
        ("roll4", R4),
        ("roll8", R8),
        ("roll13", R13),
        ("roll26", R26),
        ("roll52", R52),
        ("ar4", AR4),
        ("ar13", AR13),
        ("ar52", AR52),
        ("std13", STD13),
        ("zf13", Z13),
    ):
        d[nm] = M[i]
    d["trend"] = R4[i] / (R13[i] + 0.05)
    d["ser_mean"] = ser_mean
    d["ser_med"] = np.nan_to_num(ser_med)
    d["ser_nz"] = ser_nz
    d["ar_all"] = ar_all
    for c in cats:
        d["cat_" + c] = cats[c]
    return d


CATCOLS = ["cat_" + c for c in cats]
origins = list(range(60, nw - 1))  # train origins
FC = feats(nw - 1)  # forecast origin = last observed week

woy = np.array([w.isocalendar().week for w in weeks])
years = np.array([w.year for w in weeks])

rows = []
for h in range(1, 9):
    idxs = [i for i in origins if i + h <= nw - 1]
    X = {k: np.concatenate([feats(i)[k] for i in idxs]) for k in FC}
    tw = np.array([i + h for i in idxs])
    X["h"] = np.repeat(h, len(idxs) * ns).astype(np.float32)
    X["wsin"] = np.repeat(np.sin(2 * np.pi * woy[tw] / 52.0), ns)
    X["wcos"] = np.repeat(np.cos(2 * np.pi * woy[tw] / 52.0), ns)
    X["yr"] = np.repeat(years[tw] - years[tw].min(), ns).astype(np.float32)
    ytr = np.concatenate([Y.values[:, tw_k] for tw_k in tw])
    ytr = np.nan_to_num(ytr)
    Xtr = pd.DataFrame(X)
    for c in CATCOLS:
        Xtr[c] = Xtr[c].astype("category")
    m = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=4,
        verbose=-1,
    )
    m.fit(Xtr, np.log1p(ytr))
    # predict
    Xf = {k: FC[k] for k in FC}
    Xf["h"] = np.full(ns, h, dtype=np.float32)
    twf = nw - 1 + h
    wf = weeks[-1] + pd.Timedelta(weeks=h)
    Xf["wsin"] = np.full(ns, np.sin(2 * np.pi * wf.isocalendar().week / 52.0))
    Xf["wcos"] = np.full(ns, np.cos(2 * np.pi * wf.isocalendar().week / 52.0))
    Xf["yr"] = np.full(ns, wf.year - years.min(), dtype=np.float32)
    Xpf = pd.DataFrame(Xf)
    for c in CATCOLS:
        Xpf[c] = pd.Categorical(Xpf[c], categories=Xtr[c].cat.categories)
    pred = np.expm1(m.predict(Xpf))
    rows.append(
        pd.DataFrame(
            {"series_id": series, "horizon": h, "y_hat": np.clip(pred, 0, None)}
        )
    )

out = pd.concat(rows, ignore_index=True)
# ---- validation ----
assert len(out) == 599 * 8, len(out)
assert out.y_hat.notna().all() and np.isfinite(out.y_hat).all()
assert (out.y_hat >= 0).all()
assert out.duplicated(["series_id", "horizon"]).sum() == 0
out.to_csv(OUT, index=False)
print("GBM ok", out.shape, out.y_hat.describe().to_dict())
