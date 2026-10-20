import numpy as np, pandas as pd, lightgbm as lgb

exec(
    open(
        r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\deepseek-v4.1-flash\gbm.py"
    )
    .read()
    .split("origins = list")[0]
)

woy = np.array([w.isocalendar().week for w in weeks])
years = np.array([w.year for w in weeks])

HOLD = 8  # origin = nw-1-HOLD, targets nw-HOLD .. nw-1
o = nw - 1 - HOLD
FC = feats(o)
preds = {}
for h in range(1, 9):
    idxs = [i for i in range(60, nw - 1 - h) if i + h <= nw - 1]
    X = {k: np.concatenate([feats(i)[k] for i in idxs]) for k in FC}
    tw = np.array([i + h for i in idxs])
    X["h"] = np.repeat(h, len(idxs) * ns).astype(np.float32)
    X["wsin"] = np.repeat(np.sin(2 * np.pi * woy[tw] / 52.0), ns)
    X["wcos"] = np.repeat(np.cos(2 * np.pi * woy[tw] / 52.0), ns)
    X["yr"] = np.repeat(years[tw] - years.min(), ns).astype(np.float32)
    ytr = np.nan_to_num(np.concatenate([Y.values[:, t] for t in tw]))
    Xtr = pd.DataFrame(X)
    for c in CATCOLS:
        Xtr[c] = Xtr[c].astype("category")
    m = lgb.LGBMRegressor(
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
    Xf = dict(FC)
    Xf["h"] = np.full(ns, h, np.float32)
    wf = weeks[o] + pd.Timedelta(weeks=h)
    Xf["wsin"] = np.full(ns, np.sin(2 * np.pi * wf.isocalendar().week / 52.0))
    Xf["wcos"] = np.full(ns, np.cos(2 * np.pi * wf.isocalendar().week / 52.0))
    Xf["yr"] = np.full(ns, wf.year - years.min(), np.float32)
    Xpf = pd.DataFrame(Xf)
    for c in CATCOLS:
        Xpf[c] = pd.Categorical(Xpf[c], categories=Xtr[c].cat.categories)
    preds[h] = np.clip(np.expm1(m.predict(Xpf)), 0, None)

act = np.nan_to_num(Y.values[:, o + 1 : o + 9])  # (ns, 8)
nz = np.nan_to_num(A.values[:, o + 1 : o + 9]).astype(bool)
base = pd.read_csv(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\deepseek-v4.1-flash\pred_baseline.csv"
)
bll = (
    base.pivot(index="series_id", columns="horizon", values="y_hat")
    .reindex(series)
    .values
)
g = np.stack([preds[h] for h in range(1, 9)], axis=1)


def sc(P):
    return dict(
        WAPE=np.abs(P - act).sum() / act.sum(),
        MAE=np.abs(P - act).mean(),
        WAPE_z=np.abs(P - act)[nz].sum() / act[nz].sum(),
        bias=(P - act).mean(),
    )


for nm, P in (
    ("gbm", g),
    ("baseline", bll),
    ("avg", 0.5 * g + 0.5 * bll),
    ("0.7g", 0.7 * g + 0.3 * bll),
):
    print(nm, {k: round(v, 4) for k, v in sc(P).items()})
np.save(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\deepseek-v4.1-flash\val_gbm.npy",
    g,
)
