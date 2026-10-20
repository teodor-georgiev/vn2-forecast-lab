import numpy as np, pandas as pd, os, time

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
t0 = time.time()
s = pd.read_csv(os.path.join(D, "sales.csv"))
a = pd.read_csv(os.path.join(D, "availability.csv"))
print("load", time.time() - t0)
s["w"] = pd.factorize(s["week"])[0]
a["w"] = pd.factorize(a["week"])[0]
s = s.sort_values(["w", "series_id"])
a = a.sort_values(["w", "series_id"])
ids = s["series_id"].unique()
T = s["w"].nunique()
n = len(ids)
Y = s["y"].to_numpy().reshape(T, n).T.astype(float)  # (n, T)
A = a["is_available"].to_numpy().reshape(T, n).T.astype(bool)
print("Y", Y.shape, "A", A.shape, "dates", s["week"].iloc[0], s["week"].iloc[-1])
print("T per series check:", np.all(s.groupby("series_id").size().values == T))

np.savez_compressed(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w08\cache.npz",
    Y=Y,
    A=A,
    ids=ids.astype(str),
    weeks=s["week"].unique(),
)
print("total units", Y.sum(), "mean avail", A.mean())
avail_frac = A.mean(axis=1)
print(
    "avail frac per series: pct", np.percentile(avail_frac, [0, 5, 25, 50, 75, 95, 100])
)
tot = Y.sum(axis=1)
print("total per series: pct", np.percentile(tot, [0, 5, 25, 50, 75, 95, 100]))
print(
    "zero weeks frac pct",
    np.percentile((Y == 0).mean(axis=1), [0, 25, 50, 75, 95, 100]),
)
# zero-week fraction among available weeks
zw = ((Y == 0) & A).sum(axis=1) / np.maximum(A.sum(axis=1), 1)
print("zero frac (avail weeks) pct", np.percentile(zw, [0, 25, 50, 75, 95, 100]))
# series with low availability
print(
    "series with avail<0.9:",
    (avail_frac < 0.9).sum(),
    " <0.7:",
    (avail_frac < 0.7).sum(),
    " <0.5:",
    (avail_frac < 0.5).sum(),
)
# seasonality: per-series week-of-year index strength
dates = pd.to_datetime(s["week"].unique(), format="%Y-%m-%d")
print("first dates", dates[:3].tolist(), "last", dates[-3:].tolist())
woy = dates.isocalendar().week.to_numpy()
print("woy of last 16 weeks:", woy[-16:])
