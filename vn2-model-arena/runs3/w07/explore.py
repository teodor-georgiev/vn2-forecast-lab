import numpy as np, pandas as pd, time
from pathlib import Path

DATA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")
t0 = time.time()
sales = pd.read_csv(DATA / "sales.csv")
av = pd.read_csv(DATA / "availability.csv")
init = pd.read_csv(DATA / "initial_state.csv")
print("read", time.time() - t0)

print(sales.shape, av.shape, init.shape)
print(sales.head())
weeks = sorted(sales["week"].unique())
print("n weeks", len(weeks), weeks[0], weeks[-1])

sids = sorted(sales["series_id"].unique())
print("n series", len(sids))
S = sales.pivot(index="series_id", columns="week", values="y").reindex(sids)
A = av.pivot(index="series_id", columns="week", values="is_available").reindex(sids)
print("pivot", S.shape, A.shape, "avail missing", A.isna().sum().sum())

y = S.values
a = A.values.astype(float)
print("total y", np.nansum(y), "mean weekly", np.nanmean(y))
print("frac zero", (y == 0).mean(), "frac nan", np.isnan(y).mean())
print("avail frac", np.nanmean(a))
# per series stats
tot = np.nansum(y, axis=1)
print(
    "series weekly mean: ",
    np.percentile(np.nanmean(y, axis=1), [10, 25, 50, 75, 90, 99]),
)
print("init on_hand summary", init["on_hand"].describe())
