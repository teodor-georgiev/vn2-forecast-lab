import numpy as np, pandas as pd, sys
from pathlib import Path

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
sales = pd.read_csv(ARENA / "data" / "sales.csv")
print(sales.dtypes, sales.shape)
print(sales.head())
sales["week"] = pd.to_datetime(sales["week"])
piv = sales.pivot(index="series_id", columns="week", values="y").sort_index()
print("pivot", piv.shape)
print("date range", piv.columns.min(), piv.columns.max())
H = piv.to_numpy(float)
print("total demand sum", H.sum(), "mean per series per week", H.sum(1).mean())
print("describe all", pd.Series(H.ravel()).describe().to_string())
# weekly totals
wt = H.sum(0)
for i in range(0, 157, 13):
    print(
        piv.columns[i].date(),
        "week_tot=%.0f" % wt[i],
        "mean=%.3f" % H[:, i].mean(),
        "nonzero_frac=%.3f" % (H[:, i] > 0).mean(),
    )
print("last 20 weeks totals")
for i in range(137, 157):
    print(
        piv.columns[i].date(),
        "tot=%.0f mean=%.3f nz=%.2f" % (wt[i], H[:, i].mean(), (H[:, i] > 0).mean()),
    )
# per series demand level distribution
ps = H.sum(1)
print("per-series 157w demand: ", pd.Series(ps).describe().to_string())
init = pd.read_csv(ARENA / "data" / "initial_state.csv")
print(init.describe().to_string())
print(
    "on_hand sums",
    init.on_hand.sum(),
    "arr1",
    init.arrival_w1.sum(),
    "arr2",
    init.arrival_w2.sum(),
)
# forecast weeks
future = pd.date_range(piv.columns.max() + pd.Timedelta(weeks=1), periods=8)
print("future weeks", [str(d.date()) for d in future])
