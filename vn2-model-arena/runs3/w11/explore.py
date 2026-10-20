import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
init = pd.read_csv(D / "initial_state.csv")

# pivot
Y = sales.pivot(index="series_id", columns="week", values="y").fillna(0.0)
A = avail.pivot(index="series_id", columns="week", values="is_available").fillna(1.0)
print("sales", Y.shape, "avail", A.shape)
print(
    "init order matches sales order:",
    list(init.series_id) == list(Y.index[: len(init)]),
)
print("init order matches sorted sales order:", list(init.series_id) == list(Y.index))
print("n init", len(init), "n sales", len(Y))

y = Y.values.astype(float)
a = A.values.astype(float)
print("weeks:", Y.columns.min(), Y.columns.max())
print("total sales mean per week first 5:", y[:, :5].mean(axis=0).round(1))
print("frac avail", a.mean().round(3))
print(
    "series mean sales quantiles:",
    np.percentile(y.mean(1), [10, 25, 50, 75, 90, 99]).round(3),
)
print("frac zero weeks:", (y == 0).mean())
print("CV per series median:", np.median(y.std(1) / np.maximum(y.mean(1), 1e-9)))
print(
    "availability==0 share by series quantiles:",
    np.percentile(1 - a.mean(1), [50, 75, 90, 99]).round(3),
)
print("init state sums:", init[["on_hand", "arrival_w1", "arrival_w2"]].sum().to_dict())
print("last 8 weeks total sales:", y[:, -8:].sum(axis=0).round(1))
print("first 8 weeks total sales:", y[:, :8].sum(axis=0).round(1))
# yearly seasonality: aggregate sales by week index mod 52
tot = y.sum(axis=0)
for lag in (52, 53):
    print("autocorr lag", lag, np.corrcoef(tot[:-lag], tot[lag:])[0, 1].round(3))
print("global weekly totals by year blocks:")
print(tot.reshape(-1)[:156].reshape(3, 52).mean(axis=1).round(1))
print("week-of-year index of max/min (aggregate 3 yrs):")
block = tot[:156].reshape(3, 52).mean(0)
print("max idx", block.argmax(), "min idx", block.argmin())
print(np.round(block / block.mean(), 2))
