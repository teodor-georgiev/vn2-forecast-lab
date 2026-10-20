"""Exploration: load data, stats on censoring / intermittency / seasonality."""

import numpy as np
import pandas as pd

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"

sales = pd.read_csv(f"{DATA}\\sales.csv")
avail = pd.read_csv(f"{DATA}\\availability.csv")

weeks = sorted(sales["week"].unique())
series = sorted(sales["series_id"].unique())
print("n_weeks", len(weeks), "n_series", len(series))

sidx = {s: i for i, s in enumerate(series)}
widx = {w: i for i, w in enumerate(weeks)}
T, N = len(weeks), len(series)
Y = np.zeros((N, T))
A = np.zeros((N, T))
Y[sales["series_id"].map(sidx).to_numpy(), sales["week"].map(widx).to_numpy()] = sales[
    "y"
].to_numpy()
A[avail["series_id"].map(sidx).to_numpy(), avail["week"].map(widx).to_numpy()] = avail[
    "is_available"
].to_numpy()

print("Y", Y.shape, "min", Y.min(), "max", Y.max())
print("avail0 frac", 1 - A.mean())
print("zero-y frac", (Y == 0).mean())
# zero sales when available (true zero demand) vs not available
m1 = A == 1
print("P(y==0 | avail=1)", (Y[m1] == 0).mean())
print("P(y==0 | avail=0)", (Y[~m1] == 0).mean())
print("mean y | avail=1", Y[m1].mean(), "mean y | avail=0", Y[~m1].mean())
print("mean y avail0 weeks (censored) vs avail1:", Y[~m1].mean(), Y[m1].mean())

# per series stats
tot = Y.sum(1)
print("series total demand: median", np.median(tot), "mean", tot.mean())
p0 = (Y == 0).mean(1)
print("P(y==0) per series: quartiles", np.percentile(p0, [10, 25, 50, 75, 90]).round(3))
frac_avail0 = (A == 0).mean(1)
print(
    "censored frac per series quartiles",
    np.percentile(frac_avail0, [10, 50, 90]).round(3),
)

# seasonality: correlation of weekly aggregate with lag 52
agg = Y.sum(0)
for lag in (1, 4, 13, 26, 52):
    c = np.corrcoef(agg[:-lag], agg[lag:])[0, 1]
    print(f"agg autocorr lag {lag}: {c:.3f}")
# weekday pattern: per-week-index mean across 3 years (agg by position mod 52? weeks are calendar)
# check year alignment: mean level per year
for k in range(3):
    seg = agg[k * 52 : (k + 1) * 52]
    if len(seg):
        print("year", k, "mean", round(seg.mean(), 1))

# how much of per-series variance is seasonal (year-ago correlation per series)
c52 = []
for i in range(N):
    a, b = Y[i, :-52], Y[i, 52:]
    if a.std() > 0 and b.std() > 0:
        c52.append(np.corrcoef(a, b)[0, 1])
print(
    "per-series corr lag52 quartiles",
    np.percentile(c52, [25, 50, 75]).round(3),
    "n",
    len(c52),
)

# last weeks
print("T =", T, "last week", weeks[-1])
np.save("Y.npy", Y)
np.save("A.npy", A)
print("saved")
