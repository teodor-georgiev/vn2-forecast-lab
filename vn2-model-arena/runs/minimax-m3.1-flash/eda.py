import pandas as pd, numpy as np

d = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
s = pd.read_csv(d + r"\sales.csv")
a = pd.read_csv(d + r"\availability.csv")
s["week"] = pd.to_datetime(s["week"])
a["week"] = pd.to_datetime(a["week"])
m = s.merge(a, on=["series_id", "week"], how="left")
m["t"] = ((m["week"] - m["week"].min()).dt.days // 7).astype(int)
print("t range", m["t"].min(), m["t"].max())

# is y = log1p(count)?
z = np.expm1(m["y"].values)
r = np.abs(z - np.round(z))
print("expm1(y) integer frac:", (r < 1e-6).mean(), "median resid", np.median(r))
# maybe y = log(count+1) with count large -> test log((exp(y)-1)+1)
# check candidates
for name, f in [("log1p", lambda v: np.log1p(v)), ("log", lambda v: np.log(v))]:
    pass

# availability by time
av = m.groupby("t")["is_available"].mean()
print("avail by t head/tail:")
print(av.head(5))
print(av.tail(20))
# per-series: availability rate by half
h = (
    m.assign(half=(m["t"] >= 78).astype(int))
    .groupby(["series_id", "half"])["is_available"]
    .mean()
    .unstack()
)
print("avail rate early vs late corr:", h[0].corr(h[1]))
print("frac series with lower late:", (h[1] < h[0]).mean())

# seasonality: global mean by week-of-year
m["woy"] = m["week"].dt.isocalendar().week.astype(int)
print(m[m["is_available"]].groupby("woy")["y"].mean().round(2).to_dict())

# series level
g = m.groupby("series_id")["y"].agg(["mean", "std", "max", "sum"])
print(g.describe())

# autocorrelation of y within series (available only rows? use filled 0 series)
pv = m.pivot(index="series_id", columns="t", values="y").fillna(0.0)
print("shape", pv.shape)
for lag in [1, 2, 4, 8, 13, 26, 52]:
    c = [
        np.corrcoef(pv.iloc[:, :-lag].values.ravel(), pv.iloc[:, lag:].values.ravel())[
            0, 1
        ]
    ]
    print("lag", lag, "corr", np.round(c, 3))
# last-value vs seasonal naive scale
last = pv.iloc[:, -1].values
s52 = pv.iloc[:, -53].values
print("mean last", last.mean(), "mean s52", s52.mean())
