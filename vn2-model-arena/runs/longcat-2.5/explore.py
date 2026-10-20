import pandas as pd, numpy as np

base = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
out = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\longcat-2.5"
sales = pd.read_csv(base + r"\sales.csv", parse_dates=["week"])
av = pd.read_csv(base + r"\availability.csv", parse_dates=["week"])
m = sales.merge(av, on=["series_id", "week"], how="left")
m["t"] = (m["week"] - m["week"].min()).dt.days // 7

L = []
L.append(f"rows {len(m)}, series {m.series_id.nunique()}, weeks {m.t.max() + 1}")
L.append(f"avail frac {m.is_available.mean():.3f}")
L.append(f"y mean {m.y.mean():.4f} std {m.y.std():.4f}")
L.append(f"zero frac {(m.y == 0).mean():.4f}")
g = m.groupby("series_id")["y"].agg(["mean", "std"])
L.append("per-series mean describe: " + str(g["mean"].describe().to_dict()))
L.append(
    f"series mean<0.1: {(g['mean'] < 0.1).sum()}, mean<0.5: {(g['mean'] < 0.5).sum()}"
)
m["year"] = m.week.dt.year
L.append(f"years: {sorted(m.year.unique())}")
per_year = m.groupby("year")["y"].mean()
L.append("per-year means: " + str(per_year.to_dict()))
m["woy"] = m.week.dt.isocalendar().week.astype(int)
seas = m.groupby("woy")["y"].mean()
L.append(f"woy seasonality max/min: {seas.max() / max(seas.min(), 1e-9):.2f}")
recent = m[m.t > m.t.max() - 26].groupby("series_id")["y"].mean()
prior = (
    m[(m.t > m.t.max() - 52) & (m.t <= m.t.max() - 26)].groupby("series_id")["y"].mean()
)
L.append(
    f"recent26 mean {recent.mean():.4f} prior26 mean {prior.mean():.4f} ratio {recent.mean() / prior.mean():.3f}"
)
L.append("per-series std describe: " + str(g["std"].describe().to_dict()))

# correlation of series with recent level (persistence)
r = recent.reindex(prior.index)
L.append(
    f"corr(recent, prior) per series: {np.corrcoef(r.fillna(0), prior.fillna(0))[0, 1]:.3f}"
)

# demand in last 8 weeks stats
last8 = m[m.t > m.t.max() - 8]
L.append(
    "last-8wk mean: " + str(last8.groupby("series_id")["y"].mean().describe().to_dict())
)

open(out + r"\explore.txt", "w").write("\n".join(L))
print("done")
