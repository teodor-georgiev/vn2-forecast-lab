import pandas as pd, numpy as np

base = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
out = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\longcat-2.5"
sales = pd.read_csv(base + r"\sales.csv", parse_dates=["week"])
av = pd.read_csv(base + r"\availability.csv", parse_dates=["week"])
init = pd.read_csv(base + r"\initial_state.csv")

m = sales.merge(av, on=["series_id", "week"], how="left")
m["t"] = (m["week"] - m["week"].min()).dt.days // 7
T = int(m.t.max()) + 1
series = sorted(m.series_id.unique())
n = len(series)
idx = {s: i for i, s in enumerate(series)}

Y = np.zeros((T, n))
A = np.ones((T, n), dtype=bool)
for r in m.itertuples():
    Y[int(r.t), idx[r.series_id]] = r.y
    A[int(r.t), idx[r.series_id]] = r.is_available

L = []
L.append(f"last date: {m.week.max().date()}, T={T}")
L.append(
    f"game weeks: {[(m.week.max() + pd.Timedelta(weeks=k)).date() for k in range(1, 9)]}"
)

# availability vs demand
unav = m[~m.is_available]
L.append(
    f"unavailable frac {len(unav) / len(m):.3f}, y>0 when unavailable: {(unav.y > 0).mean():.3f}"
)
av_y = m[m.is_available]
L.append(f"y mean available {av_y.y.mean():.4f} vs unavailable {unav.y.mean():.4f}")

# game week dates -> week of year
gw = [m.week.max() + pd.Timedelta(weeks=k) for k in range(1, 9)]
woy = [w.isocalendar().week for w in gw]
L.append(f"game week woys: {woy}")

# seasonal factor per woy using last 2 full years (t 52..155) vs overall
m["woy"] = m.week.dt.isocalendar().week.astype(int)
seas = m[m.t >= T - 104].groupby("woy")["y"].mean()
overall = m[m.t >= T - 104]["y"].mean()
seas = seas / overall
L.append(
    "seasonal factors (recent 2y): " + str({k: round(v, 3) for k, v in seas.items()})
)

# residual autocorrelation per series (last 40 weeks, demeaned by rolling 8wk mean)
resid = np.zeros((T, n))
for t in range(T):
    pass
level8 = pd.DataFrame(Y).rolling(8, min_periods=4).mean().values
R = Y[:T] / np.maximum(level8, 1e-6)
acf1, acf2 = [], []
for i in range(n):
    r = R[T - 40 : T, i]
    r = r[~np.isnan(r)]
    if len(r) > 10 and np.std(r) > 1e-6:
        acf1.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        acf2.append(np.corrcoef(r[:-2], r[2:])[0, 1])
L.append(f"residual acf1 mean {np.nanmean(acf1):.3f}, acf2 mean {np.nanmean(acf2):.3f}")

# distribution of residuals
r_all = R[T - 40 : T, :].ravel()
r_all = r_all[~np.isnan(r_all)]
L.append(
    f"residual quantiles: {np.percentile(r_all, [5, 25, 50, 75, 95, 99]).round(3)}"
)

# initial state totals
L.append(
    "init sums: " + str(init[["on_hand", "arrival_w1", "arrival_w2"]].sum().to_dict())
)
ih = init.set_index("series_id").loc[series]
L.append(f"init on_hand per series: mean {ih.on_hand.mean():.2f}")

np.save(out + r"\Y.npy", Y)
np.save(out + r"\A.npy", A)
np.save(out + r"\series.npy", np.array(series))
pd.Series(woy).to_csv(out + r"\game_woy.csv", index=False)
seas.to_csv(out + r"\seas.csv")
print("\n".join(L))
