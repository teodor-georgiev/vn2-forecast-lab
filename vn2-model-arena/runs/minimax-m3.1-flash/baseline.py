import numpy as np, pandas as pd, os

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
sales = pd.read_csv(D + r"\sales.csv")
av = pd.read_csv(D + r"\availability.csv")
sales["week"] = pd.to_datetime(sales["week"])
av["week"] = pd.to_datetime(av["week"])
sids = sorted(sales["series_id"].unique())
weeks = sorted(sales["week"].unique())
S, T = len(sids), len(weeks)
si = {s: i for i, s in enumerate(sids)}
wi = {w: i for i, w in enumerate(weeks)}
Y = np.zeros((S, T))
A = np.zeros((S, T), bool)
sales["ti"] = sales["week"].map(wi)
sales["si"] = sales["series_id"].map(si)
Y[sales["si"].values, sales["ti"].values] = sales["y"].values
av["ti"] = av["week"].map(wi)
av["si"] = av["series_id"].map(si)
A[av["si"].values, av["ti"].values] = av["is_available"].values.astype(bool)
Yobs = np.where(A, Y, 0.0)
Yimp = Yobs.copy()
for i in range(S):
    m = A[i]
    if m.all():
        continue
    s = (
        pd.Series(np.where(m, Y[i], np.nan))
        .rolling(5, center=True, min_periods=1)
        .mean()
        .ffill()
        .bfill()
        .values
    )
    fb = np.nanmean(Y[i][m]) if m.any() else 0.0
    s = np.where(np.isnan(s), fb, s)
    Yimp[i, ~m] = s[~m]
mult = A[:, T - 8 : T].mean(axis=1)
rows = []
for h in range(1, 9):
    t = T - 1 + h
    base = Yimp[:, t - 52]
    p = np.maximum(base * mult, 0.0)
    for i, s in enumerate(sids):
        rows.append((s, h, float(p[i])))
out = pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"])
out.to_csv(RUN + r"\predictions.csv", index=False)
print("baseline written", out.shape, out["y_hat"].describe().to_dict())
