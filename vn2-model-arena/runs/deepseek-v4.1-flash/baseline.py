import numpy as np
import pandas as pd

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
OUT = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\deepseek-v4.1-flash\predictions.csv"

s = pd.read_csv(D + r"\sales.csv", parse_dates=["week"])
a = pd.read_csv(D + r"\availability.csv", parse_dates=["week"])
p = pd.read_csv(D + r"\product_attributes.csv")

S = s.pivot(index="series_id", columns="week", values="y").sort_index()
A = (
    a.pivot(index="series_id", columns="week", values="is_available")
    .reindex_like(S)
    .fillna(True)
)
A = A.astype(bool)
Y = S.where(A)  # NaN when unavailable

series = list(S.index)
weeks = list(S.columns)
nw = len(weeks)

# global week-of-year seasonal factor from log ratio to trailing 52w level
lev = Y.rolling(52, min_periods=8, axis=1).mean()
lv = np.log((Y + 0.1) / (lev + 0.1))
woy = np.array([w.isocalendar().week for w in weeks])
fac = pd.Series(lv.values.ravel()).groupby(np.tile(woy, len(series))).mean()
fac = fac - fac.mean()
# smooth over 5-week circular window
idx = sorted(fac.index)
sm = {}
for i in idx:
    vals = [fac[((i - 2 + k - 1) % 52) + 1] for k in range(5)]
    sm[i] = np.mean(vals)
seas = pd.Series(sm)

# recent level per series, availability adjusted
L4 = Y.iloc[:, -4:].mean(axis=1)
L8 = Y.iloc[:, -8:].mean(axis=1)
L13 = Y.iloc[:, -13:].mean(axis=1)
L52 = Y.iloc[:, -52:].mean(axis=1)
lev_hat = 0.45 * L4.fillna(L8) + 0.35 * L8.fillna(L13) + 0.20 * L13.fillna(L52)

# shrink toward group mean (ProductGroup x Department) for sparse series
attr = p.set_index("series_id").reindex(series)
grp = attr["Department"].astype(str) + "|" + attr["StoreFormat"].astype(str)
gm = lev_hat.groupby(grp).transform("mean")
w = S.notna().sum(axis=1)
alpha = (w / (w + 12)).clip(0, 1)  # weight on own level
base = alpha * lev_hat + (1 - alpha) * gm
base = base.fillna(lev_hat).fillna(gm).fillna(0.5)

# expected availability for future weeks (no future info): series recent rate + global
ar4 = A.iloc[:, -4:].mean(axis=1)
ar13 = A.iloc[:, -13:].mean(axis=1)
ar52 = A.iloc[:, -52:].mean(axis=1)
ar = 0.5 * ar4 + 0.3 * ar13 + 0.2 * ar52
gar = A.mean(axis=1)
ar = (0.7 * ar + 0.3 * gar).clip(0.05, 1.0)

rows = []
for h in range(1, 9):
    tw = weeks[-1] + pd.Timedelta(weeks=h)
    k = tw.isocalendar().week
    f = float(np.exp(seas.get(k, 0.0)))
    yh = base.values * ar.values * f
    for sid, v in zip(series, yh):
        rows.append((sid, h, max(0.0, float(v))))

out = pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"])
out.to_csv(OUT, index=False)
print(out.shape, out.y_hat.describe().to_dict())
print(out.head())
