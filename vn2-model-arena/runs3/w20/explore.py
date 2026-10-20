"""Authoring-only exploration (not part of the deliverable)."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")

sales = pd.read_csv(DATA / "sales.csv")
avail = pd.read_csv(DATA / "availability.csv")
init = pd.read_csv(DATA / "initial_state.csv")

order = init["series_id"].tolist()
piv = sales.pivot_table(index="series_id", columns="week", values="y")
piv = piv.reindex(index=order)
Y = piv.to_numpy(dtype=float)
weeks = list(piv.columns)

piva = avail.pivot_table(index="series_id", columns="week", values="is_available")
piva = piva.reindex(index=order, columns=weeks)
A = piva.to_numpy(dtype=bool)

print("Y", Y.shape, "nan:", np.isnan(Y).sum(), "weeks:", weeks[0], "->", weeks[-1])
print("censored weeks (avail False):", (~A).mean())
print("series with any censored:", (~A).any(axis=1).mean())
mean_all = np.nanmean(Y, axis=1)
mean_avail = np.nanmean(np.where(A, Y, np.nan), axis=1)
frac_zero = (Y == 0).mean(axis=1)
print(
    "mean demand: median %.3f  p90 %.3f  max %.3f"
    % (np.median(mean_all), np.quantile(mean_all, 0.9), mean_all.max())
)
print("share zero-sales weeks: median %.2f" % np.median(frac_zero))
m = (~A).mean(axis=1)
print(
    "censored share by series: median %.2f p90 %.2f"
    % (np.median(m), np.quantile(m, 0.9))
)
sel = m > 0.05
print(
    "among censored series, mean_avail/mean_all ratio median %.3f"
    % np.median(mean_avail[sel] / mean_all[sel])
)

# availability clustering: is censoring random or clustered?
runs = []
for i in range(Y.shape[0]):
    a = A[i]
    runs.append((~a).mean())
print("censored share quantiles:", np.quantile(runs, [0, 0.25, 0.5, 0.75, 0.9, 1]))


# seasonality: does same-week-last-year beat trailing 13-week mean?
# backtest forecast errors over historical weeks w >= 60 with 52+ w available
def errs():
    rows = []
    for i in range(Y.shape[0]):
        y = Y[i]
        for w in range(60, Y.shape[1] - 1):
            hist = y[max(0, w - 13) : w]
            if np.isnan(hist).any():
                continue
            f_tr = np.nanmean(hist)
            f_ly = y[w - 52] if w - 52 >= 0 and not np.isnan(y[w - 52]) else np.nan
            tgt = y[w + 1]
            if np.isnan(tgt):
                continue
            rows.append((i, w, f_tr, f_ly, tgt))
    return pd.DataFrame(rows, columns=["i", "w", "f_tr", "f_ly", "tgt"])


E = errs()
print("n forecast points:", len(E))
for name in ["f_tr", "f_ly"]:
    d = (E[name] - E["tgt"]).abs()
    print(name, "MAE %.4f  (nan %.3f)" % (np.nanmean(d), np.isnan(d).mean()))
# combined: shrink last year toward trailing mean
for a in [0.25, 0.5, 0.75]:
    f = a * E["f_ly"] + (1 - a) * E["f_tr"]
    print("blend %.2f MAE %.4f" % (a, np.nanmean((f - E["tgt"]).abs())))

# same-week-last-year value by demand scale
E["lvl"] = E.groupby("i")["tgt"].transform("mean")
E["rel"] = (E["f_ly"] - E["tgt"]).abs() / (E["lvl"] + 1e-9)
E["rel_tr"] = (E["f_tr"] - E["tgt"]).abs() / (E["lvl"] + 1e-9)
g = pd.cut(E["lvl"], [0, 0.5, 2, 10, 1e9])
print(E.groupby(g, observed=True)[["rel", "rel_tr"]].mean())
