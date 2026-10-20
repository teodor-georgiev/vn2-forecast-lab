import numpy as np, pandas as pd, time, os

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
z = np.load(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w08\cache.npz",
    allow_pickle=True,
)
Y, A, ids = z["Y"], z["A"], z["ids"]
s = pd.read_csv(os.path.join(D, "sales.csv"))
dates = pd.to_datetime(s["week"].unique(), format="%Y-%m-%d")
woy = dates.isocalendar().week.to_numpy()
n, T = Y.shape

# ---- seasonality strength on available weeks ----
# per series: ratio of weekly sales to series mean, grouped by woy
tot = (Y * A).sum(axis=1)
cnt = A.sum(axis=1)
mean_av = tot / np.maximum(cnt, 1)
woy_mat = np.tile(woy, (n, 1))  # (n, T)
ratio = np.where(A, Y / np.maximum(mean_av[:, None], 1e-9), np.nan)
num = np.zeros((n, 53))
den = np.zeros((n, 53))
for k in range(1, 53):
    m = (woy_mat == k) & A
    num[:, k] = np.where(m, Y, 0).sum(axis=1)
    den[:, k] = m.sum(axis=1)
seas_mean = num / np.maximum(den, 1)
seas_mean[den == 0] = 1.0
# seasonal amplitude: weighted std of seasonal means around 1
amp = np.sqrt(
    np.maximum(
        (den * (seas_mean - 1) ** 2).sum(axis=1) / np.maximum(den.sum(axis=1), 1), 0
    )
)
print("seasonal log-amp pct:", np.percentile(amp, [10, 25, 50, 75, 90]))
# how much does the 8-week-ahead-from-history-end season deviate from flat for these series
seas_next = seas_mean[:, [1, 2, 3, 4, 5, 6, 7, 8]] if 8 <= 52 else None
print("mean |seas-1| for woy of next 8 wks:", np.abs(seas_next - 1).mean(axis=1).mean())

# ---- autocorrelation of weekly demand ----
lag1 = []
for i in range(0, n, 1):
    x = Y[i]
    if x.std() > 0:
        lag1.append(np.corrcoef(x[:-1], x[1:])[0, 1])
lag1 = np.array(lag1)
print("lag1 autocorr pct:", np.percentile(lag1, [10, 25, 50, 75, 90]))
lag4 = []
for i in range(n):
    x = Y[i]
    if x.std() > 0:
        lag4.append(np.corrcoef(x[:-4], x[4:])[0, 1])
print("lag4 autocorr pct:", np.percentile(np.array(lag4), [10, 25, 50, 75, 90]))
# lag 52 (year over year)
yy = []
for i in range(n):
    x = Y[i]
    if x.std() > 0 and T > 60:
        yy.append(np.corrcoef(x[:-52], x[52:])[0, 1])
print("lag52 autocorr pct:", np.percentile(np.array(yy), [10, 25, 50, 75, 90]))
