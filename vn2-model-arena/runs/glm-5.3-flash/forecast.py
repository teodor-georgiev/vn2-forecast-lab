import numpy as np
import pandas as pd

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
OUT = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\glm-5.3-flash\predictions.csv"

sales = pd.read_csv(DATA + r"\sales.csv", parse_dates=["week"])
avail = pd.read_csv(DATA + r"\availability.csv", parse_dates=["week"])

df = sales.merge(avail, on=["series_id", "week"], how="left")
df["is_available"] = df["is_available"].fillna(True).astype(bool)
df = df.sort_values(["series_id", "week"])

last_week = df["week"].max()

rows = []
for sid, g in df.groupby("series_id", sort=True):
    y = g["y"].to_numpy(float)
    a = g["is_available"].to_numpy(bool)
    n = len(y)
    if n == 0:
        continue

    def mean(mask_len, avail_only=False, all_ok=False):
        # mean over the last mask_len weeks of this series
        k = min(mask_len, n)
        yy, aa = y[-k:], a[-k:]
        if avail_only:
            m = aa
            if m.any():
                return yy[m].mean()
            if all_ok:
                return yy.mean()
            return 0.0
        return yy.mean()

    # demand rate on available weeks at multiple windows
    r4 = mean(4, avail_only=True, all_ok=True)
    r13 = mean(13, avail_only=True, all_ok=True)
    r52 = mean(52, avail_only=True, all_ok=True)
    r_all = mean(n, avail_only=True, all_ok=True)

    rate = 0.45 * r4 + 0.30 * r13 + 0.15 * r52 + 0.10 * r_all

    # probability of availability, recent-weighted
    k8 = min(8, n)
    p_recent = a[-k8:].mean() if k8 > 0 else 1.0
    p_all = a.mean()
    p_avail = 0.6 * p_recent + 0.4 * p_all

    # recent overall mean (captures stockout-driven dips already in y)
    recent8 = mean(8)

    base = 0.5 * rate * p_avail + 0.5 * recent8

    # short series safety
    if n < 10:
        base = y.mean()

    # hard stockout: unavailable for the last 4+ weeks -> near zero
    if n >= 4 and (~a[-4:]).all():
        base *= 0.15
    elif n >= 2 and (~a[-2:]).all():
        base *= 0.5

    base = max(0.0, float(base))

    # small upward bias correction for intermittency (mean-ish, RMSE friendly)
    for h in range(1, 9):
        yh = base * (0.995 ** (h - 1))
        rows.append((sid, h, yh))

out = pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"])
out["y_hat"] = out["y_hat"].clip(lower=0)
assert len(out) == 4792, len(out)
assert out["y_hat"].notna().all() and np.isfinite(out["y_hat"]).all()
assert out.duplicated(["series_id", "horizon"]).sum() == 0
assert set(out["horizon"].unique()) == set(range(1, 9))
assert out["series_id"].nunique() == 599
out.to_csv(OUT, index=False)
print("wrote", OUT, len(out), "rows;", out["y_hat"].mean().round(3), "mean forecast")
