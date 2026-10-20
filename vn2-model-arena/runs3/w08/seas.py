import numpy as np, pandas as pd, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from work import Y, A
import pandas as pd

D = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
s = pd.read_csv(os.path.join(D, "sales.csv"))
dates = pd.to_datetime(s["week"].unique(), format="%Y-%m-%d")
woy = dates.isocalendar().week.to_numpy()

agg = Y.sum(axis=0)
print("total demand per week (last 40):")
for t in range(len(agg) - 40, len(agg)):
    print(f"  t={t:3d} date={dates[t].date()} woy={woy[t]:2d} agg={agg[t]:8.0f}")
print("\nagg stats: mean=%.0f std=%.0f" % (agg.mean(), agg.std()))
# week-of-year profile (mean of available-only sales)
mask = A
num = np.where(mask, Y, 0).sum(axis=0)
den = mask.sum(axis=0)
prof = {}
for k in range(1, 53):
    m = woy == k
    prof[k] = num[m].sum() / max(den[m].sum(), 1)
ks = sorted(prof)
mx = max(prof.values())
print("\nweek-of-year profile (available weeks only), normalised:")
line = " ".join(f"{k:02d}:{prof[k] / mx:.2f}" for k in ks)
print(line)
# what the profile says for the weeks of a game starting at 130 and at 157
for start in (100, 130, 157):
    ws = [(start + i) % 157 for i in range(8)]
    print(
        f"start={start}: woy={[woy[t] for t in ws]} prof={[round(prof[woy[t]] / mx, 2) for t in ws]}"
    )
# per-year comparison of the same weeks
print("\nyearly totals of weeks 0-52 vs 52-104 vs 104-156:")
for a in range(0, 157, 52):
    b = min(a + 52, 157)
    print(f"  weeks {a}-{b - 1}: {agg[a:b].sum():9.0f}")
