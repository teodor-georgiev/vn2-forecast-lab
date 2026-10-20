import pandas as pd
import numpy as np

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
OUT = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\nemotron-3.5-super-vl\predictions.csv"

sales = pd.read_csv(f"{DATA}/sales.csv", parse_dates=["week"])
avail = pd.read_csv(f"{DATA}/availability.csv", parse_dates=["week"])

# Merge availability; treat unavailable weeks as missing demand
df = sales.merge(avail, on=["series_id", "week"], how="left")
df["y"] = np.where(df["is_available"].fillna(True), df["y"], np.nan)

last_week = df["week"].max()

series_ids = sorted(df["series_id"].unique())
rows = []

for sid in series_ids:
    s = df[df["series_id"] == sid].sort_values("week")
    y = s["y"].values
    # last K available observations, weighted toward recent
    valid = y[~np.isnan(y)]
    if len(valid) == 0:
        fcast = 0.0
    else:
        k = min(8, len(valid))
        recent = valid[-k:]
        # exponential-like weights: more recent = higher weight
        w = np.arange(1, k + 1, dtype=float)
        fcast = float(np.dot(recent, w) / w.sum())
    fcast = max(0.0, fcast)
    for h in range(1, 9):
        rows.append((sid, h, fcast))

pred = pd.DataFrame(rows, columns=["series_id", "horizon", "y_hat"])
pred["y_hat"] = pred["y_hat"].clip(lower=0)
pred = pred[["series_id", "horizon", "y_hat"]]
pred.to_csv(OUT, index=False)
print("rows:", len(pred), "series:", pred["series_id"].nunique())
