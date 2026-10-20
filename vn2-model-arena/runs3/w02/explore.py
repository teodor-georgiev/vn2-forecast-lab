"""Quick data exploration for the VN2 arena (w02)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data")


def load():
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["week"])
    avail = pd.read_csv(DATA / "availability.csv", parse_dates=["week"])
    sales = sales.pivot(index="series_id", columns="week", values="y")
    avail = avail.pivot(index="series_id", columns="week", values="is_available")
    avail = avail.astype(float)
    sales = sales.sort_index(axis=1)
    avail = avail.reindex(index=sales.index, columns=sales.columns)
    return sales, avail


if __name__ == "__main__":
    sales, avail = load()
    print("sales shape", sales.shape, "avail shape", avail.shape)
    print(
        "weeks",
        sales.columns.min(),
        sales.columns.max(),
        sales.columns.is_monotonic_increasing,
    )
    print("avail dtypes", avail.dtypes.unique())
    print(
        "nan in sales:",
        sales.isna().sum().sum(),
        "nan in avail:",
        avail.isna().sum().sum(),
    )
    print("\ndescribe sales:")
    print(sales.describe().T)
    print("\navail rate overall:", avail.values.mean())
    print("per-week avail rate (first 10):", avail.mean().values[:10])
    print("per-week avail rate (last 10):", avail.mean().values[-10:])
    print("\ntotal sales per week (last 20):")
    print(sales.sum().values[-20:])
    print("\navail==0 share of weeks per series: mean", (avail == 0).mean().mean())
    print("series with any avail 0:", (avail == 0).any(axis=1).mean())
    # correlation between sales and availability
    z = avail == 0
    print("\nmean sales where avail=1:", sales.values[z.values == 0].mean())
    print("mean sales where avail=0:", sales.values[z.values == 1].mean())
    # zero sales
    print("\nfraction zero sales:", (sales.values == 0).mean())
    # initial state
    init = pd.read_csv(DATA / "initial_state.csv")
    print(
        "\ninitial state:",
        init.shape,
        init[["on_hand", "arrival_w1", "arrival_w2"]].describe().T,
    )
    # does sales have series ids not in initial?
    print(
        "series in initial not in sales:", len(set(init.series_id) - set(sales.index))
    )
    print(
        "series in sales not in initial:", len(set(sales.index) - set(init.series_id))
    )
    # weekly seasonality: total sales by week-of-year across 3 years
    tot = sales.sum()
    t = pd.DataFrame({"w": tot.index, "v": tot.values})
    t["doy"] = t.w.dt.dayofyear
    t["yr"] = t.w.dt.year
    print("\nyearly totals:", t.groupby("yr").v.sum().to_dict())
