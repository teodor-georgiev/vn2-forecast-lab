"""Produce final orders.csv: Croston forecast + greedy base-stock policy (k=0),
using all 157 weeks of history and the real initial_state.csv.
"""

import sys
import numpy as np
import pandas as pd

ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
sys.path.insert(0, ARENA + r"\tools")
from inventory_sim import simulate  # noqa: E402

DATA = ARENA + r"\data"
RUN = ARENA + r"\runs\nemotron-3.5-super-vl"

SALES = pd.read_csv(f"{DATA}/sales.csv")
INIT = pd.read_csv(f"{DATA}/initial_state.csv")

series_order = INIT["series_id"].tolist()
n = len(series_order)

piv = SALES.pivot(index="series_id", columns="week", values="y").reindex(
    index=series_order
)
H = np.nan_to_num(piv.values.astype(float), nan=0.0)  # (n, 157)

init = INIT.set_index("series_id").reindex(series_order)
STATE0 = np.column_stack(
    [
        init["on_hand"].values.astype(float),
        init["arrival_w1"].values.astype(float),
        init["arrival_w2"].values.astype(float),
    ]
)

ORDER_WEEKS = 6
WEEKS = 8


def croston_forecast(train):
    n, _ = train.shape
    mu = np.zeros((n, 8))
    sigma = np.zeros(n)
    for i in range(n):
        y = train[i]
        nz_idx = np.flatnonzero(y > 0)
        if len(nz_idx) == 0:
            continue
        sizes = y[nz_idx]
        inter = np.diff(np.concatenate([[0], nz_idx + 1]))
        inter = inter[1:]
        if len(inter) == 0:
            inter = np.array([1.0])
        fc = sizes.mean() / max(inter.mean(), 1.0)
        mu[i, :] = fc
        sigma[i] = sizes.std() if len(sizes) > 1 else max(0.5, fc)
    return mu, sigma


def greedy_orders(mu, sigma, state, k=0.0):
    d = np.maximum(0.0, mu + k * sigma[:, None])
    on_hand = np.asarray(state)[:, 0].copy()
    arr1 = np.asarray(state)[:, 1].copy()
    arr2 = np.asarray(state)[:, 2].copy()
    orders = np.zeros((ORDER_WEEKS, n))
    oh, an1, an2 = on_hand, arr1, arr2
    end = np.zeros(n)
    for t in range(WEEKS):
        supply = oh + an1
        end = np.maximum(0.0, supply - d[:, t])
        if t < ORDER_WEEKS:
            if t + 1 < WEEKS:
                end_t1 = np.maximum(0.0, end + an2 - d[:, t + 1])
            else:
                end_t1 = end.copy()
            if t + 2 < WEEKS:
                orders[t] = np.maximum(0.0, d[:, t + 2] - end_t1)
            else:
                orders[t] = 0.0
            oh, an1, an2 = end, an2, orders[t]
        else:
            oh, an1, an2 = end, an2, np.zeros(n)
    return orders


mu, sigma = croston_forecast(H)
orders = greedy_orders(mu, sigma, STATE0, k=0.0)
orders = np.rint(np.clip(orders, 0.0, None)).astype(int)

# sanity: replay against final 8 weeks of history as a stand-in scored horizon
demo = simulate(H[:, -8:].T, STATE0, orders)
print(
    "demo score (last 8 wks as horizon):",
    round(demo["score"], 2),
    "setup:",
    round(demo["setup"], 2),
)

rows = []
for i, sid in enumerate(series_order):
    for w in range(1, ORDER_WEEKS + 1):
        rows.append((sid, w, int(orders[w - 1, i])))
out = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
out.to_csv(f"{RUN}/orders.csv", index=False)
print("wrote orders.csv:", len(out), "rows")
print(out.groupby("week").quantity.sum().to_dict())
print("min qty:", out.quantity.min(), "max qty:", out.quantity.max())
