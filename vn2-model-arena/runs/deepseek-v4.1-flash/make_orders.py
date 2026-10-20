import numpy as np, pandas as pd

L = np.load("L.npy")
ids = np.load("ids.npy", allow_pickle=True)
n, T = L.shape
mu = np.maximum(L[:, T - 52 : T].mean(axis=1), 0) * 0.9
q = np.maximum(0, np.rint(mu))
rows = [(sid, w + 1, float(q[i])) for i, sid in enumerate(ids) for w in range(6)]
out = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
assert len(out) == 599 * 6 and out.series_id.nunique() == 599
assert set(out.week) == {1, 2, 3, 4, 5, 6} and (out.quantity >= 0).all()
out.to_csv("orders.csv", index=False)
print(len(out), out.quantity.min(), out.quantity.mean(), out.quantity.max())
