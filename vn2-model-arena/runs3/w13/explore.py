import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
sales = pd.read_csv(D / "sales.csv", parse_dates=["week"])
avail = pd.read_csv(D / "availability.csv", parse_dates=["week"])
init = pd.read_csv(D / "initial_state.csv")

Y = sales.pivot(index="series_id", columns="week", values="y")
A = avail.pivot(index="series_id", columns="week", values="is_available")
# align columns
assert list(Y.index) == list(A.index)
ids = list(Y.index)
y = Y.values.astype(float)  # (n,157) observed sales (0 when unavailable)
a = A.values.astype(float)
weeks = list(Y.columns)
print("shape", y.shape, a.shape)
# week of year for each column
woy = np.array([d.isocalendar().week for d in weeks], dtype=int)
year = np.array([d.year for d in weeks], dtype=int)
print("woy first 10", woy[:10])
print("years", np.unique(year))
# initial state aligned
init = init.set_index("series_id").loc[ids]
st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
print("st0 shape", st0.shape)
np.save("ids.npy", np.array(ids), allow_pickle=True)
# save arrays for quick use
np.savez("data_arrays.npz", y=y, a=a, woy=woy, year=year, st0=st0)
print("saved")
