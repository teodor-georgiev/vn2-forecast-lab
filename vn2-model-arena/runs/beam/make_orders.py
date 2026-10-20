import pandas as pd, numpy as np, sys

BASE = "C:/Users/Toni/Documents/Default Project/vn2-model-arena"
DATA = BASE + "/data"
RUN = BASE + "/runs/beam"
sys.path.insert(0, BASE + "/tools")
from inventory_sim import reference_orders

sales = pd.read_csv(DATA + "/sales.csv")
attrs = pd.read_csv(DATA + "/product_attributes.csv")
state_df = pd.read_csv(DATA + "/initial_state.csv")
series_ids = sorted(sales.series_id.unique())
n = len(series_ids)

pred = pd.read_csv(RUN + "/predictions.csv")
# build F (8,n)
F = np.zeros((8, n))
for h in range(1, 9):
    sub = pred[pred.horizon == h].set_index("series_id").y_hat
    F[h - 1] = [max(0.0, float(sub.get(s, 0.0))) for s in series_ids]

# initial state aligned
st = state_df.set_index("series_id")
init = np.zeros((n, 3))
for i, s in enumerate(series_ids):
    r = st.loc[s]
    init[i, 0] = r["on_hand"]
    init[i, 1] = r["arrival_w1"]
    init[i, 2] = r["arrival_w2"]

scale = 0.9
projection = 1.0
buffer = 0.25
state = init.copy().astype(float)
orders = np.zeros((6, n))
for w in range(6):
    fc3 = np.array([F[w], F[w + 1], F[w + 2]])
    q = reference_orders(fc3, state, scale=scale, projection=projection, buffer=buffer)
    orders[w] = q
    on_hand, arr1, arr2 = state.T.copy()
    supply = on_hand + arr1
    end = np.maximum(0.0, supply - F[w])
    state = np.column_stack((end, arr2, q))

# write long
rows = []
for wi in range(6):
    for i, s in enumerate(series_ids):
        rows.append(
            {"series_id": s, "week": wi + 1, "quantity": int(np.rint(orders[wi, i]))}
        )
out = pd.DataFrame(rows, columns=["series_id", "week", "quantity"])
out.to_csv(RUN + "/orders.csv", index=False)
print(out.shape)
print(out.head())
assert out.shape == (599 * 6, 3)
assert (out.quantity >= 0).all()
print("done")
