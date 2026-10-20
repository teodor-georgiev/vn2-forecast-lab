import numpy as np, pandas as pd, sys, itertools

sys.path.insert(0, "../../tools")
from inventory_sim import simulate

L = np.load("L.npy")
n = L.shape[0]


def fc(cut, k=13):
    out = np.zeros((8, n))
    m = L[:, max(0, cut - k + 1) : cut + 1].mean(axis=1)
    out[:] = np.maximum(m, 0)[None, :]
    return out


ids = np.load("ids.npy", allow_pickle=True)
initial = pd.read_csv("../../data/initial_state.csv")
st = (
    initial.set_index("series_id")
    .loc[list(ids)][["on_hand", "arrival_w1", "arrival_w2"]]
    .values
)

cuts = (98, 111, 124, 137)
fcst = {c: fc(c) for c in cuts}
ys = {c: L[:, c + 1 : c + 9].T for c in cuts}


def policy(mu, s, b, late_boost):
    q = np.zeros((6, n))
    for w in range(6):
        h = w + 2
        mult = s * (late_boost if w >= 4 else 1.0)
        q[w] = np.maximum(0, np.rint(mult * mu[min(h, 7)] + b))
    return q


rows = []
for s, b, lb in itertools.product(
    [0.9, 1.0, 1.1, 1.25, 1.4], [-0.5, 0, 0.5, 1.0, 2.0], [1.0, 1.2, 1.5]
):
    tot = 0
    for c in cuts:
        tot += simulate(ys[c], st, policy(fcst[c], s, b, lb))["score"]
    rows.append((tot, s, b, lb))
rows.sort()
for r in rows[:12]:
    print(round(r[0]), r[1], r[2], r[3])
np.save("grid_rows.npy", np.array(rows))
