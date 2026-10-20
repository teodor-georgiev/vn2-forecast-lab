import numpy as np, pandas as pd, sys

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\glm-5.3-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
hist = np.load(RUN + r"\hist.npy")
state0 = np.load(RUN + r"\state0.npy")
fc = np.load(RUN + r"\fc.npy")
n = hist.shape[0]
H = 8
P0, DELTA = 0.50, -0.05

scale = np.clip(fc.mean(axis=1) / np.maximum(hist.mean(axis=1), 1e-9), 0.7, 1.4)
Hs = hist * scale[:, None]

rng = np.random.default_rng(20260214)


def build(p0, delta, HIST):
    p_w = {w: p0 + delta * (w - 3) / 5.0 for w in range(3, 9)}
    K2 = 400
    d12 = np.transpose(HIST[:, rng.integers(0, HIST.shape[1], size=(K2, 2))], (1, 0, 2))
    oh, a1, a2 = state0[:, 0], state0[:, 1], state0[:, 2]
    e2 = np.maximum(
        0.0,
        np.maximum(0.0, oh[None, :] + a1[None, :] - d12[:, :, 0])
        + a2[None, :]
        - d12[:, :, 1],
    )
    I3 = e2.mean(axis=0)
    orders = np.zeros((6, n))
    cum = I3.copy()
    K = 1500
    Dc = HIST[
        np.arange(n)[None, :, None], rng.integers(0, HIST.shape[1], size=(K, n, H))
    ].cumsum(axis=2)
    for w in range(3, 9):
        q = np.quantile(Dc[:, :, w - 1], np.clip(p_w[w], 0, 1), axis=0)
        need = np.maximum(0.0, q - cum)
        orders[w - 3] = need
        cum += need
    return np.rint(np.maximum(orders, 0.0))


orders = build(P0, DELTA, Hs)
sales = pd.read_csv(DATA + r"\sales.csv")
series = np.sort(sales["series_id"].unique())
assert len(series) == 599
# orders.npy was built in st (sales-pivot) order == sales file order; reindex via stored state file order
st_index = pd.read_csv(RUN + r"\sales_wide.csv", index_col=0).index.to_numpy()
pos = {s: i for i, s in enumerate(st_index)}
perm = np.array([pos[s] for s in series])
o = orders[:, perm]

out = pd.DataFrame(
    {
        "series_id": np.repeat(series, 6),
        "week": np.tile(np.arange(1, 7), 599),
        "quantity": o.T.reshape(-1),
    }
)
assert len(out) == 3594 and out.quantity.min() >= 0 and out.quantity.max() < 1e7
out.to_csv(RUN + r"\orders.csv", index=False)

# smoke test: simulate the plan against a sample demand draw to prove compatibility
q = pd.read_csv(RUN + r"\orders.csv")
opa = np.zeros((6, 599))
for w in range(1, 7):
    opa[w - 1] = q.loc[q.week == w, "quantity"].to_numpy()
print("plan ok", out.shape, "total units", opa.sum())
res = simulate(
    Hs[np.arange(n)[None, :, None], rng.integers(0, Hs.shape[1], size=(1, n, H))][0].T,
    state0[perm],
    opa,
)
print("smoke sim score", res["score"])
