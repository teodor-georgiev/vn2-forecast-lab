import numpy as np, pandas as pd, sys, time

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"."
hist = np.load("hist.npy")
state0 = np.load("state0.npy")
n, H = hist.shape[0], 8
rng = np.random.default_rng(99)


def build_orders(p0, delta, rng):
    p_w = {w: p0 + delta * (w - 3) / 5.0 for w in range(3, 9)}
    K2 = 400
    idx12 = rng.integers(0, hist.shape[1], size=(K2, 2))
    d12 = np.transpose(hist[:, idx12], (1, 0, 2))
    oh, a1, a2 = state0[:, 0], state0[:, 1], state0[:, 2]
    e1 = np.maximum(0.0, oh[None, :] + a1[None, :] - d12[:, :, 0])
    e2 = np.maximum(0.0, e1 + a2[None, :] - d12[:, :, 1])
    I3 = e2.mean(axis=0)
    orders = np.zeros((6, n))
    cum_sup = I3.copy()
    K = 1500
    idx = rng.integers(0, hist.shape[1], size=(K, n, H))
    Dc = hist[np.arange(n)[None, :, None], idx].cumsum(axis=2)
    for w in range(3, 9):
        q = np.quantile(Dc[:, :, w - 1], np.clip(p_w[w], 0, 1), axis=0)
        need = np.maximum(0.0, q - cum_sup)
        orders[w - 3] = need
        cum_sup += need
    return orders


Keval = 600


def mc_cost(orders, rng, K=Keval):
    idx = rng.integers(0, hist.shape[1], size=(K, n, H))
    D = hist[np.arange(n)[None, :, None], idx]
    tot = np.zeros(K)
    for k in range(K):
        tot[k] = simulate(D[k].T, state0, orders)["score"]
    return tot.mean(), tot.std() / np.sqrt(K)


t0 = time.time()
# common random numbers: one demand batch for all plans
idxE = rng.integers(0, hist.shape[1], size=(Keval, n, H))
E = hist[np.arange(n)[None, :, None], idxE]  # (K, n, 8)


def ev(orders):
    tot = np.array([simulate(E[k].T, state0, orders)["score"] for k in range(Keval)])
    return tot.mean(), tot.std() / np.sqrt(Keval)


rows = []
for p0 in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
    for dlt in [-0.10, -0.05, 0.0, 0.05]:
        o = build_orders(p0, dlt, rng)
        m, se = ev(o)
        rows.append((p0, dlt, m, se, o.sum()))
        print(
            f"p0={p0:.2f} d={dlt:+.2f} score={m:8.0f} se={se:5.0f} units={o.sum():7.0f}",
            flush=True,
        )
rows.sort(key=lambda r: r[2])
print("BEST", rows[0][0], rows[0][1], rows[0][2])
print("elapsed", time.time() - t0)
np.save("eval_grid.npy", np.array([(r[0], r[1], r[2], r[4]) for r in rows]))
