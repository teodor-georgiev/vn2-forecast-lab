import numpy as np, pandas as pd, sys

sys.path.insert(0, "../../tools")
from inventory_sim import simulate

L = np.load("L.npy")
A = np.load("A.npy")
n, T = L.shape
n = L.shape[0]
n = L.shape[0]


# ---------- forecaster (level-based, chosen by backtest MAE) ----------
def fc(cut, k=13):
    out = np.zeros((8, n))
    m = L[:, max(0, cut - k + 1) : cut + 1].mean(axis=1)
    out[:] = np.maximum(m, 0)[None, :]
    return out


# ---------- per-series log-error sigma ----------
cuts = (85, 98, 111, 124, 137)
res = []
for cut in cuts:
    f = fc(cut)  # (8,n)
    y = L[:, cut + 1 : cut + 9].T  # (8,n)
    r = np.log((y + 0.5) / (f + 0.5))
    res.append(r)
res = np.concatenate(res, axis=0)  # (40,n)
var_i = res.var(axis=0)
k = res.shape[0]
gvar = np.median(var_i)
sig_i = np.sqrt((k * var_i + 6 * gvar) / (k + 6))
sig_i = np.clip(sig_i, 0.25, 1.1)
print("sigma med", np.median(sig_i), "range", sig_i.min(), sig_i.max())


# ---------- scenario cost ----------
def scen_cost(dem, initial, orders):
    """dem (S,8,n), orders (6,n) -> (n,) expected scored cost (weeks 3-8)."""
    S = dem.shape[0]
    on = np.broadcast_to(initial[:, 0], (S, n)).copy()
    a1 = np.broadcast_to(initial[:, 1], (S, n)).copy()
    a2 = np.broadcast_to(initial[:, 2], (S, n)).copy()
    tot = np.zeros(n)
    for w in range(8):
        supply = on + a1
        d = dem[:, w, :]
        end = np.maximum(0.0, supply - d)
        missed = np.maximum(0.0, d - supply)
        if w >= 2:
            tot += missed.sum(axis=0) + 0.2 * end.sum(axis=0)
        on = end
        a1 = a2
        a2 = np.broadcast_to(orders[w], (S, n)) if w < 6 else np.zeros(n)
    return tot


def optimize(initial, mu, sig, S=300, seed=0, sweeps=3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((S, 8, n))
    dem = mu[None, :, :] * np.exp(
        sig[None, None, :] * z - 0.5 * sig[None, None, :] ** 2
    )
    dem = np.maximum(dem, 0.0)
    q = np.zeros((6, n))
    # initialise with forecast of the week arriving (w+2)
    for w in range(6):
        q[w] = mu[min(w + 2, 7)]

    def cost(qq):
        return scen_cost(dem, initial, qq)

    for sweep in range(sweeps):
        for w in range(6):
            lo = np.zeros(n)
            hi = np.maximum(10.0, mu.max(axis=0) * 8 + 20)
            # ternary search (vectorised, per-series bounds)
            for _ in range(22):
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                q[w] = m1
                c1 = cost(q)
                q[w] = m2
                c2 = cost(q)
                left = c1 < c2
                hi = np.where(left, m2, hi)
                lo = np.where(left, lo, m1)
            q[w] = np.rint((lo + hi) / 2)
    return q


# ---------- backtest the whole policy on historical windows ----------
initial = pd.read_csv("../../data/initial_state.csv")
ids = np.load("ids.npy", allow_pickle=True)
st = (
    initial.set_index("series_id")
    .loc[list(ids)][["on_hand", "arrival_w1", "arrival_w2"]]
    .values
)
for cut in (98, 111, 124, 137):
    mu = fc(cut)
    q = optimize(st, mu, sig_i, S=150, sweeps=2)
    d = L[:, cut+1:cut+9].T
    r = simulate(d, st, q)
    qref = np.zeros((6, n))
    for w in range(6):
        qref[w] = np.rint(mu[min(w + 2, 7)])
    r2 = simulate(d, st, qref)
    print(cut, "opt", round(r["score"]), "naive", round(r2["score"]))
