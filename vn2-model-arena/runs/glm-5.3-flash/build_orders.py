"""Round 2: static order plan for 599 series, 6 weeks.

Method: recency-weighted level+trend forecast; empirical weekly demand
distribution via bootstrap of recent weeks; cumulative-quantile ordering rule
with per-horizon newsvendor quantiles tuned by Monte Carlo on the exact simulator.
"""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\glm-5.3-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"

sales = pd.read_csv(DATA + r"\sales.csv")
avail = pd.read_csv(DATA + r"\availability.csv")
init = pd.read_csv(DATA + r"\initial_state.csv")

st = sales.pivot(index="series_id", columns="week", values="y")
av = avail.pivot(index="series_id", columns="week", values="is_available")
st = st.reindex(index=av.index, columns=av.columns)
assert set(st.index) == set(init["series_id"]), "series mismatch"
init = init.set_index("series_id").reindex(st.index).reset_index()

Y = st.to_numpy(dtype=float)  # (599, 157) weeks ordered
A = av.to_numpy(dtype=bool)
state0 = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)  # (599,3)
n = Y.shape[0]

# --- demand-adjusted history: unavailable weeks treated as missing -> impute ---
Yd = np.where(A, Y, np.nan)
# per-series mean of available weeks (fallback: overall mean)
valid = np.isfinite(Yd)
sum_v = np.where(valid, np.nan_to_num(Yd), 0.0)
cnt = valid.sum(axis=1)
mu_s = np.where(
    cnt > 0, sum_v.sum(axis=1) / np.maximum(cnt, 1), Y[~A].mean() if (~A).any() else 0.0
)
Yadj = np.where(valid, Yd, mu_s[:, None])

# --- calendar seasonal index for the 8 upcoming weeks (2022-01-10 .. 2022-02-28) ---
weeks_dt = pd.to_datetime(st.columns)
iso = pd.Series([d.isocalendar()[1] for d in weeks_dt])
overall = Yadj.mean()
iso_mean = pd.Series(np.nanmean(Yadj, axis=0)).groupby(iso.values).mean()
gmean = iso_mean.mean()
idx_all = (iso_mean / gmean).clip(
    0.7, 1.4
)  # weekly-of-year avg demand index across all series
future_dates = pd.date_range(weeks_dt[-1], periods=9, freq="7D")[1:9]
fut_iso = np.array([d.isocalendar()[1] for d in future_dates])

# per-ISO-week index normalised to the mean of the 8 upcoming ISO weeks
vals = np.array([float(idx_all.get(i, gmean)) for i in fut_iso])
fut_idx = vals / vals.mean()

# --- forecast: level (last 13 weeks) with mild damped trend to last 26 ---
lvl13 = Yadj[:, -13:].mean(axis=1)
lvl26 = Yadj[:, -26:].mean(axis=1)
lvl52 = Yadj[:, -52:].mean(axis=1)
r = np.where(lvl26 > 1e-9, lvl13 / np.maximum(lvl26, 1e-9), 1.0)
r = np.clip(r, 0.85, 1.15)
# only trust trend where volume allows; tiny series -> 1.0
r = np.where(lvl26 >= 1.0, r, 1.0)
H = 8
t = np.arange(1, H + 1)
damp = 0.7 ** (t / 26.0 * 0)  # keep flat: trend applied via r only
fc = lvl13[:, None] * (r[:, None] ** (t[None, :] / 13.0)) * fut_idx[None, :]
fc = np.clip(fc, 0.0, None)

# --- empirical weekly distribution for Monte-Carlo: last 26 adjusted weeks ---
hist = Yadj[:, -26:]
rng = np.random.default_rng(7)


def build_orders(p0, delta, I3_hi=0.5):
    """Cumulative-quantile rule. p_w = p0 + delta*(w-3)/5 for scored horizons."""
    p_w = {w: p0 + delta * (w - 3) / 5.0 for w in range(3, 9)}
    # expected inventory entering week 3 (E[e2]) via small MC on weeks 1-2
    K2 = 400
    idx12 = rng.integers(0, hist.shape[1], size=(K2, 2))
    d12 = hist[:, idx12]  # shape (n, K2, 2)
    d12 = np.transpose(d12, (1, 0, 2))  # (K2, n, 2)
    oh, a1, a2 = state0[:, 0], state0[:, 1], state0[:, 2]
    e1 = np.maximum(0.0, oh[None, :] + a1[None, :] - d12[:, :, 0])
    e2 = np.maximum(0.0, e1 + a2[None, :] - d12[:, :, 1])
    I3 = e2.mean(axis=0)
    orders = np.zeros((6, n))
    cum_sup = I3.copy()
    cum_q = np.zeros((6, n))
    # cumulative demand quantiles for horizons 3..8
    K = 1200
    idxK = rng.integers(0, hist.shape[1], size=(K, n, H))
    D = hist[np.arange(n)[None, :, None], idxK]  # (K, n, 8)
    Dc = D.cumsum(axis=2)
    for w in range(3, 9):
        p = np.clip(p_w[w], 0.0, 1.0)
        q = np.quantile(Dc[:, :, w - 1], p, axis=0)  # (n,)
        cum_q[w - 3] = q
        need = np.maximum(0.0, q - cum_sup)
        orders[w - 3] = need
        cum_sup = cum_sup + need
    return orders


# --- Monte Carlo cost of a plan under the exact simulator ---
def mc_cost(orders, K=300):
    idxK = rng.integers(0, hist.shape[1], size=(K, n, H))
    D = hist[np.arange(n)[None, :, None], idxK]  # (K, n, 8)
    tot = np.zeros(K)
    for k in range(K):
        res = simulate(D[k].T, state0, orders)  # D[k]: (n,8) -> transpose (8,n)
        tot[k] = res["score"]
    return tot.mean(), tot.std() / np.sqrt(K)


import itertools, time

t0 = time.time()
paths = sorted(itertools.product([0.70, 0.80, 0.90], [0.0, 0.05, 0.10]))
results = []
for p0, dlt in paths:
    o = build_orders(p0, dlt)
    m, se = mc_cost(o, K=150)
    results.append((p0, dlt, m, se, o.sum()))
    print(
        f"p0={p0:.2f} delta={dlt:.2f}  score={m:,.0f}  se={se:,.0f}  units={o.sum():,.0f}",
        flush=True,
    )

results.sort(key=lambda x: x[2])
p0, dlt, m, se, _ = results[0]
print("BEST", p0, dlt, m, flush=True)

# refine around best
fine = sorted(
    itertools.product([p0 - 0.05, p0, p0 + 0.05], [dlt - 0.05, dlt, dlt + 0.05])
)
fine = [(a, b) for a, b in fine if a > 0.3]
res2 = []
for p0b, dltb in fine:
    o = build_orders(p0b, dltb)
    m2, se2 = mc_cost(o, K=300)
    res2.append((p0b, dltb, m2))
    print(f"fine p0={p0b:.2f} delta={dltb:.2f} score={m2:,.0f}", flush=True)
res2.sort(key=lambda x: x[2])
p0, dlt, m = res2[0]
print("FINAL", p0, dlt, m, flush=True)

orders = build_orders(p0, dlt)
np.save(RUN + r"\orders.npy", orders)
np.save(RUN + r"\state0.npy", state0)
np.save(RUN + r"\fc.npy", fc)
np.save(RUN + r"\hist.npy", hist)
json_meta = {"p0": p0, "delta": dlt, "mc_score": float(m)}
with open(RUN + r"\meta.txt", "w") as f:
    f.write(str(json_meta) + f"\nelapsed={time.time() - t0:.0f}s")

out = pd.DataFrame(
    {
        "series_id": np.repeat(st.index.to_numpy(), 6),
        "week": np.tile(np.arange(1, 7), n),
        "quantity": np.rint(orders.T.reshape(-1)).astype(float),
    }
)
out.to_csv(RUN + r"\orders.csv", index=False)
print("written", out.shape, flush=True)
print("total elapsed", time.time() - t0, flush=True)
