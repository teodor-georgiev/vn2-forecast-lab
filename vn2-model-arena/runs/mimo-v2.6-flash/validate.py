# -*- coding: utf-8 -*-
"""Validate final machinery at a historical cutoff: does opt beat ref out-of-sample?"""

import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\mimo-v2.6-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
t0 = time.time()

import importlib.util

spec = importlib.util.spec_from_file_location("fin", RUN + r"\final.py")
# can't import final (it runs); duplicate minimal pieces by exec of selected lines is messy.
# Instead: load the shared data prep by exec'ing the file up to the marker.
src = open(RUN + r"\final.py").read()
head = src.split("# ---- OOS multi-step residual ratios")[0]
exec(
    head
)  # provides Yf, AV, N, W, weeks, feats, train_model, recursive_forecast, sids, init? no init

init = pd.read_csv(DATA + r"\initial_state.csv")
init_state = (
    init.set_index("series_id")
    .loc[sids][["on_hand", "arrival_w1", "arrival_w2"]]
    .values.astype(float)
)

cut = W - 8
# residuals from earlier OOS windows (models trained at W-32, W-24) to avoid leak
rat_parts = []
for c2 in (W - 32, W - 24):
    m2 = train_model(c2)
    fc2 = recursive_forecast(m2, c2, 8)
    r2 = np.clip((Yf[c2 : c2 + 8] + 0.5) / (fc2 + 0.5), 0.05, 10.0)
    rat_parts.append(r2)
logr = np.log(np.vstack(rat_parts))
bias = logr.mean(0)
dev = logr - bias[None, :]
print("oos mean ratio", float(np.exp(logr.mean())), flush=True)

mcut = train_model(cut)
mu8 = recursive_forecast(mcut, cut, 8)
mu9 = np.vstack([mu8, mu8[-1][None, :]])
dem_true = Yf[cut : cut + 8]


def make_scenarios(n_scen, seed):
    rng = np.random.default_rng(seed)
    k = rng.integers(0, dev.shape[0], size=(n_scen, 4))
    shared = np.zeros((n_scen, 8, N))
    for j in range(4):
        shared += dev[k[:, j]][:, None, :] / 4.0
    wi = rng.integers(0, dev.shape[0], size=(n_scen, 8))
    noise = dev[wi]
    L = bias[None, None, :] + 0.7 * shared + 0.7 * noise
    return np.clip(mu8[None, :, :] * np.exp(L), 0, None)


def eval_plan(ords, dem_scen):
    n_scen = dem_scen.shape[0]
    base = np.broadcast_to(init_state[:, None, :], (N, n_scen, 3)).copy()
    st = base
    tot = np.zeros((N, n_scen))
    for w in range(8):
        oh, a1, a2 = st[..., 0], st[..., 1], st[..., 2]
        sup = oh + a1
        d = dem_scen[:, w, :].T
        end = np.maximum(0.0, sup - d)
        mis = np.maximum(0.0, d - sup)
        if w >= 2:
            tot += mis + 0.2 * end
        o = ords[w] if w < 6 else np.zeros(N)
        st = np.stack(
            [end, a2, np.broadcast_to(np.asarray(o)[:, None], (N, n_scen))], axis=-1
        )
    return tot.mean(1)


def optimize_orders(
    dem,
    passes=3,
    qs=(0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99),
):
    n_scen = dem.shape[0]
    orders = np.zeros((6, N))
    base = np.broadcast_to(init_state[:, None, :], (N, n_scen, 3)).copy()

    def cost_of(ords):
        st = base
        tot = np.zeros((N, n_scen))
        for w in range(8):
            oh, a1, a2 = st[..., 0], st[..., 1], st[..., 2]
            sup = oh + a1
            d = dem[:, w, :].T
            end = np.maximum(0.0, sup - d)
            mis = np.maximum(0.0, d - sup)
            if w >= 2:
                tot += mis + 0.2 * end
            o = ords[w] if w < 6 else np.zeros(N)
            st = np.stack(
                [end, a2, np.broadcast_to(np.asarray(o)[:, None], (N, n_scen))], axis=-1
            )
        return tot.mean(1)

    best = cost_of(orders)
    best_orders = orders.copy()
    for p in range(passes):
        for w in range(6):
            dq = np.quantile(dem[:, w + 2, :], qs, axis=0)
            mref = mu8[w + 2]
            cands = np.rint(
                np.vstack(
                    [
                        dq,
                        mref[None, :]
                        * np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])[
                            :, None
                        ],
                    ]
                )
            )
            for i in range(cands.shape[0]):
                trial = orders.copy()
                trial[w] = cands[i]
                cst = cost_of(trial)
                upd = cst < best - 1e-12
                best = np.where(upd, cst, best)
                best_orders[w][upd] = trial[w][upd]
            orders = best_orders.copy()
    return best_orders


def reference_plan(mu9v, buf):
    stt = init_state.copy()
    ords = np.zeros((6, N))
    for w in range(6):
        a = np.maximum(0, mu9v[w + 1])
        b = np.maximum(0, mu9v[w + 2])
        c = np.maximum(0, mu9v[w + 3])
        oh, a1, a2 = stt.T
        proj = np.maximum(0.0, oh + a1 + a2 - (a + b))
        o = np.rint(np.clip(c + buf * np.sqrt(c) - proj, 0, None))
        ords[w] = o
        end = np.maximum(0.0, (oh + a1) - np.maximum(0, mu9v[w]))
        stt = np.column_stack([end, a2, o])
    return ords


dem = make_scenarios(128, seed=7)
ords_opt = optimize_orders(dem)
best = None
for buf in (0.75, 1.0, 1.25, 1.5, 2.0):
    o = reference_plan(mu9, buf)
    c = eval_plan(o, dem).sum()
    print(
        "ref buf",
        buf,
        "scen-cost",
        round(c, 1),
        "true",
        round(simulate(dem_true, init_state, o)["score"], 1),
        flush=True,
    )
    if best is None or c < best[0]:
        best = (c, o, buf)
print("best ref buf", best[2], flush=True)
ords_ref = best[1]
c_opt = eval_plan(ords_opt, dem)
c_ref = eval_plan(ords_ref, dem)
final = np.where(c_opt <= c_ref, ords_opt, ords_ref)
print("TRUE score opt:", round(simulate(dem_true, init_state, ords_opt)["score"], 1))
print("TRUE score ref:", round(simulate(dem_true, init_state, ords_ref)["score"], 1))
print("TRUE score blend:", round(simulate(dem_true, init_state, final)["score"], 1))
print("elapsed", time.time() - t0)
