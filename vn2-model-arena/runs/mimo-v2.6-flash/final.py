# -*- coding: utf-8 -*-
"""Round 2 final plan.

LightGBM latent-demand forecast (8-step recursive), out-of-sample multi-step
residual scenarios (bias-correlated), per-series coordinate-descent order
optimization against the exact simulator cost model; per-series selection
between optimized plan and a tuned reference-deficit rule.
"""

import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\tools")
from inventory_sim import simulate

RUN = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\mimo-v2.6-flash"
DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"
t0 = time.time()

sales = pd.read_csv(DATA + r"\sales.csv")
avail = pd.read_csv(DATA + r"\availability.csv")
attrs = pd.read_csv(DATA + r"\product_attributes.csv")
init = pd.read_csv(DATA + r"\initial_state.csv")

weeks = sorted(sales["week"].unique())
W = len(weeks)
widx = {w: i for i, w in enumerate(weeks)}
sids = sorted(sales["series_id"].unique())
N = len(sids)
sidx = {s: i for i, s in enumerate(sids)}

Y = np.zeros((W, N))
AV = np.ones((W, N), dtype=bool)
for sid, w, y in zip(sales["series_id"], sales["week"], sales["y"]):
    Y[widx[w], sidx[sid]] = y
for sid, w, a in zip(avail["series_id"], avail["week"], avail["is_available"]):
    AV[widx[w], sidx[sid]] = bool(a)

raw = pd.DataFrame(Y, index=range(W), columns=range(N)).where(AV)
Yf = raw.rolling(53, min_periods=6, center=True).mean()
Yf = raw.fillna(Yf).bfill().ffill().fillna(0.0).values

attrs_a = attrs.set_index("series_id").loc[sids]
attr_cols = {
    c: attrs_a[c].astype("category").cat.codes.values
    for c in [
        "ProductGroup",
        "Division",
        "Department",
        "DepartmentGroup",
        "StoreFormat",
        "Format",
    ]
}

import lightgbm as lgb

IDW = np.arange(N)
WK_PROXY = np.column_stack([IDW % 7, IDW % 13, IDW % 26])


def feats(Ym, avm, t):
    f = {}
    for lag in (1, 2, 3, 4, 8, 13, 26, 52):
        f[f"lag{lag}"] = Ym[t - lag] if t - lag >= 0 else np.zeros(N)
    f["rm4"] = Ym[max(0, t - 3) : t + 1].mean(0)
    f["rm13"] = Ym[max(0, t - 12) : t + 1].mean(0)
    f["rm26"] = Ym[max(0, t - 25) : t + 1].mean(0)
    f["rm52"] = Ym[max(0, t - 51) : t + 1].mean(0)
    f["rmax13"] = Ym[max(0, t - 12) : t + 1].max(0)
    f["std13"] = Ym[max(0, t - 12) : t + 1].std(0)
    f["zero26"] = (Ym[max(0, t - 25) : t + 1] < 1e-9).mean(0)
    f["last"] = Ym[t]
    f["av4"] = avm[max(0, t - 3) : t + 1].mean(0)
    iso = pd.Timestamp(weeks[min(t, W - 1)]).isocalendar().week
    f["isoweek"] = iso
    f["iw_sin"] = np.sin(2 * np.pi * iso / 52.345)
    f["iw_cos"] = np.cos(2 * np.pi * iso / 52.345)
    for c, v in attr_cols.items():
        f[c] = v
    for j in range(3):
        f[f"p{j}"] = WK_PROXY[:, j]
    return pd.DataFrame(f)


def train_model(cut, seed=0, num_boost=500):
    rows_X, rows_y = [], []
    for t in range(53, min(cut, W - 1)):
        X = feats(Yf, AV, t)
        m = AV[t + 1]
        rows_X.append(X[m])
        rows_y.append(np.log1p(Yf[t + 1][m]))
    Xtr = pd.concat(rows_X, ignore_index=True)
    ytr = np.concatenate(rows_y)
    return lgb.train(
        dict(
            objective="regression",
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=200,
            seed=seed,
            num_threads=6,
            verbosity=-1,
        ),
        lgb.Dataset(Xtr, ytr),
        num_boost_round=num_boost,
    )


def recursive_forecast(model, cut, horizon):
    Ym, avm = Yf.copy(), AV.copy()
    fc = np.zeros((horizon, N))
    for h in range(horizon):
        t = cut + h - 1
        X = feats(Ym, avm, t)
        X["av4"] = 1.0
        mu = np.clip(np.expm1(model.predict(X)), 0, None)
        fc[h] = mu
        Ym = np.vstack([Ym, mu])
        avm = np.vstack([avm, np.ones((1, N), bool)])
    return fc


# ---- OOS multi-step residual ratios: train at W-16 and W-8, forecast 8 ahead ----
print("OOS residuals...", flush=True)
rat_parts = []
for cut in (W - 16, W - 8):
    m = train_model(cut)
    fc = recursive_forecast(m, cut, 8)
    act = Yf[cut : cut + 8]
    r = np.clip((act + 0.5) / (fc + 0.5), 0.05, 10.0)
    rat_parts.append(r)
rats_oos = np.vstack(rat_parts)  # (16, N): each row one historical residual week-vector
logr = np.log(rats_oos)
bias = logr.mean(axis=0)  # (N,) mean log bias
dev = logr - bias[None, :]  # deviations
print(
    "mean ratio overall:",
    float(np.exp(logr.mean())),
    "elapsed",
    time.time() - t0,
    flush=True,
)

# ---- final model on full history ----
print("final model...", flush=True)
model = train_model(W)
mu8 = recursive_forecast(model, W, 8)
np.save(RUN + r"\mu8_final.npy", mu8)
print("forecast done, elapsed", time.time() - t0, flush=True)


# ---- scenarios with correlated bias ----
def make_scenarios(n_scen, seed):
    rng = np.random.default_rng(seed)
    S = n_scen
    # shared series-level shock: mean of 4 sampled deviation weeks
    k = rng.integers(0, dev.shape[0], size=(S, 4))
    shared = np.zeros((S, 8, N))
    for j in range(4):
        shared += dev[k[:, j]][:, None, :] / 4.0
    # independent weekly deviation: pick random (week, series) deviation rows per scenario-week
    wi = rng.integers(0, dev.shape[0], size=(S, 8))
    noise = dev[wi]  # (S, 8, N)
    L = bias[None, None, :] + 0.7 * shared + 0.7 * noise
    dem = np.clip(mu8[None, :, :] * np.exp(L), 0, None)  # (S, 8, N)
    return dem


def optimize_orders(
    dem,
    init_state,
    passes=3,
    qs=(0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99),
):
    """coordinate descent on 6 order weeks minimizing expected cost weeks 3-8."""
    n_scen = dem.shape[0]
    n = N
    orders = np.zeros((6, n))
    base = np.broadcast_to(init_state[:, None, :], (n, n_scen, 3)).copy()

    def cost_of(ords):
        st = base.copy()
        tot = np.zeros((n, n_scen))
        for w in range(8):
            oh, a1, a2 = st[..., 0], st[..., 1], st[..., 2]
            sup = oh + a1
            d = dem[:, w, :].T
            end = np.maximum(0.0, sup - d)
            mis = np.maximum(0.0, d - sup)
            if w >= 2:
                tot += mis + 0.2 * end
            o = ords[w] if w < 6 else np.zeros(n)
            st = np.stack(
                [end, a2, np.broadcast_to(np.asarray(o)[:, None], (n, n_scen))], axis=-1
            )
        return tot.mean(1), tot

    best, _ = cost_of(orders)
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
                cst, _ = cost_of(trial)
                upd = cst < best - 1e-12
                best = np.where(upd, cst, best)
                best_orders[w][upd] = trial[w][upd]
            orders = best_orders.copy()
    cst_opt, _ = cost_of(best_orders)
    return best_orders, cst_opt


def reference_plan(mu9, init_state, buf):
    stt = init_state.copy()
    ords = np.zeros((6, N))
    for w in range(6):
        a = np.maximum(0, mu9[w + 1])
        b = np.maximum(0, mu9[w + 2])
        c = np.maximum(0, mu9[w + 3])
        oh, a1, a2 = stt.T
        proj = np.maximum(0.0, oh + a1 + a2 - (a + b))
        o = np.rint(np.clip(c + buf * np.sqrt(c) - proj, 0, None))
        ords[w] = o
        end = np.maximum(0.0, (oh + a1) - np.maximum(0, mu9[w]))
        stt = np.column_stack([end, a2, o])
    return ords


def eval_plan(ords, dem_scen, init_state):
    """expected cost per series under scenarios"""
    n_scen = dem_scen.shape[0]
    base = np.broadcast_to(init_state[:, None, :], (N, n_scen, 3)).copy()
    st = base.copy()
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


init_state = (
    init.set_index("series_id")
    .loc[sids][["on_hand", "arrival_w1", "arrival_w2"]]
    .values.astype(float)
)

mu9 = np.vstack([mu8, mu8[-1][None, :]])
dem = make_scenarios(128, seed=7)
print("optimizing...", flush=True)
ords_opt, cst_opt = optimize_orders(dem, init_state)
print(
    "opt expected cost", float(cst_opt.sum()), "elapsed", time.time() - t0, flush=True
)

# reference plans with a few buffers + small forecast scales
best_ref_ords, best_ref_c = None, None
for scale in (0.95, 1.0, 1.05, 1.1):
    for buf in (0.75, 1.0, 1.25, 1.5, 2.0):
        o = reference_plan(mu9 * scale, init_state, buf)
        c = eval_plan(o, dem, init_state)
        s = c.sum()
        if best_ref_c is None or s < best_ref_c:
            best_ref_c, best_ref_ords, best_ref_desc = s, o, (scale, buf)
print("best ref expected", float(best_ref_c), best_ref_desc, flush=True)

# per-series selection under the same scenario estimate
c_opt = eval_plan(ords_opt, dem, init_state)
c_ref = eval_plan(best_ref_ords, dem, init_state)
final = np.where(c_opt <= c_ref, ords_opt, best_ref_ords)
print(
    "blend expected cost",
    float(np.minimum(c_opt, c_ref).sum()),
    "opt wins series:",
    int((c_opt < c_ref).sum()),
    flush=True,
)

# sanity check with simulator on the scenario-mean demand path and on biased paths
mean_dem = dem.mean(0)
r = simulate(mean_dem, init_state, final)
print("sim on mean path:", round(r["score"], 1), flush=True)

df = pd.DataFrame(
    {
        "series_id": np.repeat(sids, 6),
        "week": np.tile(np.arange(1, 7), N),
        "quantity": np.rint(final.T).ravel().astype(int),
    }
)
assert len(df) == N * 6 and (df["quantity"] >= 0).all()
df.to_csv(RUN + r"\orders.csv", index=False)
print(
    "wrote orders.csv rows",
    len(df),
    "total units",
    int(df["quantity"].sum()),
    "elapsed",
    time.time() - t0,
)
