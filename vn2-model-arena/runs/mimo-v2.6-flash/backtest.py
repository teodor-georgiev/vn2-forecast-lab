# -*- coding: utf-8 -*-
"""Round 2: latent-demand LightGBM forecasts + scenario-based order optimization."""

import sys, time, json
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

# latent demand fill for unavailable weeks
raw = pd.DataFrame(Y, index=range(W), columns=range(N)).where(AV)
Yf = raw.rolling(53, min_periods=6, center=True).mean()
Yf = raw.fillna(Yf).bfill().ffill().fillna(0.0).values

attrs = attrs.set_index("series_id").loc[sids]
attr_cols = {
    c: attrs[c].astype("category").cat.codes.values
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


def feats(Ym, avm, t, hname):
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


def train_forecast(cut, horizon, seed=0, num_boost=500):
    """train on weeks 0..cut, forecast horizon weeks after cut; also return one-step
    preds for weeks cut-25..cut for residual estimation."""
    rows_X, rows_y = [], []
    for t in range(53, cut):
        X = feats(Yf, AV, t, "tr")
        m = AV[t + 1]
        rows_X.append(X[m])
        rows_y.append(np.log1p(Yf[t + 1][m]))
    Xtr = pd.concat(rows_X, ignore_index=True)
    ytr = np.concatenate(rows_y)
    model = lgb.train(
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
    Ym, avm = Yf.copy(), AV.copy()
    fc = np.zeros((horizon, N))
    for h in range(horizon):
        t = cut + h
        X = feats(Ym, avm, t, "fc")
        X["av4"] = 1.0
        mu = np.clip(np.expm1(model.predict(X)), 0, None)
        fc[h] = mu
        Ym = np.vstack([Ym, mu])
        avm = np.vstack([avm, np.ones((1, N), bool)])
    # one-step in-sample preds for last 30 training weeks (residuals)
    preds = []
    for t in range(max(53, cut - 30), cut):
        X = feats(Yf, AV, t, "tr")
        preds.append((t + 1, np.clip(np.expm1(model.predict(X)), 0, None)))
    return fc, preds


def recent_cv(cut):
    """per-series multiplicative sigma from last 26 available weeks"""
    lo = max(0, cut - 26)
    seg = Yf[lo:cut]
    m = seg.mean(0)
    s = seg.std(0)
    cv = np.where(m > 0.15, s / np.maximum(m, 1e-9), 1.0)
    return np.clip(cv, 0.20, 1.6)


# ---------------- order optimizer ----------------
def residual_ratios(cut, preds):
    """empirical per-series multiplicative residual ratios (y+0.5)/(pred+0.5)"""
    rats = []
    for t, p in preds:
        if t < cut:
            r = (Yf[t] + 0.5) / (p + 0.5)
            rats.append(r)
    return np.clip(np.array(rats), 0.05, 8.0)  # (T, N)


def optimize_orders(
    mu8,
    init_state,
    rats,
    n_scen=96,
    passes=3,
    seed=1,
    qs=(0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99),
):
    """mu8: (8,N); rats (T,N) empirical ratios; returns (6,N) integer orders
    minimizing expected cost weeks 3-8 over empirical scenario demand."""
    rng = np.random.default_rng(seed)
    T = rats.shape[0]
    idx = rng.integers(0, T, size=(n_scen, 8))
    ratio = rats[idx]  # (S,8,N)
    dem = np.clip(mu8[None, :, :] * ratio, 0, None)
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
        return tot.mean(1)

    cur = cost_of(orders)
    best = cur.copy()
    best_orders = orders.copy()
    for p in range(passes):
        for w in range(6):
            # candidate order levels: scenario quantiles of arrival-week demand
            dq = np.quantile(dem[:, w + 2, :], qs, axis=0)  # (len(qs), N)
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
                o = cands[i]
                trial = orders.copy()
                trial[w] = o
                cst = cost_of(trial)
                upd = cst < best - 1e-12
                best = np.where(upd, cst, best)
                best_orders[w][upd] = trial[w][upd]
            orders = best_orders.copy()
    return best_orders


# ---------------- backtest ----------------
def bt_initial_state(cut):
    """synthetic starting state at cutoff: on_hand = last week demand (avail), arrivals 0"""
    last = Yf[cut - 1]
    return np.column_stack([last, np.zeros(N), np.zeros(N)])


results = {}
for cut in [W - 16, W - 8]:
    mu9, preds = train_forecast(cut, 9)
    mu8 = mu9[:8]
    rats = residual_ratios(cut, preds)
    st = bt_initial_state(cut)
    dem_true = Yf[cut : cut + 8]
    ref_scores = []
    for buf in [0.0, 0.5, 1.0, 1.5]:
        ords = np.zeros((6, N))
        # reference-style rolling: order each week w using h1..h3 of that week
        stt = st.copy()
        for w in range(6):
            a = np.maximum(0, mu9[w + 1])
            b = np.maximum(0, mu9[w + 2])
            c = np.maximum(0, mu9[w + 3])
            oh, a1, a2 = stt.T
            proj = np.maximum(0.0, oh + a1 + a2 - (a + b))
            d = c + buf * np.sqrt(c) - proj
            o = np.rint(np.clip(d, 0, None))
            ords[w] = o
            # advance state one week
            sup = oh + a1
            end = np.maximum(0.0, sup - dem_true[w])
            stt = np.column_stack([end, a2, o])
        r = simulate(dem_true, st, ords)
        ref_scores.append((buf, r["score"]))
    print("cut", weeks[cut], "ref:", ref_scores, flush=True)
    results[str(cut)] = {"ref": ref_scores}
    for nsc, ps in [(96, 3), (192, 3)]:
        ords = optimize_orders(mu8, st, rats, n_scen=nsc, passes=ps, seed=42)
        r = simulate(dem_true, st, ords)
        print("  opt n_scen", nsc, "score", round(r["score"], 1), flush=True)
        results[str(cut)].setdefault("opt", []).append((nsc, r["score"]))

print(json.dumps(results, indent=1))
print("elapsed", time.time() - t0)
