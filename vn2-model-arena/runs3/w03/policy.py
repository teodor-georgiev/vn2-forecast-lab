"""Adaptive VN2 inventory policy (worker w03)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle

HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / ".." / "data").resolve()

BASE_N = None
HAS_DATA = False
BASE_Y = None
BASE_STOCK = None
INITIAL = None
STATIC = None
DAYS = None
MODELS = None
SERIES_IDS = None
CORR = None
CACHE_PATH = HERE / "models_cache.pkl"

PARAMS = {
    "scale": 1.2786126610897948,
    "projection": 1.2993357000756158,
    "buffer": 0.5605104664228981,
    "policy": "mean",
    "mix": 0.030370144022356646,
    "cap": 1.996857867734481,
    "reorder": 1.0,
    "adapt_strength": 0.0,
}


def _load_data():
    global BASE_N, HAS_DATA, BASE_Y, BASE_STOCK, INITIAL, STATIC, DAYS, SERIES_IDS
    sales_path = DATA_DIR / "sales.csv"
    avail_path = DATA_DIR / "availability.csv"
    attrs_path = DATA_DIR / "product_attributes.csv"
    init_path = DATA_DIR / "initial_state.csv"
    if not (
        sales_path.exists()
        and avail_path.exists()
        and attrs_path.exists()
        and init_path.exists()
    ):
        return False
    try:
        sales = pd.read_csv(sales_path)
        avail = pd.read_csv(avail_path)
        attrs = pd.read_csv(attrs_path)
        init = pd.read_csv(init_path)

        series_ids = sorted(sales["series_id"].unique())
        weeks = sorted(sales["week"].unique())
        sp = sales.pivot(index="series_id", columns="week", values="y")
        sp = sp.loc[series_ids, weeks]
        ap = avail.pivot(index="series_id", columns="week", values="is_available")
        ap = ap.loc[series_ids, weeks]

        y = sp.to_numpy(dtype=float).T.copy()  # (157, n)
        stock = ap.to_numpy().astype(bool).T.copy()  # (157, n)

        init = init.set_index("series_id").loc[series_ids]
        iv = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)

        attrs = attrs.set_index("series_id").loc[series_ids]
        static_cols = [c for c in attrs.columns]
        static = np.column_stack(
            [pd.factorize(attrs[c], sort=True)[0] for c in static_cols]
        ).astype("float32")

        n_weeks_total = 157 + 8
        days = pd.date_range(start="2019-01-07", periods=n_weeks_total, freq="W-MON")

        BASE_Y = y
        BASE_STOCK = stock
        INITIAL = iv
        STATIC = static
        DAYS = days
        SERIES_IDS = series_ids
        BASE_N = y.shape[1]
        HAS_DATA = True
        return True
    except Exception:
        HAS_DATA = False
        return False


def imputed(y, instock, cutoff):
    out = y[:cutoff].copy().astype(float)
    for t in range(cutoff):
        from_t = max(0, t - 13)
        mask = instock[from_t:t]
        ct = mask.sum(axis=0)
        mean = (out[from_t:t] * mask).sum(axis=0) / np.maximum(ct, 1)
        out[t] = np.where(~instock[t] & (ct >= 3), np.maximum(out[t], mean), out[t])
    return out


def make_features_ewma(hist, days, static, t, h):
    n = hist.shape[1]
    feat = [static]
    for lag in (1, 2, 3, 4, 8, 13, 26, 52):
        feat.append(hist[t - lag, :, None])
    for window in (4, 8, 13, 26, 52):
        recent = hist[t - window : t]
        feat.extend(
            [
                recent.mean(axis=0)[:, None],
                recent.std(axis=0)[:, None],
                (recent <= 0).mean(axis=0)[:, None],
            ]
        )
    annual = t + h - 1 - 52
    feat.append(hist[annual, :, None])
    feat.append(hist[max(0, annual - 2) : min(t, annual + 3)].mean(axis=0)[:, None])
    recent13 = hist[t - 13 : t].mean(axis=0)
    feat.append(((hist[t - 4 : t].mean(axis=0) + 0.1) / (recent13 + 0.1))[:, None])
    date = days[t + h - 1]
    wk = int(date.isocalendar().week)
    feat.append(
        np.tile(
            [
                np.sin(2 * np.pi * wk / 52.18),
                np.cos(2 * np.pi * wk / 52.18),
                float(date.month),
                float(t),
            ],
            (n, 1),
        )
    )
    old = np.concatenate(feat, axis=1).astype("float32")
    tail = np.asarray(hist[t - 52 : t], dtype=np.float64)
    exps = []
    for span in (4, 8, 13, 26):
        alpha = 2.0 / (span + 1.0)
        weights = np.power(1.0 - alpha, np.arange(51, -1, -1, dtype=float))
        weights /= weights.sum()
        exps.append(weights @ tail)
    m4 = tail[-4:].mean(0)
    m13 = tail[-13:].mean(0)
    m26 = tail[-26:].mean(0)
    sd4 = tail[-4:].std(0)
    sd13 = tail[-13:].std(0)
    extra = np.column_stack(
        [
            *exps,
            m4 - m13,
            m13 - m26,
            (m4 - m26) / (2.0 + m26),
            (exps[0] - exps[2]) / (1.0 + exps[2]),
            sd4 / (1.0 + m4),
            sd13 / (1.0 + m13),
        ]
    ).astype("float32")
    augmented = np.column_stack([old, extra]).astype("float32")
    return augmented


def make_features_stockout(hist, days, static, t, h, instock_history, known_until=None):
    ewma = make_features_ewma(hist, days, static, t, h)
    if known_until is None:
        known_until = t
    effective = min(int(t), int(known_until))
    mask = np.asarray(instock_history)
    if (
        mask.ndim != 2
        or mask.shape[1] != ewma.shape[0]
        or effective < 52
        or effective > len(mask)
    ):
        return ewma
    past = mask[:effective].astype(bool)
    recent = past[-52:]
    out = ~recent
    rates = [out[-w:].mean(axis=0) for w in (4, 13, 26, 52)]
    n = recent.shape[1]
    streak = np.zeros(n, dtype=np.float64)
    contiguous = np.ones(n, dtype=bool)
    for row in out[::-1]:
        contiguous &= row
        streak += contiguous.astype(float)
    idx = np.arange(52, dtype=np.int32)[:, None]
    latest_in = np.where(recent, idx, -1).max(axis=0)
    latest_out = np.where(out, idx, -1).max(axis=0)
    since_in = np.where(latest_in >= 0, 51 - latest_in, 52).astype(float)
    since_out = np.where(latest_out >= 0, 51 - latest_out, 52).astype(float)
    extras = np.column_stack(
        [
            *rates,
            streak,
            since_in,
            since_out,
            out[-1].astype(float),
            out[-4].astype(float),
            out[-13].astype(float),
            np.full(n, t - effective, dtype=float),
        ]
    ).astype("float32")
    result = np.column_stack([ewma, extras]).astype("float32")
    if result.shape[1] != ewma.shape[1] + 11:
        return ewma
    return result


def fit_direct():
    global MODELS
    cutoff = BASE_Y.shape[0]
    data = BASE_Y
    stock = BASE_STOCK
    days = DAYS
    static = STATIC
    hist = imputed(data, stock, cutoff)
    models = []
    for h in (1, 2, 3):
        XX, YY, WW = [], [], []
        for t in range(55, cutoff - h + 1):
            target = t + h - 1
            X = make_features_stockout(hist, days, static, t, h, stock, known_until=t)
            XX.append(X)
            YY.append(hist[target])
            WW.append(
                np.where(stock[target], 1.0, 0.2) * (0.994 ** (cutoff - 1 - target))
            )
        X = np.concatenate(XX)
        target_arr = np.concatenate(YY)
        weight = np.concatenate(WW)
        model = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=120,
            learning_rate=0.045,
            num_leaves=23,
            min_child_samples=120,
            reg_lambda=8,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            max_depth=-1,
            verbosity=-1,
            n_jobs=2,
            random_state=20261008 + h,
        )
        model.fit(
            X,
            target_arr,
            sample_weight=weight,
            categorical_feature=list(range(static.shape[1])),
        )
        models.append(model)
    MODELS = models


def _load_cache():
    global MODELS
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                MODELS = pickle.load(f)
            return True
        except Exception:
            return False
    return False


def _compute_corr():
    global CORR
    CORR = np.ones(BASE_N, dtype=float)


def _ensure_models():
    global MODELS
    if MODELS is None and HAS_DATA:
        if not _load_cache():
            fit_direct()
        _compute_corr()


def order_for(fore, state, params):
    scale = params["scale"]
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) * scale for f in fore]
    on, a1, a2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on + a1 + a2 - params["projection"] * (a + b))
    buffer = params["buffer"] * np.sqrt(c)
    deficit = c + buffer - projected
    pos = a + b + c + buffer - on - a1 - a2
    kind = params["policy"]
    if kind == "deficit":
        order = deficit
    elif kind == "position":
        order = pos
    elif kind == "hybrid":
        order = params["mix"] * deficit + (1.0 - params["mix"]) * pos
    elif kind == "mean":
        order = c - projected
    elif kind == "cap":
        order = np.minimum(deficit, params["cap"] * c)
    elif kind == "reorder":
        order = np.where(projected < params["reorder"] * c, deficit, 0.0)
    else:
        order = deficit
    return np.rint(np.clip(order, 0.0, None))


# Load data at import
_load_data()
if HAS_DATA:
    try:
        _ensure_models()
    except Exception:
        HAS_DATA = False


def decide(week_index, state, revealed):
    state = np.asarray(state, dtype=float)
    n = state.shape[0]
    if not HAS_DATA or n != BASE_N or MODELS is None:
        wi = int(week_index)
        if wi > 0 and revealed is not None and np.asarray(revealed).size > 0:
            r = np.asarray(revealed, dtype=float).reshape(wi, n)
            fc = r.mean(axis=0)
        else:
            fc = np.zeros(n, dtype=float)
        on, a1, a2 = state.T
        projected = np.maximum(0.0, on + a1 + a2)
        order = np.maximum(0.0, fc * 3.0 - projected)
        return np.rint(order).astype(float)

    wi = int(week_index)
    revealed = np.asarray(revealed, dtype=float)
    if revealed.size == 0:
        rev = np.zeros((0, n), dtype=float)
    else:
        rev = revealed.reshape(wi, n) if revealed.ndim == 2 else revealed.reshape(wi, n)

    base = BASE_Y.copy()
    stock_hist = BASE_STOCK.copy().astype(bool)
    if wi > 0:
        r = rev[-wi:] if rev.shape[0] >= wi else rev
        r = r.reshape(wi, n)
        base = np.vstack([base, r])
        stock_hist = np.vstack([stock_hist, np.ones((wi, n), dtype=bool)])

    cutoff = 157 + wi
    hist = imputed(base, stock_hist, cutoff)
    end = cutoff

    fore = []
    for i, h in enumerate((1, 2, 3)):
        X = make_features_stockout(
            hist, DAYS, STATIC, end, h, stock_hist, known_until=end
        )
        pred = MODELS[i].predict(X)
        pred = np.maximum(0.0, pred)
        if CORR is not None:
            pred = pred * CORR
        fore.append(pred)

    # adaptive correction from revealed feedback
    strength = float(PARAMS.get("adapt_strength", 0.0))
    if strength > 0 and wi > 0 and rev.size > 0:
        try:
            pred_sum = 0.0
            for k in range(wi):
                origin = 157 + k
                # history up to origin (base + first k+1 revealed)
                base_k = np.vstack([BASE_Y, rev[:k+1]]) if k+1 <= rev.shape[0] else BASE_Y
                # trim to origin+1 weeks
                base_k = base_k[:origin+1]
                stock_k = np.vstack([BASE_STOCK, np.ones((k+1, n), dtype=bool)])[:origin+1]
                hist_k = imputed(base_k, stock_k, origin+1)
                Xk = make_features_stockout(hist_k, DAYS, STATIC, origin, 1, stock_k, known_until=origin)
                pk = MODELS[0].predict(Xk)
                pred_sum += float(np.sum(np.maximum(0.0, pk)))
            actual_sum = float(np.sum(rev[:wi]))
            if pred_sum > 0:
                factor = actual_sum / pred_sum
                factor = float(np.clip(factor, 0.7, 1.3))
                adj = 1.0 + strength * (factor - 1.0)
                fore = [f * adj for f in fore]
        except Exception:
            pass

    order = order_for(fore, state, PARAMS)
    return np.asarray(order, dtype=float).ravel()
