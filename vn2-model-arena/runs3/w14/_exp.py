"""Experiment: per-horizon direct quantile regression -> order-up-to policy."""

import os, sys, time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _prep, _bt

SIDS, Y, A, INIT = _prep.load_raw()
N, T = Y.shape


def train_q(Dtr, Atr, anchor, qs, span=3, n_est=180, leaves=15):
    origins = np.arange(25, anchor - span)
    Xf = _prep.build_features(Dtr, Atr, origins)
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(Dtr, axis=1)], axis=1)
    ysum = np.stack([cs[:, t + 1 + span] - cs[:, t + 1] for t in origins])
    X = Xf.reshape(-1, Xf.shape[2])
    y = ysum.reshape(-1)
    models = {}
    for q in qs:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=n_est,
            learning_rate=0.06,
            num_leaves=leaves,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            n_jobs=2,
            verbose=-1,
            random_state=7,
        )
        m.fit(X, y)
        models[q] = m
    return models, X.shape


def make_decide(models, Dtr, Atr, qfun):
    def decide(week_index, state, revealed):
        r = np.asarray(revealed, dtype=float)
        if r.size:
            Dk = np.concatenate([Dtr, r.T], axis=1)
            Ak = np.concatenate([Atr, np.ones((N, r.shape[0]))], axis=1)
        else:
            Dk, Ak = Dtr, Atr
        t = Dk.shape[1] - 1
        Xf = _prep.build_features(Dk, Ak, [t])
        pos = np.asarray(state, dtype=float).sum(axis=1)
        q = qfun(week_index)
        S = np.maximum(0.0, models[q].predict(Xf[0]))
        return np.maximum(0.0, S - pos)

    return decide


def run(
    anchor,
    q,
    qs=(0.5, 0.65, 0.8, 0.9),
    init=None,
    span=3,
    n_est=180,
    quiet=False,
    qfun=None,
):
    Ytr, Atr = Y[:, :anchor], A[:, :anchor]
    Dtr = _prep.impute(Ytr, Atr)
    t0 = time.time()
    models, shp = train_q(Dtr, Atr, anchor, qs, span=span, n_est=n_est)
    tt = time.time() - t0
    qf = qfun if qfun else (lambda w: q)
    dec = make_decide(models, Dtr, Atr, qf)
    res = _bt.backtest(dec, Y, start_week=anchor, initial=init)
    if not quiet:
        print(
            f"anchor={anchor} q={q} train={tt:.1f}s rows={shp[0]} score={res['score']:.1f} ord={res['orders'].sum():.0f}"
        )
    return res, models, tt


if __name__ == "__main__":
    for anchor in [100, 130]:
        for q in [0.5, 0.65, 0.8, 0.9, 0.95]:
            run(anchor, q, init=INIT)
