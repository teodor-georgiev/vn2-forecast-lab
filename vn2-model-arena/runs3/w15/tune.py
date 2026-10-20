"""Rolling weekly backtests: re-tune a small deficit-rule grid on recent windows."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

import mybt as hb

DATA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\data"

Y = np.load("Y.npy")  # (599, 157) sorted series order
A = np.load("A.npy")
init = pd.read_csv(f"{DATA}\\initial_state.csv")
print("init sorted ids match:", list(init["series_id"]) == sorted(init["series_id"]))
INIT = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
assert INIT.shape == (Y.shape[0], 3), INIT.shape

N, T = Y.shape
WIN = 13
H = Y.shape[1]


def censor_correct(Y, A, half=3):
    """Impute stockout weeks (avail==0 -> y==0) with local uncensored level."""
    valid = A > 0
    V = Y * valid
    S = np.zeros((Y.shape[0], Y.shape[1] + 1))
    C = np.zeros_like(S)
    S[:, 1:] = np.cumsum(V, axis=1)
    C[:, 1:] = np.cumsum(valid, axis=1)
    T_ = Y.shape[1]
    t = np.arange(T_)
    a = np.clip(t - half, 0, T_)
    b = np.clip(t + half, 0, T_)
    cnt = C[:, b] - C[:, a]
    lvl = (S[:, b] - S[:, a]) / np.maximum(cnt, 1.0)
    return np.where(valid, Y, lvl)


YC = censor_correct(Y, A)


def ref_orders(f, state, scale, buffer, projection):
    a, b, c = [np.maximum(0.0, np.asarray(x, dtype=float)) * scale for x in f]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - projection * (a + b))
    deficit = c + buffer * np.sqrt(np.maximum(c, 0.0)) - projected
    return np.rint(np.clip(deficit, 0.0, None))


def make_decide(hist, cfg, level_win=WIN):
    """hist: (N, L0) known history before the game (backtest cutoff)."""
    N_ = hist.shape[0]
    L0 = hist.shape[1]
    scale, buffer, projection = cfg["scale"], cfg["buffer"], cfg["projection"]
    base = cfg["base"]  # 'raw' | 'corr'
    adapt = cfg.get("adapt", 0.0)

    def decide(w, state, revealed):
        if revealed is None:
            rev = np.zeros((N_, 0))
        else:
            rev = np.asarray(revealed, dtype=float).reshape(w, N_).T
        avail = np.concatenate([hist, rev], axis=1) if w else hist
        level_now = np.maximum(avail[:, -level_win:].mean(axis=1), 0.0)
        f = [level_now, level_now, level_now]
        if adapt > 0:
            # in-game level multiplier from revealed demand vs. trailing forecasts
            if w > 0:
                pred = np.zeros(N_)
                for j in range(w):
                    pred += avail[:, L0 + j - level_win : L0 + j].mean(axis=1)
                obs = rev.sum(axis=1)
                kappa = adapt * level_now
                mult = np.clip((obs + kappa) / np.maximum(pred + kappa, 1e-9), 0.4, 2.5)
                mult = np.where(np.isfinite(mult), mult, 1.0)
            else:
                mult = 1.0
            f = [x * mult for x in f]
        return ref_orders(f, state, scale, buffer, projection)

    return decide


def run_grid(
    windows,
    base_variants,
    scales,
    buffers,
    projections,
    adapts,
    history_full,
    initial,
    verbose=True,
):
    rows = []
    for base in base_variants:
        Hmat = YC if base == "corr" else Y
        for scale, buffer, projection, adapt in itertools.product(
            scales, buffers, projections, adapts
        ):
            cfg = dict(
                scale=scale,
                buffer=buffer,
                projection=projection,
                base=base,
                adapt=adapt,
            )
            scores = []
            for s in windows:
                decide = make_decide(Hmat[:, :s], cfg)
                res = hb.backtest(decide, history_full, start_week=s, initial=initial)
                scores.append(res["score"])
            rows.append((float(np.mean(scores)), cfg, scores))
    rows.sort(key=lambda r: r[0])
    if verbose:
        for mean_s, cfg, scores in rows[:12]:
            print(f"{mean_s:9.1f}  {cfg}  {[round(x, 1) for x in scores]}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    windows = [100, 115, 130, 145]
    print(
        "=== calibration: reference rule, raw 13w mean, scale=1 buffer=.75 proj=1 ==="
    )
    ref = dict(scale=1.0, buffer=0.75, projection=1.0, base="raw", adapt=0.0)
    for s in windows:
        r = hb.backtest(make_decide(Y[:, :s], ref), Y, start_week=s, initial=INIT)
        print("  start", s, round(r["score"], 1))
    print("=== same with zeros initial ===")
    for s in windows:
        r = hb.backtest(make_decide(Y[:, :s], ref), Y, start_week=s, initial=None)
        print("  start", s, round(r["score"], 1))

    print("=== grid (init=initial_state) ===")
    rows = run_grid(
        windows=[100, 115, 130, 145],
        base_variants=["raw", "corr"],
        scales=[0.9, 1.0, 1.1, 1.2, 1.35],
        buffers=[0.0, 0.5, 0.75, 1.1],
        projections=[0.85, 1.0, 1.15],
        adapts=[0.0, 6.0],
        history_full=Y,
        initial=INIT,
    )
    print("elapsed", round(time.time() - t0, 1), "s")
