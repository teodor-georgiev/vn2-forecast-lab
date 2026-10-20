"""Core: data loading, forecasting, order planning, backtesting."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ARENA = Path(r"C:\Users\Toni\Documents\Default Project\vn2-model-arena")
sys.path.insert(0, str(ARENA / "tools"))
import inventory_sim as sim  # noqa: E402


def load():
    sales = pd.read_csv(ARENA / "data" / "sales.csv")
    sales["week"] = pd.to_datetime(sales["week"])
    piv = sales.pivot(index="series_id", columns="week", values="y").sort_index()
    H = piv.to_numpy(float).T  # (157, n) time-major
    series = piv.index.to_numpy()
    init = (
        pd.read_csv(ARENA / "data" / "initial_state.csv")
        .set_index("series_id")
        .loc[series]
    )
    I = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
    return H, series, I


# ---------------------------------------------------------------- forecasts
def fc_mean13(H, T, **kw):
    return np.repeat(H[max(0, T - 13) : T].mean(0)[None, :], 8, axis=0)


def fc_blend(H, T, alpha=0.5, recent=4, lo=0.4, hi=2.5, **kw):
    """Recent level blended with year-ago seasonal pattern rescaled to current level."""
    hist = H[:T]
    n = H.shape[1]
    mu_rec = hist[-recent:].mean(0)
    cur_base = hist[-13:].mean(0)
    out = np.zeros((8, n))
    for k in range(1, 9):
        idx = T + k - 52  # same week last year
        if idx - 12 >= 0:
            yoy = hist[idx]
            yoy_base = hist[idx - 12 : idx + 1].mean(0)
            ratio = np.clip(cur_base / np.maximum(yoy_base, 0.3), lo, hi)
            seasonal = yoy * ratio
        else:
            seasonal = hist[-13:].mean(0)
        out[k - 1] = alpha * mu_rec + (1 - alpha) * seasonal
    return out


def fc_blend2(H, T, alpha=0.5, recent=4, **kw):
    """Blend of recent level with an average of the last two year-ago patterns."""
    hist = H[:T]
    n = H.shape[1]
    mu_rec = hist[-recent:].mean(0)
    cur_base = hist[-13:].mean(0)
    out = np.zeros((8, n))
    for k in range(1, 9):
        idx = T + k - 52
        if idx - 12 >= 0:
            yoy = hist[idx]
            base = hist[idx - 12 : idx + 1].mean(0)
            ratio = np.clip(cur_base / np.maximum(base, 0.3), 0.4, 2.5)
            seasonal = yoy * ratio
            if idx - 52 - 12 >= 0:
                y2 = hist[idx - 52]
                b2 = hist[idx - 52 - 12 : idx - 52 + 1].mean(0)
                r2 = np.clip(cur_base / np.maximum(b2, 0.3), 0.4, 2.5)
                seasonal = 0.7 * seasonal + 0.3 * (y2 * r2)
        else:
            seasonal = hist[-13:].mean(0)
        out[k - 1] = alpha * mu_rec + (1 - alpha) * seasonal
    return out


FORECASTS = {"mean13": fc_mean13, "blend": fc_blend, "blend2": fc_blend2}


def sigma_from(H, T, n, win=26, floor=0.0):
    hist = H[max(0, T - win) : T]
    return hist.std(0) if hist.shape[0] > 1 else np.zeros(n)


# ---------------------------------------------------------------- planners
def plan_news(F, sig, I, z=1.0):
    """Order so that the supply available in the arrival week hits mu + z*sigma.

    Forward pass over the *expected* demand path (no feedback), each order covers
    the week it lands in; leftover projections are netted off.
    """
    n = F.shape[1]
    on_hand, arr1, arr2 = I.T
    Ew = np.zeros((9, n))  # Ew[k] = inventory at start of week k+1
    Ew[0] = on_hand
    arr = {1: np.array(arr1, float), 2: np.array(arr2, float)}
    Q = np.zeros((6, n))
    for t in range(1, 9):
        supply = Ew[t - 1] + arr.get(t, np.zeros(n))
        Ew[t] = np.maximum(0.0, supply - F[t - 1])
        if t <= 6:
            target_week = t + 2
            supply_next = Ew[t] + arr.get(t + 1, np.zeros(n))
            proj = np.maximum(0.0, supply_next - F[t])  # start of week t+2
            target = F[target_week - 1] + z * sig[target_week - 1]
            Q[t - 1] = np.maximum(0.0, target - proj)
            arr[target_week] = arr.get(target_week, np.zeros(n)) + Q[t - 1]
    return Q


def plan_ref(H, T, I, mu):
    """Reference static baseline (13-week mean + reference deficit rule)."""
    n = I.shape[0]
    orders = np.zeros((6, n))
    state = I.copy()
    for w in range(6):
        fc = np.repeat(mu[None, :], 3, axis=0)
        order = sim.reference_orders(fc, state, 1.0, 1.0, 0.75)
        orders[w] = order
        on_hand, arr1, arr2 = state.T
        end = np.maximum(0.0, on_hand + arr1 - mu)
        state = np.column_stack((end, arr2, order))
    return orders


def plan_cover(F, sig, I, z=1.0, cover=1.0):
    """Like plan_news but the target covers `cover` weeks of the arrival week."""
    n = F.shape[1]
    on_hand, arr1, arr2 = I.T
    Ew = np.zeros((9, n))
    Ew[0] = on_hand
    arr = {1: np.array(arr1, float), 2: np.array(arr2, float)}
    Q = np.zeros((6, n))
    for t in range(1, 9):
        supply = Ew[t - 1] + arr.get(t, np.zeros(n))
        Ew[t] = np.maximum(0.0, supply - F[t - 1])
        if t <= 6:
            tw = t + 2
            supply_next = Ew[t] + arr.get(t + 1, np.zeros(n))
            proj = np.maximum(0.0, supply_next - F[t])
            mu = F[tw - 1]
            sg = sig[tw - 1]
            if cover == 1.0:
                target = mu + z * sg
            else:
                idx = list(range(tw - 1, min(8, tw - 1 + int(np.ceil(cover)))))
                mu = F[idx].sum(0)
                sg = np.sqrt((sig[idx] ** 2).sum(0))
                target = mu + z * sg
            Q[t - 1] = np.maximum(0.0, target - proj)
            arr[tw] = arr.get(tw, np.zeros(n)) + Q[t - 1]
    return Q
