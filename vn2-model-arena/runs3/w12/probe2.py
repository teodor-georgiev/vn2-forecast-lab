"""Probe 2: impute true demand for censored weeks and re-score the reference rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt  # noqa: E402

DATA = HERE.parents[1] / "data"


def load_all():
    sales = pd.read_csv(DATA / "sales.csv")
    avail = pd.read_csv(DATA / "availability.csv")
    ids = sorted(sales["series_id"].unique().tolist())
    weeks = sorted(sales["week"].unique().tolist())
    Y = (
        sales.pivot(index="series_id", columns="week", values="y")
        .reindex(index=ids, columns=weeks)
        .values.astype(float)
    )
    A = (
        avail.pivot(index="series_id", columns="week", values="is_available")
        .reindex(index=ids, columns=weeks)
        .values.astype(float)
    )
    init = pd.read_csv(DATA / "initial_state.csv").sort_values("series_id")
    I = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
    return Y, A, I


def true_demand(Y, A):
    """Impute demand in censored weeks: seasonal expectation from uncensored weeks."""
    n, T = Y.shape
    mask = A > 0.5
    cnt = mask.sum(1)
    ssum = (Y * mask).sum(1)
    mu = np.where(cnt > 0, ssum / np.maximum(cnt, 1), Y.mean(1))
    kk = np.arange(T) % 52
    num = np.zeros((n, 52))
    den = np.zeros((n, 52))
    c2 = np.zeros((n, 52))
    for j in range(52):
        m = kk == j
        num[:, j] = (Y[:, m] * mask[:, m]).sum(1)
        den[:, j] = (mask[:, m] * mu[:, None]).sum(1)
        c2[:, j] = mask[:, m].sum(1)
    glob = np.zeros(52)
    gden = np.zeros(52)
    for j in range(52):
        m = kk == j
        glob[j] = (Y[:, m] * mask[:, m]).sum()
        gden[j] = (mask[:, m] * mu[:, None]).sum()
    glob = np.where(gden > 0, glob / np.maximum(gden, 1e-9), 1.0)
    glob = glob / glob[glob > 0].mean()
    seas = (c2 * num / np.maximum(den, 1e-9) + 4.0 * glob) / np.maximum(c2 + 4.0, 1e-9)
    seas = seas / seas.mean(1, keepdims=True)
    s_i = seas[:, kk]
    imp = np.where(mask, Y, np.maximum(Y, mu[:, None] * s_i))
    return imp, seas


def ref_factory(Y, window=13, buffer=0.75):
    T = Y.shape[1]

    def ref(week_index, state, revealed):
        n = state.shape[0]
        t = T + week_index
        hist = Y[:, max(0, t - window) : t]
        m = hist.mean(axis=1) if hist.shape[1] else np.zeros(n)
        f = np.stack([m, m, m], axis=0)
        a, b, c = [np.maximum(0.0, f[i]) for i in range(3)]
        on_hand, arr1, arr2 = state[:, 0], state[:, 1], state[:, 2]
        projected = np.maximum(0.0, on_hand + arr1 + arr2 - (a + b))
        deficit = c + buffer * np.sqrt(c) - projected
        return np.rint(np.clip(deficit, 0.0, None))

    return ref


def main():
    Y, A, I = load_all()
    D, seas = true_demand(Y, A)
    print("censored weeks in history:", int((A < 0.5).sum()))
    print("last 8 weeks (149..156) censored share: %.4f" % (1 - A[:, 149:157].mean()))
    for c in [149, 141, 133, 125, 117, 109, 101, 93, 85]:
        r_cens = bt.backtest(ref_factory(Y), Y, start_week=c, initial=I)
        r_true = bt.backtest(ref_factory(Y), D, start_week=c, initial=I)
        print(
            f"cutoff {c:4d}  censored-panel {r_cens['score']:8.1f}   true-demand-panel {r_true['score']:8.1f}"
            f"   demand8 {Y[:, c : c + 8].sum():7.0f} / {D[:, c : c + 8].sum():7.0f}"
        )


if __name__ == "__main__":
    main()
