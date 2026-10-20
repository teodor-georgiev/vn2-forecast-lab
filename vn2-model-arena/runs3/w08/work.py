"""Backtest harness for the w08 policy (experiment script, not the deliverable)."""

from __future__ import annotations

import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import helper_backtest as hb  # noqa: E402
import hb_fixed as hb  # noqa: E402  (manager helper + the missing reveal append)

z = np.load(os.path.join(HERE, "cache.npz"), allow_pickle=True)
Y = z["Y"].astype(float)  # (n, T) raw sales (censored truth)
A = z["A"].astype(bool)


def corrected_truth(theta=0.6):
    W = 13
    half = W // 2
    masked = np.where(A, Y, np.nan)
    padded = np.pad(masked, ((0, 0), (half, half)), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(padded, W, axis=1)
    with np.errstate(all="ignore"):
        lq = np.nanquantile(win, 0.75, axis=-1)
    lvl = np.where(A, Y, 0.0).sum(axis=1) / np.maximum(A.sum(axis=1), 1)
    lq = np.where(np.isfinite(lq), lq, lvl[:, None])
    imp = np.where(A, Y, np.maximum(Y, np.maximum(lq, 0.0)))
    return np.where(A, Y, Y + theta * (imp - Y))


D = corrected_truth()


def run(cutoff, truth="raw", start=None, policy_mod="policy", verbose=True):
    """cutoff: number of history weeks the policy may see (0-based)."""
    os.environ["VN2_CUTOFF"] = str(cutoff)
    if policy_mod in sys.modules:
        del sys.modules[policy_mod]
    mod = importlib.import_module(policy_mod)
    t0 = time.perf_counter()
    h = Y if truth == "raw" else D
    starts = [start] if start is not None else [cutoff]
    out = []
    for stw in starts:
        res = hb.backtest(mod.decide, h, stw)
        out.append(res["score"])
    el = time.perf_counter() - t0
    if verbose:
        print(
            f"  cutoff={cutoff:4d} truth={truth:8s} scores={[round(s, 1) for s in out]} "
            f"mean={np.mean(out):9.1f}  ({el:.1f}s)"
        )
    return float(np.mean(out)), out


if __name__ == "__main__":
    import policy  # noqa: F401  (baseline: default config)

    print("== baseline: default config ==")
    for cut in (100, 130):
        for truth in ("raw", "corr"):
            run(cut, truth)
