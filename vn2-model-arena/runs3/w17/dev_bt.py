"""Dev: backtest policy.py with honest, truncated history (no lookahead)."""

from __future__ import annotations

import time
import importlib
import numpy as np
import pandas as pd
from pathlib import Path

import hb_local as hb  # manager's helper with the missing revealed.append restored

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data"

_s = pd.read_csv(DATA / "sales.csv")
_piv = _s.pivot(index="series_id", columns="week", values="y").sort_index()
S = _piv.to_numpy(dtype=float)
_a = pd.read_csv(DATA / "availability.csv")
_ap = _a.pivot(index="series_id", columns="week", values="is_available").sort_index()
_ap = _ap.reindex(index=_piv.index, columns=_piv.columns)
A = np.where(np.isnan(_ap.to_numpy(dtype=float)), 1.0, _ap.to_numpy(dtype=float)) > 0.5
ATTRS = (
    pd.read_csv(DATA / "product_attributes.csv")
    .set_index("series_id")
    .reindex(_piv.index)
)
INIT = (
    pd.read_csv(DATA / "initial_state.csv").set_index("series_id").reindex(_piv.index)
)
INIT = INIT[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)

N, H = S.shape


def truncate(h):
    """Point policy.py's data loader at history[:h] only (honest replay)."""
    import policy

    def loader():
        return S[:, :h].copy(), A[:, :h].copy(), ATTRS

    policy._load_raw = loader
    policy._PROFILE_CACHE.clear()


def truncate_restore():
    import policy

    importlib.reload(policy)


def ref_decide_factory(h, window=13):
    """Honest 13-week-mean reference rule (deficit + 0.75*sqrt buffer)."""
    base = S[:, max(0, h - window) : h].mean(axis=1)

    def decide(week_index, state, revealed):
        f = np.tile(base * 2.6 / 2.6, (3, 1))  # constant level forecast
        return _ref(f, state)

    return decide


def _ref(forecast, state):
    a, b, c = [np.maximum(0.0, np.asarray(f, dtype=float)) for f in forecast]
    on_hand, arr1, arr2 = np.asarray(state, dtype=float).T
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - (a + b))
    deficit = c + 0.75 * np.sqrt(c) - projected
    return np.rint(np.clip(deficit, 0.0, None))


def run(cutoffs, initial=None, verbose=True, label=""):
    import policy

    scores = {}
    t0 = time.perf_counter()
    for h in cutoffs:
        truncate(h)
        # honest reference bar on the same truncated history
        ref = hb.backtest(ref_decide_factory(h), S, h, initial=initial)["score"]
        res = hb.backtest(policy.decide, S, h, initial=initial)
        scores[h] = res["score"]
        if verbose:
            print(
                f"  cutoff {h:3d}: policy {res['score']:9.1f}   ref13 {ref:9.1f}   "
                f"ratio {res['score'] / max(ref, 1e-9):5.3f}"
            )
    el = time.perf_counter() - t0
    if verbose:
        mean = float(np.mean(list(scores.values())))
        print(f"  {label} mean {mean:9.1f}   ({el:.2f}s)")
    return scores


if __name__ == "__main__":
    import sys

    cuts = [int(x) for x in sys.argv[1:]] or [60, 100, 130, 145]
    print("policy backtest (honest truncated history), initial=zeros")
    run(cuts)
    print("same, initial=arena initial_state.csv")
    run(cuts, initial=INIT)
