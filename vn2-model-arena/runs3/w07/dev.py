"""Dev harness: honest backtests of policy.decide at several historical cutoffs."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARENA = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ARENA / "tools"))
import hb_fixed as hb  # noqa: E402
import inventory_sim as sim  # noqa: E402
import policy  # noqa: E402

import pandas as pd  # noqa: E402

DATA = ARENA / "data"


def load():
    sales = pd.read_csv(DATA / "sales.csv")
    avail = pd.read_csv(DATA / "availability.csv")
    s = sales.pivot(index="series_id", columns="week", values="y")
    a = avail.pivot(index="series_id", columns="week", values="is_available")
    a = a.reindex(index=s.index, columns=s.columns)
    init = pd.read_csv(DATA / "initial_state.csv").set_index("series_id")
    init = init.reindex(s.index)
    y = s.to_numpy(dtype=float)
    av = a.to_numpy(dtype=float)
    st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(dtype=float)
    return y, av, st0


Y, AV, ST0 = load()
N = Y.shape[0]
print("history", Y.shape, "init", ST0.shape)


def run(policy_mod, cutoff, initial=ST0, prior_cols=None, window=26, verbose=False):
    m, v = policy_mod.build_prior(Y, AV, cols=slice(0, cutoff), window=window)
    policy_mod._PRIOR_M, policy_mod._PRIOR_V = m, v
    t0 = time.perf_counter()
    res = hb.backtest(policy_mod.decide, Y, start_week=cutoff, initial=initial)
    dt = time.perf_counter() - t0
    if verbose:
        print(f"  cutoff {cutoff}: score {res['score']:.0f}  ({dt:.1f}s)")
    return res["score"], dt


def anchor(cutoff, initial=ST0, k=13):
    """13-week-mean forecast + reference deficit rule (the 3,983-ish bar)."""
    m = Y[:, max(0, cutoff - k) : cutoff].mean(axis=1)

    def fc(w, state, hist):
        return np.tile(m, (3, 1))

    t0 = time.perf_counter()
    res = sim.order_plan_from_policy(
        Y[:, cutoff : cutoff + 8].T, initial, fc, policy=sim.reference_orders
    )
    return res["score"], time.perf_counter() - t0


if __name__ == "__main__":
    cuts = [int(c) for c in sys.argv[1:]] or [100, 130]
    for c in cuts:
        sc, dt = anchor(c)
        print(f"anchor cutoff {c}: {sc:.0f} ({dt:.1f}s)")
        sc, dt = run(policy, c, verbose=False)
        print(f"policy cutoff {c}: {sc:.0f} ({dt:.1f}s)")
