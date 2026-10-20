"""Diagnostics: reference vs my policy, with/without revealed blend."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
arena = here.parent.parent
import policy

policy._ensure_loaded()

init = pd.read_csv(arena / "data" / "initial_state.csv")
sales = pd.read_csv(arena / "data" / "sales.csv")
weeks = sorted(sales["week"].astype(str).unique())
raw = (
    sales.pivot(index="series_id", columns="week", values="y")
    .reindex(index=init["series_id"], columns=weeks)
    .fillna(0.0)
    .to_numpy(float)
)
avail = pd.read_csv(arena / "data" / "availability.csv")
av = (
    avail.pivot(index="series_id", columns="week", values="is_available")
    .reindex(index=init["series_id"], columns=weeks)
    .fillna(True)
    .to_numpy(bool)
)
corr = raw.copy()
for i in range(raw.shape[0]):
    v = raw[i][av[i]]
    corr[i] = np.where(av[i], raw[i], float(v.mean()) if v.size else 0.0)
initv = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
n = policy._n_series


def replay(decide_fn, history, start_week, initial=None):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    state = initial.copy() if initial is not None else np.zeros((n, 3))
    demand = history[:, start_week : start_week + 8].T
    revealed = []
    weekly = []
    for w in range(8):
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(
                    0.0, np.asarray(decide_fn(w, state.copy(), rev), dtype=float)
                )
            )
        else:
            order = np.zeros(n)
        oh, a1, a2 = state.T.copy()
        sup = oh + a1
        end = np.maximum(0.0, sup - demand[w])
        miss = np.maximum(0.0, demand[w] - sup)
        weekly.append(
            (end.sum() * 0.2 + miss.sum(), end.sum(), miss.sum(), order.sum())
        )
        state = np.column_stack((end, a2, order))
        revealed.append(demand[w])
    scored = [x for x in weekly[2:]]
    return sum(x[0] for x in scored), weekly


# reference rule
def reference_decide(w, state, revealed):
    level = policy._level_hist[:, -1]
    f = [level, level, level]

    on_hand = np.asarray(state)[:, 0]
    arr1 = np.asarray(state)[:, 1]
    arr2 = np.asarray(state)[:, 2]
    a, b, c = f
    projected = np.maximum(0.0, on_hand + arr1 + arr2 - (a + b))
    deficit = c + 0.75 * np.sqrt(np.maximum(c, 0.0)) - projected
    return np.clip(np.rint(deficit), 0.0, None)


# my policy, no revealed blend: monkeypatch lam=0 by wrapping
orig_decide = policy.decide


def decide_no_reveal(w, state, revealed):
    return orig_decide(w, state, np.empty((0, state.shape[1])))


for sw in (100, 130):
    print(f"=== cutoff {sw} (demand mean {corr[:, sw : sw + 8].mean():.2f}) ===")
    s, _ = replay(reference_decide, corr, sw, initv)
    print(f"  reference (level, buffer .75): {s:.1f}")
    s, _ = replay(decide_no_reveal, corr, sw, initv)
    print(f"  mine no-reveal:              {s:.1f}")
    s, _ = replay(policy.decide, corr, sw, initv)
    print(f"  mine with-reveal:            {s:.1f}")
