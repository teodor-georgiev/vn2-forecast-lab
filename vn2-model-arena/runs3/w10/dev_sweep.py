"""Sweep reveal-blend knobs (not a deliverable)."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
arena = here.parent.parent
import policy
import importlib

policy = importlib.reload(policy)
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


def replay(decide_fn, history, start_week, initial=None):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    state = initial.copy() if initial is not None else np.zeros((n, 3))
    demand = history[:, start_week : start_week + 8].T
    revealed = []
    total = 0.0
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
        total += end.sum() * 0.2 + miss.sum()
        state = np.column_stack((end, a2, order))
        revealed.append(demand[w])
    return total


cases = [
    ("no-reveal", dict(cap=0.0)),
    ("mean cap .25", dict(cap=0.25, mean=True)),
    ("mean cap .5", dict(cap=0.5, mean=True)),
    ("mean cap .75", dict(cap=0.75, mean=True)),
    ("mean cap .9", dict(cap=0.9, mean=True)),
    ("ewma cap .25", dict(cap=0.25)),
    ("ewma cap .5", dict(cap=0.5)),
    ("ewma cap .75", dict(cap=0.75)),
    ("ewma cap .9", dict(cap=0.9)),
    ("ewma cap .9 trend", dict(cap=0.9, trend=True)),
]
for kw in cases:
    tag = kw.pop(0) if False else None
# simpler: iterate dicts below
importlib.reload(policy)
policy._ensure_loaded()
for name, params in [
    ("no-reveal", {"cap": 0.0}),
    ("mean .25", {"cap": 0.25, "mean": True}),
    ("mean .5", {"cap": 0.5, "mean": True}),
    ("mean .75", {"cap": 0.75, "mean": True}),
    ("mean .9", {"cap": 0.9, "mean": True}),
    ("ewma .25", {"cap": 0.25}),
    ("ewma .5", {"cap": 0.5}),
    ("ewma .75", {"cap": 0.75}),
    ("ewma .9", {"cap": 0.9}),
    ("ewma .9 +trend", {"cap": 0.9, "trend": True}),
]:
    policy.MAX_REVEALED_WEIGHT = params["cap"]
    policy.REVEAL_USE_MEAN = params.get("mean", False)
    policy.REVEAL_TREND = params.get("trend", False)
    s1 = replay(policy.decide, corr, 100, initv)
    s2 = replay(policy.decide, corr, 130, initv)
    print(
        f"{name:16s} cap={params['cap']:<4} mean={params.get('mean', False)} trend={params.get('trend', False)}  c100={s1:8.1f} c130={s2:8.1f}"
    )
