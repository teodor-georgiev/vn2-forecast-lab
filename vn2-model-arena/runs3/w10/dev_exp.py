"""Experiment: try different safety-stock calibrations (not a deliverable)."""

import sys, time
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
corrected = raw.copy()
for i in range(raw.shape[0]):
    vals = raw[i][av[i]]
    fill = float(vals.mean()) if vals.size else 0.0
    corrected[i] = np.where(av[i], raw[i], fill)
initial = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)

n = policy._n_series
level_end = policy._level_hist[:, -1]
trend_end = policy._trend_hist[:, -1]
f3 = np.maximum(0.0, level_end + 2.0 * trend_end)


def replay(decide, history, start_week, initial=None):
    history = np.asarray(history, dtype=float)
    n = history.shape[0]
    state = initial.copy() if initial is not None else np.zeros((n, 3))
    demand = history[:, start_week : start_week + 8].T
    revealed = []
    weekly = []
    t0 = time.perf_counter()
    for w in range(8):
        if w < 6:
            rev = (
                np.array(revealed, dtype=float).reshape(w, n) if w else np.empty((0, n))
            )
            order = np.rint(
                np.maximum(0.0, np.asarray(decide(w, state.copy(), rev), dtype=float))
            )
        else:
            order = np.zeros(n)
        on_hand, arr1, arr2 = state.T.copy()
        supply = on_hand + arr1
        end = np.maximum(0.0, supply - demand[w])
        missed = np.maximum(0.0, demand[w] - supply)
        weekly.append(
            {
                "week": w + 1,
                "holding": float(0.2 * end.sum()),
                "shortage": float(1.0 * missed.sum()),
                "ordered": int(order.sum()),
            }
        )
        state = np.column_stack((end, arr2, order))
        revealed.append(demand[w])
    score = sum(r["holding"] + r["shortage"] for r in weekly[2:])
    return score, time.perf_counter() - t0


def set_safety(safety):
    safety = np.asarray(safety, dtype=float)
    if safety.ndim == 1:
        safety = np.repeat(safety[:, None], 3, axis=1)
    policy._safety = safety


def make_decide():
    import importlib
    import policy as p

    return p.decide


for sw in (100, 130):
    print(f"=== cutoff {sw} ===")
    base_q = np.quantile(corrected[:, 3:], policy.CRIT, axis=1)
    quant_safety = np.maximum(0.0, base_q - f3)
    set_safety(quant_safety)
    s, el = replay(policy.decide, corrected, sw, initial)
    print(f"  demand-quantile safety: score={s:.1f} ({el:.2f}s)")
    for k in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        set_safety(k * np.sqrt(np.maximum(f3, 1.0)))
        s, el = replay(policy.decide, corrected, sw, initial)
        print(f"  k*sqrt(f3) k={k}: score={s:.1f}")
    # capped quantile
    cap = 1.5 * np.sqrt(np.maximum(f3, 1.0))
    set_safety(np.minimum(quant_safety, cap))
    s, el = replay(policy.decide, corrected, sw, initial)
    print(f"  quantile capped 1.5*sqrt: score={s:.1f}")
