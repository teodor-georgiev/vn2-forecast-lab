"""Backtest policy variants on historical windows, then write final orders."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import core
from core import (
    load,
    fc_mean13,
    fc_blend,
    fc_blend2,
    plan_news,
    plan_ref,
    plan_cover,
    sigma_from,
)
import inventory_sim as sim

RUN = Path(
    r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs\minimax-m3.1-flash"
)
H, series, I = load()
T_HIST, N = H.shape
print("history", H.shape)

# --- how big is the real starting position relative to each series' weekly demand?
mu_last = H[-13:].mean(0)
on_ratio = I[:, 0] / np.maximum(mu_last, 0.05)
ar_ratio = I[:, 2] / np.maximum(mu_last, 0.05)
print(
    "on_hand / weekly-mean: median=%.3f mean=%.3f  | arrival_w2: median=%.3f mean=%.3f"
    % (np.median(on_ratio), on_ratio.mean(), np.median(ar_ratio), ar_ratio.mean())
)
R_ON, R_AR = 1.45, 1.10


def proxy_state(mu):
    return np.column_stack([np.rint(R_ON * mu), np.zeros(len(mu)), np.rint(R_AR * mu)])


# ---------------------------------------------------------------- backtest
WINDOWS = list(range(60, 150, 6)) + [104, 52]
WINDOWS = sorted(set(w for w in WINDOWS if w + 8 <= T_HIST))


def run_backtest(fcname, fc_kw, planner, planner_kw, windows=WINDOWS, sig_win=26):
    tot = 0.0
    per = []
    for T in windows:
        fc = FORECASTS[fcname](H, T, **fc_kw)
        sig = np.repeat(sigma_from(H, T, N, sig_win)[None, :], 8, axis=0)
        I0 = proxy_state(H[max(0, T - 13) : T].mean(0))
        Q = planner(fc, sig, I0, **planner_kw)
        res = sim.simulate(H[T : T + 8], I0, Q)
        tot += res["score"]
        per.append((T, res["score"]))
    return tot, per


FORECASTS = {"mean13": fc_mean13, "blend": fc_blend, "blend2": fc_blend2}

# baseline: reference static 13w-mean deficit rule
base = 0.0
for T in WINDOWS:
    mu = H[max(0, T - 13) : T].mean(0)
    I0 = proxy_state(H[max(0, T - 13) : T].mean(0))
    Q = plan_ref(H, T, I0, mu)
    base += sim.simulate(H[T : T + 8], I0, Q)["score"]
print("BASELINE ref_static_13wmean  total=%.1f" % base)

rows = []
for fcname, fckw in [
    ("mean13", {}),
    ("blend", dict(alpha=0.5, recent=4)),
    ("blend2", dict(alpha=0.5, recent=4)),
    ("blend", dict(alpha=0.35, recent=4)),
    ("blend", dict(alpha=0.65, recent=4)),
    ("blend", dict(alpha=0.5, recent=8)),
]:
    for z in [0.0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]:
        tot, per = run_backtest(fcname, fckw, plan_news, dict(z=z))
        rows.append((f"{fcname}{fckw}", f"news z={z}", tot))
        print("%-40s z=%.1f  total=%.1f" % (fcname + str(fckw), z, tot))

res_df = pd.DataFrame(rows, columns=["fc", "policy", "score"]).sort_values("score")
print(res_df.head(15).to_string())
res_df.to_csv(RUN / "backtest_grid.csv", index=False)
