"""Stage-4 tuning: push P_TARGET lower, ALPHA x P_DEP x SCALE (w05 scratch)."""

from __future__ import annotations

import importlib.util
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
import sys

sys.path.insert(0, str(HERE))
import backtest_local as hb  # noqa: E402

spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

D = HERE.parents[1] / "data"
sales = pd.read_csv(D / "sales.csv")
y = sales.pivot_table(
    index="series_id", columns="week", values="y", aggfunc="first"
).sort_index(axis=1)
H = np.nan_to_num(y.values.astype(float), nan=0.0)
ini = pd.read_csv(D / "initial_state.csv").set_index("series_id")
INIT = np.nan_to_num(
    ini.reindex(y.index)[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float),
    nan=0.0,
)

CUTOFFS = [100, 130]
INITS = [("zeros", None), ("init", INIT)]


def run_cfg(cfg, detail=False):
    for k, v in cfg.items():
        setattr(P, k, v)
    tot, det, ords = 0.0, [], []
    for cut in CUTOFFS:
        for nm, init in INITS:
            r = hb.backtest(P.decide, H, start_week=cut, initial=init)
            tot += r["score"]
            det.append(round(r["score"]))
            ords.append(int(r["orders"].sum()))
    if detail:
        print(f"    {cfg}\n      {det}  ordered={ords}")
    return tot


if __name__ == "__main__":
    t0 = time.perf_counter()
    rows = []
    for pt, pd_, sc, al in itertools.product(
        [0.30, 0.40, 0.50, 0.60], [0.50, 0.80], [1.0, 1.15], [0.15, 0.30]
    ):
        cfg = {"P_TARGET": pt, "P_DEP": pd_, "SCALE": sc, "ALPHA": al, "BETA": 0.30}
        rows.append((run_cfg(cfg), cfg))
    rows.sort(key=lambda r: r[0])
    for tot, cfg in rows[:10]:
        print(f"{tot:.1f}  {cfg}")
    print("--- detail of best ---")
    run_cfg(dict(rows[0][1]), detail=True)
    print(f"(elapsed {time.perf_counter() - t0:.0f}s)")
