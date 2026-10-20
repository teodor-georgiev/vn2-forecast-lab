"""Tune policy.py on historical cutoffs (worker w05, scratch script)."""

from __future__ import annotations

import importlib.util
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_local as hb

HERE = Path(__file__).parent
import sys

sys.path.insert(0, str(HERE.parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

D = HERE.parents[1] / "data"
sales = pd.read_csv(D / "sales.csv")
avail = pd.read_csv(D / "availability.csv")
y = sales.pivot_table(
    index="series_id", columns="week", values="y", aggfunc="first"
).sort_index(axis=1)
H = np.nan_to_num(y.values.astype(float), nan=0.0)
rank = {s: i for i, s in enumerate(y.index)}
ids_sales = sales["series_id"].drop_duplicates().tolist()
perm_sales = np.array([rank[s] for s in ids_sales])
H_sales = H[perm_sales]
ini = pd.read_csv(D / "initial_state.csv").set_index("series_id")
INIT = np.nan_to_num(
    ini.reindex(y.index)[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float),
    nan=0.0,
)
INIT_sales = INIT[perm_sales]

CUTOFFS = [100, 130]
INITS = [("zeros", None), ("init", INIT)]


def run_ref():
    """Manager's adaptive bar: 13-week trailing-mean forecast + reference rule."""
    from inventory_sim import reference_orders  # arena tool

    tot = 0.0
    for cut in CUTOFFS:
        for nm, init in INITS:

            def fc(w, state, hist, cut=cut):
                past = H[:, max(0, cut + w - 13) : cut + w]
                f = past.mean(axis=1)
                return np.vstack([f, f, f])

            def decide(w, state, hist, fc=fc):
                return reference_orders(fc(w, state, hist), state)

            r = hb.backtest(decide, H, start_week=cut, initial=init)
            tot += r["score"]
            print(f"    ref13 cut={cut} init={nm}: {r['score']:.1f}")
    return tot


def run_cfg(cfg, record_order_test=False):
    for k, v in cfg.items():
        setattr(P, k, v)
    if "CENS_IMPUTE" in cfg:
        P._MODEL = None
    tot, det = 0.0, []
    for cut in CUTOFFS:
        for nm, init in INITS:
            r = hb.backtest(P.decide, H, start_week=cut, initial=init)
            tot += r["score"]
            det.append(r["score"])
    extra = None
    if record_order_test:
        # same games but with the series presented in sales-appearance order
        P._LOCK = None
        det2, extra = [], 0.0
        for cut in CUTOFFS:
            for nm, init in [("zeros", None), ("init", INIT_sales)]:
                r = hb.backtest(P.decide, H_sales, start_week=cut, initial=init)
                det2.append(r["score"])
                extra += r["score"]
        print(f"    sales-order variant: {[round(x, 1) for x in det2]} sum={extra:.1f}")
    return tot, det, extra


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("== reference bar (13-week mean adaptive rule) ==")
    ref = run_ref()
    print(f"    REF TOTAL = {ref:.1f}")

    print("== stage 1: P_TARGET x P_DEP x SCALE (alpha=0.30, cens=on) ==")
    best = None
    rows = []
    for pt, pd_, sc in itertools.product(
        [0.60, 0.70, 0.80, 0.85, 0.90, 0.95], [0.30, 0.50, 0.80], [1.0, 1.15]
    ):
        cfg = {"P_TARGET": pt, "P_DEP": pd_, "SCALE": sc, "ALPHA": 0.30, "BETA": 0.30}
        tot, det, _ = run_cfg(cfg)
        rows.append((tot, cfg, det))
        print(
            f"  pt={pt:.2f} pd={pd_:.2f} sc={sc:.2f} -> {tot:.1f}  {[round(x) for x in det]}"
        )
    rows.sort(key=lambda r: r[0])
    best = rows[0]
    print("best stage1:", best[0], best[1])

    print("== stage 2: ALPHA/BETA/CENS/SCALE at best stage-1 fractiles ==")
    rows2 = []
    base = dict(best[1])
    for al, be, sc, cens in itertools.product(
        [0.15, 0.30, 0.50], [0.30, 0.50], [1.0, 1.15, 1.3], [True, False]
    ):
        cfg = dict(base, ALPHA=al, BETA=be, SCALE=sc, CENS_IMPUTE=cens)
        tot, det, _ = run_cfg(cfg)
        rows2.append((tot, cfg, det))
    rows2.sort(key=lambda r: r[0])
    for tot, cfg, det in rows2[:8]:
        print(f"  {tot:.1f} {cfg} {[round(x) for x in det]}")

    print("== stage 3: final check + series-order robustness ==")
    fin = dict(rows2[0][1])
    tot, det, extra = run_cfg(fin, record_order_test=True)
    print(
        f"FINAL cfg={fin}\n  sorted-order sum={tot:.1f} det={[round(x, 1) for x in det]}"
    )
    print(f"  vs ref {ref:.1f}  (elapsed {time.perf_counter() - t0:.0f}s)")
