"""Tuning harness: backtest policy.py at several historical cutoffs."""

import importlib.util, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_policy():
    spec = importlib.util.spec_from_file_location("agent_policy", HERE / "policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_history():
    import pandas as pd

    root = HERE.parents[1]
    sales = pd.read_csv(root / "data" / "sales.csv", parse_dates=["week"])
    Y = sales.pivot(index="series_id", columns="week", values="y").sort_index()
    return Y.values.astype(float)


def run(mod, Y, cutoffs):
    import hb2 as hb

    res = {}
    for sw in cutoffs:
        mod._HISTORY_BASE = sw
        t0 = time.perf_counter()
        r = hb.backtest(mod.decide, Y, start_week=sw)
        res[sw] = (r["score"], time.perf_counter() - t0)
    mod._HISTORY_BASE = None
    return res


if __name__ == "__main__":
    Y = get_history()
    mod = load_policy()
    cut = [int(x) for x in (sys.argv[1:] or ["60", "100", "130", "149"])]
    out = run(mod, Y, cut)
    for k, v in out.items():
        print(f"cutoff {k}: score {v[0]:.1f}  ({v[1]:.1f}s)")
    print("mean:", np.mean([v[0] for v in out.values()]).round(1))
