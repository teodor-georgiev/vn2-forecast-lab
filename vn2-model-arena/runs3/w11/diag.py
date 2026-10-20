"""Diagnose a policy config: cost split, orders, and compare to a simple baseline."""

import sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hb2  # noqa: E402


def get_data():
    import pandas as pd

    root = HERE.parents[1]
    sales = pd.read_csv(root / "data" / "sales.csv", parse_dates=["week"])
    Y = sales.pivot(index="series_id", columns="week", values="y").sort_index()
    init = pd.read_csv(root / "data" / "initial_state.csv")
    return Y.values.astype(float), init[
        ["on_hand", "arrival_w1", "arrival_w2"]
    ].values.astype(float)


def report(name, decide, Y, init, cutoffs):
    print(f"--- {name}")
    for sw in cutoffs:
        r = hb2.backtest(decide, Y, start_week=sw, initial=init)
        wk = r["weekly"][2:]
        hold = sum(x["holding"] for x in wk)
        short = sum(x["shortage"] for x in wk)
        dem = Y[:, sw + 2 : sw + 8].sum()
        print(
            f"  cut{sw}: score {r['score']:8.1f} | hold {hold:8.1f} short {short:8.1f} "
            f"| ordered {r['orders'].sum():8.0f} demand6 {dem:8.0f} ratio {r['orders'].sum() / dem:5.2f}"
        )
        print(
            "      per-week:", [f"{x['holding']:.0f}/{x['shortage']:.0f}" for x in wk]
        )


if __name__ == "__main__":
    Y, init = get_data()
    import importlib.util

    spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
    pol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pol)
    cuts = [100, 130, 149]
    pol._HISTORY_BASE = 149
    report("current", pol.decide, Y, init, cuts)
