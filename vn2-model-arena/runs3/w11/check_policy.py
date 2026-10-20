"""Score the real policy.py (frozen file) at several cutoffs."""

import importlib.util, sys, time
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hb2  # noqa: E402
from exp import load  # noqa: E402


def get_policy():
    spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
    pol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pol)
    return pol


if __name__ == "__main__":
    Y, A, D, init_s = load()
    pol = get_policy()
    cuts = [int(x) for x in (sys.argv[1:] or ["53", "79", "105", "130", "149"])]
    tot = []
    for c in cuts:
        pol._HISTORY_BASE = c
        t = time.perf_counter()
        r = hb2.backtest(pol.decide, D, start_week=c, initial=init_s)
        el = time.perf_counter() - t
        tot.append(r["score"])
        print("cut%3d: %8.1f  (%.2fs)" % (c, r["score"], el))
    pol._HISTORY_BASE = None
    print("mean: %.1f" % np.mean(tot))
