"""Final verification of the tuned policy (w05 scratch)."""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import backtest_local as hb  # noqa: E402

spec = importlib.util.spec_from_file_location("pol", HERE / "policy.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

print("cfg:", P.P_TARGET, P.P_DEP, P.ALPHA, P.BETA, P.SCALE, P.CENS_IMPUTE)
D = HERE.parents[1] / "data"
sales = pd.read_csv(D / "sales.csv")
y = sales.pivot_table(
    index="series_id", columns="week", values="y", aggfunc="first"
).sort_index(axis=1)
H = np.nan_to_num(y.values.astype(float))
rank = {s: i for i, s in enumerate(y.index)}
perm = np.array([rank[s] for s in sales["series_id"].drop_duplicates().tolist()])
H_sales = H[perm]
ini = pd.read_csv(D / "initial_state.csv").set_index("series_id")
INIT = np.nan_to_num(
    ini.reindex(y.index)[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float),
    nan=0.0,
)

t = time.perf_counter()
tot = 0.0
for cut in (100, 130):
    for nm, init in (("zeros", None), ("init", INIT)):
        r = hb.backtest(P.decide, H, start_week=cut, initial=init)
        tot += r["score"]
        print(f"final cut={cut} init={nm}: {r['score']:.1f}")
print(
    f"sum={tot:.1f}  (ref13 local bar sum was 23081.6)  elapsed {time.perf_counter() - t:.1f}s"
)

# series-order detection trace on the sales-appearance variant, zeros initial
P._LOCK = None
orig = P._select
sel = []


def spy(m, w, st, rev, n):
    mp, pooled = orig(m, w, st, rev, n)
    sel.append(
        [i for i, c in enumerate(m["cands"]) if c is mp][0] if mp is not None else -1
    )
    return mp, pooled


P._select = spy
r = hb.backtest(P.decide, H_sales, start_week=100, initial=None)
print(
    "sales-order zeros-init cut=100:",
    round(r["score"], 1),
    "| candidate idx per decide call:",
    sel,
)
