import sys
import numpy as np

sys.path.insert(0, ".")
import bt
import policy
from cmp import ref_decide_factory

Y, I = bt.load()
w = Y.sum(0)
print("total weekly demand (last 24 wks):")
for i in range(len(w) - 24, len(w), 4):
    print("  wk", i, " ".join(f"{x:7.0f}" for x in w[i : i + 4]))
ref = ref_decide_factory(Y)
for c in [149, 141, 133]:
    policy._LIMIT = c
    policy._PREP = {}
    r = bt.backtest(policy.decide, Y, start_week=c, initial=I)
    rr = bt.backtest(ref, Y, start_week=c, initial=I)
    print(f"cutoff {c} policy {r['score']:.0f} ref {rr['score']:.0f}")
    for a, b in zip(r["weekly"], rr["weekly"]):
        print(
            f"   w{a['week']} policy hold {a['holding']:7.1f} short {a['shortage']:7.1f}"
            f" ord {a['ordered']:6d}  | ref hold {b['holding']:7.1f} short {b['shortage']:7.1f}"
            f" ord {b['ordered']:6d}"
        )
