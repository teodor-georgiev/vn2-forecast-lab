"""Weekly diagnostics + order-multiplier probe."""

from __future__ import annotations

import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hb_fixed as hb
from work import Y, D

cutoff = int(sys.argv[1]) if len(sys.argv) > 1 else 100
truth = sys.argv[2] if len(sys.argv) > 2 else "raw"
mult = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
os.environ["VN2_CUTOFF"] = str(cutoff)
sys.modules.pop("policy", None)
import policy

h = Y if truth == "raw" else D
n = h.shape[0]
base = policy.decide


def decide(w, state, revealed):
    q = base(w, state, revealed)
    return q * mult


res = hb.backtest(decide, h, cutoff)
dem = h[:, cutoff : cutoff + 8]
print(f"cutoff={cutoff} truth={truth} mult={mult} score={res['score']:.0f}")
for i, wk in enumerate(res["weekly"]):
    print(
        f"  wk{i + 1}: demand={dem[:, i].sum():8.0f} ordered={wk['ordered']:8d} "
        f"hold={wk['holding']:7.0f} short={wk['shortage']:7.0f} "
        f"(scored={'Y' if i >= 2 else 'n'})"
    )
scored = res["weekly"][2:]
print(
    f"  TOTAL scored hold={sum(x['holding'] for x in scored):.0f} "
    f"short={sum(x['shortage'] for x in scored):.0f} "
    f"demand(w3-8)={dem[:, 2:].sum():.0f} ordered={sum(x['ordered'] for x in res['weekly']):.0f}"
)
