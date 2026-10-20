"""Diagnose cutoff 100 vs 130; test more adaptive variants."""

import numpy as np
import itertools
import tune
from tune import Y, A, init, Policy, backtest, run_reference

INIT = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)
cuts = [100, 130]

for s in cuts:
    d = Y[:, s : s + 8]
    past = Y[:, s - 26 : s]
    print(
        f"cutoff {s}: game demand sum {d.sum():.0f}, mean {d.mean():.3f} | "
        f"prev 26w mean {past.mean():.3f}, prev8w {Y[:, s - 8 : s].mean():.3f}"
    )

# --- reference with different windows/buffers, real init
print("\n== reference variants ==")
best = []
for W in (8, 13, 26, 40):
    for b in (0.5, 0.75, 1.0, 1.3):
        sc = [
            run_reference(Y, s, window=W, buffer=b, initial=INIT)["score"] for s in cuts
        ]
        best.append((np.mean(sc), W, b, sc))
for m, W, b, sc in sorted(best)[:6]:
    print(
        "  mean %6.0f  W=%d b=%.2f  [%s]" % (m, W, b, " ".join("%.0f" % x for x in sc))
    )

# --- policy grid, wider
print("\n== policy grid ==")
best = []
for W, u, p, k, capq in itertools.product(
    (13, 26), (1.0, 1.2), (0.75, 0.85, 0.95), (6, 10, 16), (0.9, 0.97, 1.05)
):
    cfg = {
        "W": W,
        "u": u,
        "p": p,
        "k": k,
        "seas": False,
        "cap": True,
        "capq": capq,
        "mc": 64,
    }
    sc = [
        Policy(s, cfg) and backtest(Policy(s, cfg).decide, Y, s, initial=INIT)["score"]
        for s in cuts
    ]
    best.append((np.mean(sc), cfg, sc))
for m, cfg, sc in sorted(best, key=lambda x: x[0])[:8]:
    print(
        "  mean %6.0f  %s  [%s]"
        % (
            m,
            " ".join(f"{k}={v}" for k, v in cfg.items()),
            " ".join("%.0f" % x for x in sc),
        )
    )
