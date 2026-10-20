import numpy as np
import tune
from tune import Y, init, Policy, backtest, run_reference

cuts = [100, 130]
INIT = init[["on_hand", "arrival_w1", "arrival_w2"]].to_numpy(float)

print("zero-init:")
for W in (13, 26):
    sc = [run_reference(Y, s, window=W, initial=None)["score"] for s in cuts]
    print(
        "  ref W=%d: [%s] mean %.0f"
        % (W, " ".join("%.0f" % x for x in sc), np.mean(sc))
    )
for cfg in [
    {
        "W": 26,
        "u": 1.0,
        "p": 0.85,
        "k": 6,
        "seas": False,
        "cap": True,
        "capq": 0.95,
        "mc": 64,
    },
    {
        "W": 26,
        "u": 1.0,
        "p": 0.85,
        "k": 10,
        "seas": False,
        "cap": True,
        "capq": 0.95,
        "mc": 64,
    },
    {
        "W": 13,
        "u": 1.0,
        "p": 0.85,
        "k": 6,
        "seas": False,
        "cap": True,
        "capq": 0.95,
        "mc": 64,
    },
]:
    sc = [backtest(Policy(s, cfg).decide, Y, s, initial=None)["score"] for s in cuts]
    print(
        "  pol %s: [%s] mean %.0f"
        % (cfg["W"], " ".join("%.0f" % x for x in sc), np.mean(sc))
    )

print("real-init, extra cutoffs 115/145:")
for cfg in [
    {
        "W": 26,
        "u": 1.0,
        "p": 0.85,
        "k": 6,
        "seas": False,
        "cap": True,
        "capq": 0.95,
        "mc": 64,
    },
    {
        "W": 26,
        "u": 1.0,
        "p": 0.85,
        "k": 10,
        "seas": False,
        "cap": True,
        "capq": 0.95,
        "mc": 64,
    },
]:
    sc = [
        backtest(Policy(s, cfg).decide, Y, s, initial=INIT)["score"] for s in [115, 145]
    ]
    ref = [run_reference(Y, s, window=26, initial=INIT)["score"] for s in [115, 145]]
    print(
        "  pol k=%d: [%s] | ref [%s]"
        % (
            cfg["k"],
            " ".join("%.0f" % x for x in sc),
            " ".join("%.0f" % x for x in ref),
        )
    )
