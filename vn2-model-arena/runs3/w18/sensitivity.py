"""Final sensitivity: seasonal weight and censoring correction (documented choices)."""

import sys, time
import numpy as np

HERE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w18"
sys.path.insert(0, HERE)
import backtest_fixed as hb
import policy

sv = np.load(HERE + r"\sv.npy")
history = sv
CUT = (100, 115, 130, 145)


def run(tag, **kw):
    for k, v in kw.items():
        setattr(policy, k, v)
    sc = []
    for c in CUT:
        policy.HISTORY_END = c
        policy._FC_CACHE.clear()
        sc.append(hb.backtest(policy.decide, history, start_week=c)["score"])
    print(
        "  %-42s mean %8.1f  %s" % (tag, np.mean(sc), " ".join("%6.0f" % x for x in sc))
    )
    return np.mean(sc)


print("backtest on cutoffs", CUT, "(zero initial state, raw sales as demand)")
run("FINAL (seas 0.25, censoring correction ON)")
run("seas_w = 0.0", SEAS_W=0.0)
run("seas_w = 0.40", SEAS_W=0.40)
run("fractile 0.65", FRACTILE=0.65)
run("fractile 0.75", FRACTILE=0.75)
run("level window 13", LEVEL_K=13)
run("level window 39", LEVEL_K=39)
run("no family pooling (FAMILY_W=0)", FAMILY_W=0.0)
run("family pooling 0.40", FAMILY_W=0.40)

# censoring correction OFF (diagnostic only)
orig_correct = policy._correct
policy._correct = lambda sv_, av_, H: sv_[:, :H].copy()
run("no censoring correction", SEAS_W=0.25, FAMILY_W=0.25, LEVEL_K=26, FRACTILE=0.70)
policy._correct = orig_correct
print("\n(the last row is a diagnostic: the real game's demand is unconstrained,")
print(" so the censoring correction should be kept even though the backtest,")
print(" which replays censored sales, penalises it)")
