"""Final validation for w18.

1. backtest (helper_backtest with the one-line reveal fix) on 4 historical cutoffs
2. parity with the arena's frozen engine tools/inventory_sim.py::order_plan_from_policy
3. full 8-week replay of the REAL game (n=599, H=157, real initial state) with timing
4. reference comparison: 13-week mean + reference_orders (the manager's adaptive bar)
"""

import sys, time
import numpy as np
import pandas as pd

HERE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w18"
ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
sys.path.insert(0, HERE)
sys.path.insert(0, ARENA + r"\tools")
import backtest_fixed as hb
import inventory_sim as sim
import policy

sv = np.load(HERE + r"\sv.npy")
history = sv
CUTOFFS = (100, 115, 130, 145)


def ref_run(cutoffs):
    out = []
    for c in cutoffs:
        tail = history[:, c - 13 : c].mean(axis=1)

        def decide(w, state, revealed, tail=tail):
            return sim.reference_orders(np.stack([tail] * 3), state, 1.0, 1.0, 0.75)

        out.append(hb.backtest(decide, history, start_week=c)["score"])
    return np.array(out)


def pol_run(cutoffs):
    out = []
    for c in cutoffs:
        policy.HISTORY_END = c
        policy._FC_CACHE.clear()
        res = hb.backtest(policy.decide, history, start_week=c)
        out.append(res["score"])
        print("    cutoff %3d -> %8.1f" % (c, res["score"]))
    return np.array(out)


print("== 1. backtest on historical cutoffs (helper_backtest + reveal fix) ==")
r = ref_run(CUTOFFS)
print(
    "  REF 13wk-mean + reference_orders : mean %8.1f  %s"
    % (r.mean(), " ".join("%6.0f" % x for x in r))
)
p = pol_run(CUTOFFS)
print(
    "  POL hierarchical (this policy)     : mean %8.1f  %s"
    % (p.mean(), " ".join("%6.0f" % x for x in p))
)
print(
    "  delta vs reference: %+.1f (%.1f%%)"
    % (p.mean() - r.mean(), 100 * (p.mean() - r.mean()) / r.mean())
)

print("\n== 2. parity with the arena's frozen engine (order_plan_from_policy) ==")
c = 130
policy.HISTORY_END = c
policy._FC_CACHE.clear()
dem = history[:, c : c + 8].T
comp = policy._components(c)
F = policy._assemble(comp)
r_ = policy._dispersion(comp["D"])


def forecast_fn(w, state, history_rows):
    rev = None if history_rows is None else np.asarray(history_rows)
    Fw = policy._adapt(F, w, rev) if rev is not None else F
    return np.stack([Fw[w], Fw[w + 1], Fw[w + 2]])


def pol(forecast, state):
    w = _w[0]
    _w[0] += 1
    rng = np.random.default_rng(1234 + w)
    return policy._orders(np.asarray(forecast), r_, state, w, rng)


_w = [0]
res_frozen = sim.order_plan_from_policy(
    dem, np.zeros((dem.shape[1], 3)), forecast_fn, pol
)
res_mine = hb.backtest(policy.decide, history, start_week=c)
print(
    "  frozen engine score %.2f   helper_backtest score %.2f   diff %.4f"
    % (
        res_frozen["score"],
        res_mine["score"],
        abs(res_frozen["score"] - res_mine["score"]),
    )
)

print("\n== 3. real-game replay (n=599, H=157, real initial state) ==")
init = pd.read_csv(ARENA + r"\data\initial_state.csv")
st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
order = history.shape[0]
policy.HISTORY_END = None
policy._FC_CACHE.clear()
t0 = time.perf_counter()
state = st0.copy()
revealed = []
tot = 0.0
for w in range(8):
    if w < 6:
        rev = np.array(revealed).reshape(w, -1) if w else np.empty((0, order))
        o = np.rint(np.maximum(0.0, policy.decide(w, state.copy(), rev)))
    else:
        o = np.zeros(order)
    on_hand, arr1, arr2 = state.T
    supply = on_hand + arr1
    end = np.maximum(0.0, supply - 0.0)
    state = np.column_stack((end, arr2, o))
    revealed.append(np.zeros(order))
elapsed = time.perf_counter() - t0
print("  full 8-week decision replay (no demand known): %.1f s" % elapsed)
print("  total units ordered: %s" % [int(x.sum()) for x in [o]])

print("\n== 4. selfcheck ==")
import subprocess

r = subprocess.run(
    [sys.executable, "selfcheck.py"], cwd=HERE, capture_output=True, text=True
)
print(r.stdout.strip() or r.stderr.strip())
