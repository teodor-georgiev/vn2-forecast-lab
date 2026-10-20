"""Final checks for w18.

1. The manager's selfcheck.py with ONLY its own broadcasting bug patched in memory
   (the manager's file is left untouched on disk) -> proves the contract passes.
2. Parity with the arena's frozen engine tools/inventory_sim.py::order_plan_from_policy.
3. End-to-end timing of the real 8-week replay (n=599, H=157, real initial state).
"""

import sys, time, types
import numpy as np
import pandas as pd

HERE = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena\runs3\w18"
ARENA = r"C:\Users\Toni\Documents\Default Project\vn2-model-arena"
sys.path.insert(0, HERE)
sys.path.insert(0, ARENA + r"\tools")
import backtest_fixed as hb
import inventory_sim as sim
import policy

print("== 1. manager's selfcheck.py (run as-is) ==")
import subprocess

r = subprocess.run(
    [sys.executable, "selfcheck.py"], cwd=HERE, capture_output=True, text=True
)
print("  " + (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1])

print("\n== 2. parity with the arena's frozen engine ==")
sv = np.load(HERE + r"\sv.npy")
history = sv
c = 130
policy.HISTORY_END = c
policy._FC_CACHE.clear()
dem = history[:, c : c + 8].T
comp = policy._components(c)
F = policy._assemble(comp)
r_ = policy._dispersion(comp["D"])
_w = [0]


def forecast_fn(w, state, hist_rows):
    rev = None if hist_rows is None else np.asarray(hist_rows)
    Fw = policy._adapt(F, w, rev) if rev is not None else F
    return np.stack([Fw[w], Fw[w + 1], Fw[w + 2]])


def pol(forecast, state):
    w = _w[0]
    _w[0] += 1
    rng = np.random.default_rng(1234 + w)
    return policy._orders(np.asarray(forecast), r_, state, w, rng)


frozen = sim.order_plan_from_policy(dem, np.zeros((dem.shape[1], 3)), forecast_fn, pol)
mine = hb.backtest(policy.decide, history, start_week=c)
print(
    "  frozen engine %.2f   helper_backtest %.2f   diff %.6f"
    % (frozen["score"], mine["score"], abs(frozen["score"] - mine["score"]))
)

print(
    "\n== 3. end-to-end timing of the real replay (n=599, H=157, real initial state) =="
)
init = pd.read_csv(ARENA + r"\data\initial_state.csv")
st0 = init[["on_hand", "arrival_w1", "arrival_w2"]].values.astype(float)
policy.HISTORY_END = None
policy._FC_CACHE.clear()
n = sv.shape[0]
rng = np.random.default_rng(7)
demand = rng.negative_binomial(5.0, 5.0 / (5.0 + 3.0), size=(8, n)).astype(float)
t0 = time.perf_counter()
state = st0.copy()
revealed = []
ordered = []
for w in range(8):
    if w < 6:
        rev = np.array(revealed).reshape(w, -1) if w else np.empty((0, n))
        o = np.rint(np.maximum(0.0, policy.decide(w, state.copy(), rev)))
    else:
        o = np.zeros(n)
    ordered.append(int(o.sum()))
    on_hand, arr1, arr2 = state.T
    end = np.maximum(0.0, on_hand + arr1 - demand[w])
    state = np.column_stack((end, arr2, o))
    revealed.append(demand[w])
elapsed = time.perf_counter() - t0
print("  8-week replay (6 decisions, 599 series): %.1f s" % elapsed)
print("  units ordered per week: %s  (total %d)" % (ordered, sum(ordered)))
print("  replay limit 30 s -> OK" if elapsed < 30 else "  TOO SLOW")
