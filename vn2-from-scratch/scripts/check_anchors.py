"""Anchor gate: prove the lead-time alignment and initial-state semantics."""

import sys

import numpy as np

from vn2_from_scratch.config import (
    ANCHOR_POLICY_INDEPENDENT_WEEK1,
    ANCHOR_POLICY_INDEPENDENT_WEEK2,
)
from vn2_from_scratch.data import load_dataset
from vn2_from_scratch.simulator import simulate

TOL = 0.011

ds = load_dataset()
print("keys", ds.keys.shape, "history", ds.history.shape, "game", ds.game_sales.shape)
print("in_stock_full", ds.in_stock_full.shape)
print(
    "on_hand0 sum",
    ds.on_hand0.sum(),
    "pipe_w1 sum",
    ds.pipe_w1.sum(),
    "pipe_w2 sum",
    ds.pipe_w2.sum(),
)

first_two = {}
for label, orders in (
    ("orders=0", np.zeros((6, ds.n))),
    ("orders=1e4", np.full((6, ds.n), 1e4)),
):
    r = simulate(orders, ds.game_sales.T, ds.on_hand0, ds.pipe_w1, ds.pipe_w2)
    first_two[label] = r.weekly_total[:2]
    print(f"\n[{label}] weekly_total = {np.round(r.weekly_total, 2).tolist()}")
    print(
        f"   scored(w3-8) = {r.scored_cost:.2f}  "
        f"holding {r.scored_holding:.2f}  shortage {r.scored_shortage:.2f}"
    )

print(
    f"\nANCHOR week1 expected {ANCHOR_POLICY_INDEPENDENT_WEEK1}, "
    f"week2 expected {ANCHOR_POLICY_INDEPENDENT_WEEK2}"
)

rng = np.random.default_rng(0)
w1 = []
for _ in range(5):
    o = rng.integers(0, 200, size=(6, ds.n)).astype(float)
    r = simulate(o, ds.game_sales.T, ds.on_hand0, ds.pipe_w1, ds.pipe_w2)
    w1.append((r.weekly_total[0], r.weekly_total[1]))
w1 = np.array(w1)
spread = float(np.ptp(w1, axis=0).max())
print("policy invariance week1-2 max spread:", spread)

gates = []
for label, weeks in first_two.items():
    gates.append((f"{label} week1", weeks[0], ANCHOR_POLICY_INDEPENDENT_WEEK1))
    gates.append((f"{label} week2", weeks[1], ANCHOR_POLICY_INDEPENDENT_WEEK2))
gates.append(("policy invariance spread", spread, 0.0))

failed = 0
for name, got, want in gates:
    ok = abs(got - want) < TOL
    if not ok:
        failed += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name:<26} got {got:>10.2f} want {want:>10.2f}")

print(f"\nGATE {'FAILED' if failed else 'PASSED'}  ({failed} failing check(s))")
sys.exit(1 if failed else 0)
