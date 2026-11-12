"""Reproduction gate: official organizer benchmark must equal EUR 4,334.00."""

import sys

import numpy as np

from vn2_from_scratch.config import (
    ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8,
    ANCHOR_ORGANIZER_HOLDING,
    ANCHOR_ORGANIZER_SHORTAGE,
    ANCHOR_ORGANIZER_WEEKLY_SCORED,
    ANCHOR_POLICY_INDEPENDENT_WEEK1,
    ANCHOR_POLICY_INDEPENDENT_WEEK2,
)
from vn2_from_scratch.data import load_dataset
from vn2_from_scratch.organizer import run_organizer_benchmark

ds = load_dataset()
orders, res = run_organizer_benchmark(ds)

print("total (all 8 weeks) :", round(res.total_cost, 2))
print("  holding           :", round(res.holding_cost, 2))
print("  shortage          :", round(res.shortage_cost, 2))
print("scored weeks 3-8    :", round(res.scored_cost, 2))
print("  scored holding    :", round(res.scored_holding, 2))
print("  scored shortage   :", round(res.scored_shortage, 2))
print("weekly total        :", np.round(res.weekly_total, 2).tolist())
print("total order units   :", int(orders.sum()))

checks = [
    ("week1 policy-independent", res.weekly_total[0], ANCHOR_POLICY_INDEPENDENT_WEEK1),
    ("week2 policy-independent", res.weekly_total[1], ANCHOR_POLICY_INDEPENDENT_WEEK2),
    ("scored weeks 3-8", res.scored_cost, ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8),
    ("holding total", res.holding_cost, ANCHOR_ORGANIZER_HOLDING),
    ("shortage total", res.shortage_cost, ANCHOR_ORGANIZER_SHORTAGE),
]
failed = 0
for i, (name, got, want) in enumerate(checks):
    ok = abs(got - want) < 0.011
    if not ok:
        failed += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name:<26} got {got:>10.2f} want {want:>10.2f}")

prof = np.allclose(res.weekly_total[2:], ANCHOR_ORGANIZER_WEEKLY_SCORED, atol=0.011)
if not prof:
    failed += 1
print(f"{'PASS' if prof else 'FAIL'}  weekly profile matches reference")
print("\nREFERENCE weekly scored:", list(ANCHOR_ORGANIZER_WEEKLY_SCORED))
print("MEASURED   weekly scored:", np.round(res.weekly_total[2:], 2).tolist())

print(f"\nGATE {'FAILED' if failed else 'PASSED'}  ({failed} failing check(s))")
sys.exit(1 if failed else 0)
