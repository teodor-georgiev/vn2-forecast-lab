"""Derive the exact scored-week holding and shortage targets for the benchmark."""

import numpy as np

from vn2_from_scratch.config import (
    ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8,
    ANCHOR_ORGANIZER_HOLDING,
    ANCHOR_ORGANIZER_SHORTAGE,
    ANCHOR_ORGANIZER_WEEKLY_SCORED,
)
from vn2_from_scratch.data import load_dataset
from vn2_from_scratch.simulator import simulate

ds = load_dataset()
r0 = simulate(np.zeros((6, ds.n)), ds.game_sales.T, ds.on_hand0, ds.pipe_w1, ds.pipe_w2)

print("orders=0 weekly holding :", np.round(r0.weekly_holding, 2).tolist())
print("orders=0 weekly shortage:", np.round(r0.weekly_shortage, 2).tolist())

h12 = float(r0.weekly_holding[:2].sum())
s12 = float(r0.weekly_shortage[:2].sum())
print(
    f"\npolicy-independent weeks 1-2: holding {h12:.2f}  shortage {s12:.2f}  total {h12 + s12:.2f}"
)

target_h = ANCHOR_ORGANIZER_HOLDING - h12
target_s = ANCHOR_ORGANIZER_SHORTAGE - s12
print(f"\nDERIVED scored-week targets:")
print(f"  scored holding  = {target_h:.2f}")
print(f"  scored shortage = {target_s:.2f}")
print(f"  sum             = {target_h + target_s:.2f}")
print(f"  anchor scored   = {ANCHOR_ORGANIZER_BASELINE_WEEKS_3_8:.2f}")
print(f"  profile sum     = {sum(ANCHOR_ORGANIZER_WEEKLY_SCORED):.2f}")

print(f"\nshortage share of scored cost = {target_s / (target_h + target_s):.4f}")
print(
    "units: scored shortage units =", target_s, "| scored held units =", target_h / 0.2
)
