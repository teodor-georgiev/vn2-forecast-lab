import numpy as np, sys, time
from pathlib import Path
import dev
import hb_fixed as hb
import inventory_sim as sim

Y, AV, ST0 = dev.Y, dev.AV, dev.ST0
N = Y.shape[0]


def week_table(res, label, demand8):
    miss = [r["shortage"] for r in res["weekly"]]
    hold = [r["holding"] for r in res["weekly"]]
    tot = [r["holding"] + r["shortage"] for r in res["weekly"]]
    ordered = [r["ordered"] for r in res["weekly"]]
    print(f"{label}: score={sum(tot[2:]):.0f}")
    print("  wk  demand   hold    short    tot    ordered")
    for i in range(8):
        print(
            f"  {i + 1}  {demand8[i].sum():7.0f} {hold[i]:8.0f} {miss[i]:8.0f} {tot[i]:8.0f} {ordered[i]:8.0f}"
        )


for cut in [100, 130]:
    demand8 = Y[:, cut : cut + 8].T
    print("=" * 70)
    print("cutoff", cut, "demand weeks3-8:", demand8[2:].sum())
    # no order
    res = sim.simulate(demand8, ST0, np.zeros((6, N)))
    week_table(res, "  no-order", demand8)
    # zeros initial
    res = sim.simulate(demand8, np.zeros((N, 3)), np.zeros((6, N)))
    print(f"  no-order, zero init: {res['score']:.0f}")
    sc, dt = dev.anchor(cut)
    # anchor breakdown
    m = Y[:, cut - 13 : cut].mean(axis=1)
    fc = lambda w, s, h: np.tile(m, (3, 1))
    r2 = sim.order_plan_from_policy(demand8, ST0, fc, policy=sim.reference_orders)
    week_table(r2, "  anchor(13wk mean, ST0 init)", demand8)
    r3 = sim.order_plan_from_policy(
        demand8, np.zeros((N, 3)), fc, policy=sim.reference_orders
    )
    print(f"  anchor zero init: {r3['score']:.0f}")
