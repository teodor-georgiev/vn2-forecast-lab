"""Dev: focused second-stage grid (6 cutoffs)."""

from __future__ import annotations
import numpy as np
import dev_grid2 as g

CUTS6 = [70, 90, 105, 120, 135, 148]


def run(cfgs, cuts=CUTS6, tag=""):
    g.CUTS = cuts
    print(f"--- {tag} on {cuts}")
    with open("results.txt", "a") as fh:
        fh.write(f"--- {tag} cuts={cuts}\n")
    g.sweep(cfgs, cutoffs=cuts)


if __name__ == "__main__":
    b = g.BASE
    run([("ref-ish q0.7", {"q_mode": "fixed", "q": 0.7})], tag="base")
    cfgs = []
    for q in (0.6, 0.65, 0.7, 0.75, 0.8):
        for pw in (0.6, 0.8, 1.0):
            cfgs.append((f"q{q}_pw{pw}", {"q_mode": "fixed", "q": q, "proj_w": pw}))
    run(cfgs, tag="q x proj_w")
    run(
        [
            (f"lev_b{b}", {"q_mode": "level", "q_beta": b, "q_lo": 0.05, "q_hi": 0.99})
            for b in (6.0, 8.0, 12.0, 20.0)
        ],
        tag="level q_beta big",
    )
    run(
        [
            (f"L{L}", {"q_mode": "fixed", "q": 0.7, "hist_weeks_level": L})
            for L in (8, 10, 13, 18, 26)
        ],
        tag="L",
    )
    run(
        [(f"m{m}", {"q_mode": "fixed", "q": 0.7, "m_k": m}) for m in (0.3, 1.0, 5.0)],
        tag="m_k",
    )
    run(
        [
            (f"sp{s}", {"q_mode": "fixed", "q": 0.7, "seasonal_power": s})
            for s in (0.8, 1.0, 1.3)
        ],
        tag="seasonal power",
    )
