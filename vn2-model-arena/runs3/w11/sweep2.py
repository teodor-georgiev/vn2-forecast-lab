"""Focused structure sweep around the current best config."""

import numpy as np
from sweep import grid, CUTS

BASE = dict(
    win_short=13,
    w_recent=0.7,
    use_trend=False,
    seas_mode="blend",
    q_A=0.70,
    q_B=0.60,
    q_C=0.80,
)

if __name__ == "__main__":
    r1 = grid(
        BASE,
        dict(seas_k=[2.0, 4.0, 8.0, 16.0], flat_C=[False, True], c_win=[None, 26, 52]),
        top=8,
        label="seas/dispersion structure",
    )
    best = r1[0][3]
    print("best structure:", best)
    b2 = dict(BASE)
    b2.update(best)
    r2 = grid(
        b2,
        dict(q_A=[0.55, 0.65, 0.75], q_B=[0.45, 0.55, 0.65], q_C=[0.7, 0.8, 0.9]),
        top=8,
        label="quantiles refine",
    )
    b3 = dict(b2)
    b3.update(r2[0][3])
    r3 = grid(
        b3,
        dict(
            win_short=[8, 10, 13, 20],
            w_recent=[0.5, 0.7, 0.9],
            a_thr=[2.0, 3.0, 5.0],
            c_thr=[0.5, 1.0, 1.5],
        ),
        top=8,
        label="level/class thresholds",
    )
    print("FINAL:", dict(b3, **r3[0][3]))
